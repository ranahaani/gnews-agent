"""GNews 0.8.2 wrapper with per-topic spacing and rate-limit cooldown.

GNews 0.8.2 already retries 429s with capped exponential backoff + jitter
(see ``GNews.max_retries``). This wrapper layers two things on top:

1. **Per-topic spacing** — never call GNews more than once per second for a
   given topic within the same fetcher instance. Cheap politeness above the
   library's own retry policy.
2. **Topic cooldown on retry exhaustion** — when GNews finally raises
   ``RateLimitError`` after exhausting its retries, freeze that topic for the
   remainder of the run instead of looping over it again.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

from gnews import GNews
from gnews.exceptions import RateLimitError


logger = logging.getLogger(__name__)


@dataclass
class FetchResult:
    topic: str
    method: str
    articles: list[dict[str, Any]] = field(default_factory=list)
    rate_limited: bool = False
    error: str | None = None


class Fetcher:
    """Topic-aware wrapper around GNews.

    The wrapper holds last-fetch timestamps per topic and a cooldown set of
    topics that have already exhausted retries this run. Instantiate one per
    ingestion run (or per ``NewsMemory``); state is intentionally not shared
    across processes.
    """

    def __init__(
        self,
        *,
        language: str = "en",
        country: str = "US",
        max_results: int = 50,
        min_interval_seconds: float = 1.0,
        gnews_client: GNews | None = None,
    ) -> None:
        # Keep max_results <= 100 by default — GNews's >100 pagination walk
        # discards date filters (see 0.8.2 _get_news_more_than_100 docstring).
        self._client = gnews_client or GNews(
            language=language,
            country=country,
            max_results=max_results,
        )
        self._min_interval = min_interval_seconds
        self._last_fetch_at: dict[str, float] = {}
        self._cooldown: set[str] = set()

    # ------------------------------------------------------------------
    # cooldown helpers (exposed so tests + the MCP layer can introspect)
    # ------------------------------------------------------------------

    def is_cooled_down(self, topic: str) -> bool:
        return topic in self._cooldown

    def cool_down(self, topic: str) -> None:
        self._cooldown.add(topic)

    # ------------------------------------------------------------------
    # core fetch
    # ------------------------------------------------------------------

    def _wait_for_topic(self, topic: str) -> None:
        last = self._last_fetch_at.get(topic)
        if last is None:
            return
        wait = self._min_interval - (time.monotonic() - last)
        if wait > 0:
            time.sleep(wait)

    def fetch(self, topic: str, *, method: str = "get_news") -> FetchResult:
        if self.is_cooled_down(topic):
            logger.info("skipping topic %r — cooldown active this run", topic)
            return FetchResult(topic=topic, method=method, rate_limited=True)

        self._wait_for_topic(topic)
        result = FetchResult(topic=topic, method=method)
        try:
            articles = self._dispatch(topic, method)
            self._last_fetch_at[topic] = time.monotonic()
            result.articles = [self._normalise(a, topic) for a in articles]
        except RateLimitError as exc:
            logger.warning("topic %r rate-limited after GNews retries: %s", topic, exc)
            self.cool_down(topic)
            result.rate_limited = True
            result.error = str(exc)
        return result

    def _dispatch(self, topic: str, method: str) -> list[dict[str, Any]]:
        if method == "get_news":
            return self._client.get_news(topic)
        if method == "get_top_news":
            return self._client.get_top_news()
        if method == "get_news_by_topic":
            return self._client.get_news_by_topic(topic)
        if method == "get_news_by_location":
            return self._client.get_news_by_location(topic)
        if method == "get_news_by_site":
            return self._client.get_news_by_site(topic)
        raise ValueError(f"unknown GNews method: {method!r}")

    @staticmethod
    def _normalise(raw: dict[str, Any], topic: str) -> dict[str, Any]:
        publisher = raw.get("publisher") or {}
        if isinstance(publisher, dict):
            publisher_name = publisher.get("title") or publisher.get("href")
            publisher_href = publisher.get("href")
        else:
            publisher_name = str(publisher)
            publisher_href = None
        return {
            "title": raw.get("title"),
            "url": raw.get("url"),
            "summary": raw.get("description"),
            "published_date": raw.get("published date") or raw.get("published_date"),
            "publisher_name": publisher_name,
            "publisher_href": publisher_href,
            "topic": topic,
        }
