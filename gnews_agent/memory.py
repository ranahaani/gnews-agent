"""Public :class:`NewsMemory` facade.

Stage 1 wires ingestion and stats. Query/AI/MCP land in later stages and are
left as ``NotImplementedError`` stubs that point at the responsible stage.
"""
from __future__ import annotations

import logging
import sqlite3
import time
from typing import Any

from gnews_agent.config import NewsMemoryConfig
from gnews_agent.ingestion.embedder import Embedder, make_embedder
from gnews_agent.ingestion.enricher import enriched_text
from gnews_agent.ingestion.fetcher import Fetcher
from gnews_agent.storage.sqlite_store import SqliteStore
from gnews_agent.storage.vector_store import VectorStore, make_vector_store


logger = logging.getLogger(__name__)


class NewsMemory:
    """Persistent semantic memory over GNews articles."""

    def __init__(
        self,
        config: NewsMemoryConfig | None = None,
        *,
        fetcher: Fetcher | None = None,
        embedder: Embedder | None = None,
        sqlite_store: SqliteStore | None = None,
        vector_store: VectorStore | None = None,
        **overrides: Any,
    ) -> None:
        self.config = config or NewsMemoryConfig(**overrides)
        self.config.ensure_dirs()

        self._sqlite = sqlite_store or SqliteStore(self.config.db_path)
        self._embedder = embedder or make_embedder(
            backend=self.config.embed_backend,
            embed_model=self.config.embed_model,
            embed_dim=self.config.embed_dim,
        )
        self._vectors = vector_store or make_vector_store(
            self.config.vector_backend,
            persist_path=str(self.config.vector_path),
            embed_model=self._embedder.embed_model,
            embed_dim=self._embedder.embed_dim,
        )
        self._fetcher = fetcher or Fetcher(
            language=self.config.language,
            country=self.config.country,
            max_results=self.config.max_fetch_results,
            min_interval_seconds=self.config.fetch_min_interval_seconds,
        )

    # ------------------------------------------------------------------
    # ingestion
    # ------------------------------------------------------------------

    def ingest(
        self,
        topics: str | list[str],
        *,
        method: str = "get_news",
    ) -> dict[str, dict[str, int]]:
        """Fetch + dedup + embed + store articles for one or many topics.

        Returns ``{topic: {fetched, new, skipped, status}}``.
        """
        if isinstance(topics, str):
            topics = [topics]

        summary: dict[str, dict[str, int]] = {}
        for topic in topics:
            summary[topic] = self._ingest_topic(topic, method=method)
        return summary

    def _ingest_topic(self, topic: str, *, method: str) -> dict[str, int | str]:
        started = time.monotonic()
        run_id = self._sqlite.start_crawl_run(topic=topic, method=method)

        try:
            result = self._fetcher.fetch(topic, method=method)
        except Exception as exc:
            self._sqlite.finish_crawl_run(
                run_id,
                fetched=0,
                new_articles=0,
                skipped_dupes=0,
                status="failed",
                error_message=str(exc),
                duration_seconds=time.monotonic() - started,
            )
            raise

        if result.rate_limited:
            self._sqlite.finish_crawl_run(
                run_id,
                fetched=0,
                new_articles=0,
                skipped_dupes=0,
                status="rate_limited",
                error_message=result.error,
                duration_seconds=time.monotonic() - started,
            )
            return {"fetched": 0, "new": 0, "skipped": 0, "status": "rate_limited"}

        new_count = 0
        skipped = 0
        for raw in result.articles:
            if not raw.get("title") or not raw.get("url"):
                continue
            raw.setdefault("country", self.config.country)
            raw.setdefault("language", self.config.language)
            raw["embed_model"] = self._embedder.embed_model
            raw["embed_dim"] = self._embedder.embed_dim

            if self._sqlite.is_duplicate(raw["title"], raw.get("publisher_name")):
                skipped += 1
                self._sqlite.record_seen(raw["title"], raw.get("publisher_name"), None)
                continue

            try:
                article_id = self._sqlite.insert_article(raw)
            except sqlite3.IntegrityError:
                # url_hash UNIQUE backstop fired — a different headline shares the URL.
                skipped += 1
                continue

            text = enriched_text(raw)
            embedding = self._embedder.embed([text])[0]
            self._vectors.upsert(
                article_id,
                embedding,
                metadata={
                    "topic": raw.get("topic"),
                    "country": raw.get("country"),
                    "language": raw.get("language"),
                    "publisher": raw.get("publisher_name"),
                    "published_date": raw.get("published_date"),
                    "title": raw.get("title"),
                    "url": raw.get("url"),
                    "document": text,
                },
            )
            self._sqlite.record_seen(raw["title"], raw.get("publisher_name"), article_id)
            new_count += 1

        self._sqlite.finish_crawl_run(
            run_id,
            fetched=len(result.articles),
            new_articles=new_count,
            skipped_dupes=skipped,
            status="success",
            duration_seconds=time.monotonic() - started,
        )
        return {
            "fetched": len(result.articles),
            "new": new_count,
            "skipped": skipped,
            "status": "success",
        }

    # ------------------------------------------------------------------
    # introspection
    # ------------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        total = self._sqlite.count_articles()
        vector_count = self._vectors.count()
        return {
            "total_articles": total,
            "vector_count": vector_count,
            "embed_model": self._embedder.embed_model,
            "embed_dim": self._embedder.embed_dim,
        }

    # ------------------------------------------------------------------
    # later-stage stubs — fail loudly until wired up
    # ------------------------------------------------------------------

    def search(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        raise NotImplementedError("NewsMemory.search lands in Stage 2")

    def timeline(self, topic: str, **kwargs: Any) -> list[dict[str, Any]]:
        raise NotImplementedError("NewsMemory.timeline lands in Stage 2")

    def brief(self, topic: str, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("NewsMemory.brief lands in Stage 3")

    def sentiment(self, topic: str, **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("NewsMemory.sentiment lands in Stage 3")

    def monitor(self, topics: list[str], **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("NewsMemory.monitor lands alongside the MCP server in Stage 5")

    def export(self, path: str, **kwargs: Any) -> str:
        raise NotImplementedError("NewsMemory.export is post-v1")

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._sqlite.close()
