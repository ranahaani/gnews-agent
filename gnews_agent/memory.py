"""Public :class:`NewsMemory` facade.

Stage 1 wires ingestion and stats. Query/AI/MCP land in later stages and are
left as ``NotImplementedError`` stubs that point at the responsible stage.
"""
from __future__ import annotations

import datetime as _dt
import logging
import sqlite3
import time
from typing import Any

from gnews_agent.ai.client import LLMClient, LLMConfig
from gnews_agent.ai.sentiment import score_sentiment, score_sentiment_timeline
from gnews_agent.ai.summarizer import make_brief
from gnews_agent.config import NewsMemoryConfig
from gnews_agent.exceptions import LLMKeyMissingError
from gnews_agent.ingestion.embedder import Embedder, make_embedder
from gnews_agent.ingestion.enricher import enriched_text
from gnews_agent.ingestion.fetcher import Fetcher
from gnews_agent.query import blend, parse_date, recency_score
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
        llm_client: LLMClient | None = None,
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
        self._llm = llm_client  # built lazily by _get_llm()

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

    # ------------------------------------------------------------------
    # query
    # ------------------------------------------------------------------

    def search(
        self,
        query: str,
        *,
        days: int | None = None,
        country: str | None = None,
        language: str | None = None,
        semantic: bool = True,
        limit: int = 10,
    ) -> list[dict[str, Any]]:
        """Semantic search with metadata filter + recency-blended re-rank.

        Falls back to SQLite FTS5 keyword search when ``semantic=False`` or
        when the configured vector store reports zero rows (cold cache).
        """
        since_iso = self._since_iso(days)
        wants_vector = semantic and self._vectors.count() > 0
        if not wants_vector:
            rows = self._sqlite.fts_search(
                query,
                limit=limit,
                country=country,
                language=language,
                since_iso=since_iso,
            )
            return [
                {**self._row_to_article(row), "score": 0.0, "search_mode": "keyword"}
                for row in rows
            ]

        embedding = self._embedder.embed([query])[0]
        where: dict[str, Any] = {}
        if country:
            where["country"] = country
        if language:
            where["language"] = language
        # Fetch a wider candidate window so post-filter still hits ``limit``.
        candidates = self._vectors.query(embedding, k=max(limit * 3, limit), where=where or None)
        ids = [hit.article_id for hit in candidates]
        articles = self._sqlite.get_articles(ids)

        results: list[dict[str, Any]] = []
        for hit in candidates:
            row = articles.get(hit.article_id)
            if row is None:
                continue
            if since_iso and (row.get("published_at") or "") < since_iso:
                continue
            published = parse_date(row.get("published_at") or row.get("published_date"))
            score = blend(hit.score, recency_score(published))
            results.append(
                {**self._row_to_article(row), "score": score, "search_mode": "semantic"}
            )

        results.sort(key=lambda r: r["score"], reverse=True)
        return results[:limit]

    def timeline(
        self,
        topic: str | None = None,
        *,
        days: int | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> list[dict[str, Any]]:
        """Day-by-day article counts for ``topic`` (keyless, SQL only)."""
        start_iso = start or self._since_iso(days)
        end_iso = end
        return self._sqlite.timeline(topic, start_iso=start_iso, end_iso=end_iso)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _since_iso(days: int | None) -> str | None:
        if days is None:
            return None
        cutoff = _dt.datetime.utcnow() - _dt.timedelta(days=days)
        return cutoff.strftime("%Y-%m-%d")

    @staticmethod
    def _row_to_article(row: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": row["id"],
            "title": row["title"],
            "url": row["url"],
            "publisher": row.get("publisher_name"),
            "published_date": row.get("published_date"),
            "summary": row.get("summary"),
            "country": row.get("country"),
            "language": row.get("language"),
            "topic": row.get("topic"),
        }

    # ------------------------------------------------------------------
    # AI layer
    # ------------------------------------------------------------------

    def brief(
        self,
        topic: str,
        *,
        days: int = 7,
        max_articles: int = 20,
        include_citations: bool = True,
    ) -> dict[str, Any]:
        """Cited brief on ``topic`` for the last ``days`` days.

        Requires an LLM key — raises :class:`LLMKeyMissingError` otherwise.
        v1 ships without story clustering; articles are passed in
        recency-blended search order.
        """
        articles = self.search(topic, days=days, limit=max_articles)
        if not articles:
            return {
                "summary": f"No articles for {topic!r} in the last {days} days.",
                "citations": [],
                "sentiment": "neutral",
                "article_count": 0,
            }
        llm = self._get_llm()
        result = make_brief(llm, topic=topic, days=days, articles=articles)
        if not include_citations:
            result.pop("citations", None)
        return result

    def sentiment(
        self,
        topic: str,
        *,
        days: int = 14,
        timeline: bool = False,
    ) -> dict[str, Any]:
        """Sentiment over the corpus for ``topic`` — optional day-by-day breakdown."""
        articles = self.search(topic, days=days, limit=50)
        llm = self._get_llm()
        if timeline:
            return {
                "overall": _aggregate_overall(score_sentiment(llm, topic=topic, articles=articles)),
                "timeline": score_sentiment_timeline(llm, topic=topic, articles=articles),
                "article_count": len(articles),
            }
        return score_sentiment(llm, topic=topic, articles=articles)

    def _get_llm(self) -> LLMClient:
        if self._llm is not None:
            return self._llm
        if not self.config.llm_provider or not self.config.llm_model:
            raise LLMKeyMissingError(
                "brief()/sentiment() require llm_provider + llm_model on NewsMemoryConfig. "
                "Set them explicitly or pass an LLMClient when constructing NewsMemory."
            )
        self._llm = LLMClient(LLMConfig(
            provider=self.config.llm_provider,
            model=self.config.llm_model,
        ))
        return self._llm

    def monitor(self, topics: list[str], **kwargs: Any) -> dict[str, Any]:
        raise NotImplementedError("NewsMemory.monitor lands alongside the MCP server in Stage 5")

    def export(self, path: str, **kwargs: Any) -> str:
        raise NotImplementedError("NewsMemory.export is post-v1")

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        self._sqlite.close()


def _aggregate_overall(corpus_level: dict[str, Any]) -> dict[str, Any]:
    """Pluck the corpus-wide verdict for the timeline mode's ``overall`` slot."""
    return {
        "label": corpus_level.get("overall", "neutral"),
        "score": corpus_level.get("score", 0.0),
        "rationale": corpus_level.get("rationale"),
    }
