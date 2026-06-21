"""SQLite-backed metadata store.

Pure stdlib ``sqlite3`` — no ORM. The schema is shipped as ``schema.sql`` and
applied idempotently at connect time so ``pip install gnews-agent`` users get
a working DB on first call without a migration step.
"""
from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from importlib.resources import files
from pathlib import Path
from typing import Any, Iterator

from gnews_agent.ingestion.deduplicator import composite_key, url_hash
from gnews_agent.query import parse_date


class SqliteStore:
    """Thin facade over ``sqlite3`` for the article + dedup + crawl tables.

    The store does not enforce its own schema version yet — Stage 1 ships a
    single schema, additive migrations land later behind a ``schema_version``
    sentinel table.
    """

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        self._apply_schema()

    def _apply_schema(self) -> None:
        schema_sql = files("gnews_agent.storage").joinpath("schema.sql").read_text(encoding="utf-8")
        self._conn.executescript(schema_sql)
        self._conn.commit()

    def close(self) -> None:
        self._conn.close()

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        try:
            yield self._conn
            self._conn.commit()
        except Exception:
            self._conn.rollback()
            raise

    # ---- dedup -----------------------------------------------------------

    def is_duplicate(self, title: str, publisher: str | None) -> bool:
        key = composite_key(title, publisher)
        row = self._conn.execute(
            "SELECT 1 FROM dedup_index WHERE composite_key = ? LIMIT 1",
            (key,),
        ).fetchone()
        return row is not None

    def record_seen(self, title: str, publisher: str | None, article_id: int | None) -> None:
        key = composite_key(title, publisher)
        with self.transaction() as conn:
            conn.execute(
                """
                INSERT INTO dedup_index (composite_key, title_slug, publisher_norm, article_id)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(composite_key) DO UPDATE SET
                    seen_count = seen_count + 1,
                    last_seen  = CURRENT_TIMESTAMP
                """,
                (key, _title_slug(title), _publisher_norm(publisher), article_id),
            )

    # ---- articles --------------------------------------------------------

    def insert_article(self, article: dict[str, Any]) -> int:
        """Insert an article row. Returns the new row id.

        Raises ``sqlite3.IntegrityError`` on ``url_hash`` collision — the
        caller (ingestion pipeline) should treat that as a dedup hit.
        """
        article_url_hash = url_hash(article["url"])
        published_iso = self._iso_from(article.get("published_date"))
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO articles (
                    title, url, url_hash, publisher_name, publisher_href,
                    published_date, published_at, summary, full_text,
                    country, language, topic, embed_model, embed_dim
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article["title"],
                    article["url"],
                    article_url_hash,
                    article.get("publisher_name"),
                    article.get("publisher_href"),
                    article.get("published_date"),
                    published_iso,
                    article.get("summary"),
                    article.get("full_text"),
                    article.get("country"),
                    article.get("language"),
                    article.get("topic"),
                    article["embed_model"],
                    article["embed_dim"],
                ),
            )
        return cur.lastrowid

    @staticmethod
    def _iso_from(raw: str | None) -> str | None:
        parsed = parse_date(raw)
        return parsed.strftime("%Y-%m-%dT%H:%M:%S") if parsed else None

    def get_article(self, article_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM articles WHERE id = ?", (article_id,)
        ).fetchone()
        return dict(row) if row else None

    def count_articles(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM articles").fetchone()
        return int(row["n"])

    def get_articles(self, ids: list[int]) -> dict[int, dict[str, Any]]:
        """Hydrate the given article ids in a single round-trip."""
        if not ids:
            return {}
        placeholders = ",".join("?" for _ in ids)
        rows = self._conn.execute(
            f"SELECT * FROM articles WHERE id IN ({placeholders})", ids  # noqa: S608 - ids are ints
        ).fetchall()
        return {int(r["id"]): dict(r) for r in rows}

    def fts_search(
        self,
        query: str,
        *,
        limit: int = 20,
        country: str | None = None,
        language: str | None = None,
        since_iso: str | None = None,
    ) -> list[dict[str, Any]]:
        """SQLite FTS5 keyword-fallback search over title + summary + full_text."""
        clauses = ["articles_fts MATCH ?"]
        params: list[Any] = [query]
        if country:
            clauses.append("articles.country = ?")
            params.append(country)
        if language:
            clauses.append("articles.language = ?")
            params.append(language)
        if since_iso:
            clauses.append("articles.published_at >= ?")
            params.append(since_iso)
        where_sql = " AND ".join(clauses)
        sql = f"""
            SELECT articles.*, bm25(articles_fts) AS rank
            FROM articles_fts
            JOIN articles ON articles.id = articles_fts.rowid
            WHERE {where_sql}
            ORDER BY rank ASC
            LIMIT ?
        """
        params.append(limit)
        rows = self._conn.execute(sql, params).fetchall()
        return [dict(r) for r in rows]

    def timeline(
        self,
        topic: str | None,
        *,
        start_iso: str | None = None,
        end_iso: str | None = None,
    ) -> list[dict[str, Any]]:
        """Group-by-day count, optionally scoped to a topic and date range."""
        clauses: list[str] = []
        params: list[Any] = []
        if topic:
            clauses.append("topic = ?")
            params.append(topic)
        if start_iso:
            clauses.append("coalesce(published_at, ingested_at) >= ?")
            params.append(start_iso)
        if end_iso:
            clauses.append("coalesce(published_at, ingested_at) <= ?")
            params.append(end_iso)
        where_sql = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT
                substr(coalesce(published_at, ingested_at), 1, 10) AS day,
                COUNT(*) AS count
            FROM articles
            {where_sql}
            GROUP BY day
            ORDER BY day ASC
        """
        rows = self._conn.execute(sql, params).fetchall()
        return [{"date": r["day"], "count": int(r["count"])} for r in rows if r["day"]]

    # ---- crawl runs ------------------------------------------------------

    def start_crawl_run(self, topic: str, method: str) -> int:
        with self.transaction() as conn:
            cur = conn.execute(
                "INSERT INTO crawl_runs (topic, method, status) VALUES (?, ?, 'partial')",
                (topic, method),
            )
        return cur.lastrowid

    def finish_crawl_run(
        self,
        run_id: int,
        *,
        fetched: int,
        new_articles: int,
        skipped_dupes: int,
        status: str,
        error_message: str | None = None,
        duration_seconds: float | None = None,
    ) -> None:
        with self.transaction() as conn:
            conn.execute(
                """
                UPDATE crawl_runs SET
                    fetched          = ?,
                    new_articles     = ?,
                    skipped_dupes    = ?,
                    status           = ?,
                    error_message    = ?,
                    duration_seconds = ?
                WHERE id = ?
                """,
                (fetched, new_articles, skipped_dupes, status, error_message, duration_seconds, run_id),
            )


# avoid a circular re-import — re-export from the deduplicator module
from gnews_agent.ingestion.deduplicator import (  # noqa: E402
    publisher_norm as _publisher_norm,
    title_slug as _title_slug,
)
