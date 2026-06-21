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
        with self.transaction() as conn:
            cur = conn.execute(
                """
                INSERT INTO articles (
                    title, url, url_hash, publisher_name, publisher_href,
                    published_date, summary, full_text, country, language, topic,
                    embed_model, embed_dim
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    article["title"],
                    article["url"],
                    article_url_hash,
                    article.get("publisher_name"),
                    article.get("publisher_href"),
                    article.get("published_date"),
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

    def get_article(self, article_id: int) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT * FROM articles WHERE id = ?", (article_id,)
        ).fetchone()
        return dict(row) if row else None

    def count_articles(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM articles").fetchone()
        return int(row["n"])

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
