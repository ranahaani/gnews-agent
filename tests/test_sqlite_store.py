"""SqliteStore: schema apply, article insert, dedup detection, crawl-run lifecycle."""
from __future__ import annotations

import sqlite3

import pytest

from gnews_agent.storage.sqlite_store import SqliteStore


def _article(**overrides):
    base = {
        "title": "OpenAI ships GPT-5",
        "url": "https://reuters.com/article/openai-gpt5",
        "publisher_name": "Reuters",
        "publisher_href": "https://reuters.com",
        "published_date": "Mon, 16 Jun 2026 12:00:00 GMT",
        "summary": "OpenAI today announced the long-anticipated GPT-5.",
        "full_text": None,
        "country": "US",
        "language": "en",
        "topic": "OpenAI",
        "embed_model": "all-MiniLM-L6-v2",
        "embed_dim": 384,
    }
    base.update(overrides)
    return base


@pytest.fixture
def store(tmp_path):
    s = SqliteStore(tmp_path / "news.db")
    yield s
    s.close()


def test_schema_applied_on_connect(store):
    # All four tables present.
    conn = store._conn
    names = {
        row["name"]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','virtual')"
        )
    }
    assert {"articles", "dedup_index", "crawl_runs"}.issubset(names)
    assert "articles_fts" in names


def test_insert_and_get_article(store):
    article_id = store.insert_article(_article())
    fetched = store.get_article(article_id)
    assert fetched is not None
    assert fetched["title"] == "OpenAI ships GPT-5"
    assert fetched["embed_dim"] == 384
    assert store.count_articles() == 1


def test_url_hash_collision_raises(store):
    store.insert_article(_article())
    # Same URL, different title → url_hash UNIQUE backstop fires.
    with pytest.raises(sqlite3.IntegrityError):
        store.insert_article(_article(title="Different headline"))


def test_dedup_index_records_seen(store):
    article_id = store.insert_article(_article())
    store.record_seen("OpenAI ships GPT-5", "Reuters", article_id)

    assert store.is_duplicate("OpenAI ships GPT-5", "Reuters") is True
    assert store.is_duplicate("openai SHIPS gpt-5", "REUTERS") is True  # canonicalisation
    assert store.is_duplicate("OpenAI ships GPT-5", "BBC News") is False  # different publisher
    assert store.is_duplicate("OpenAI delays GPT-5", "Reuters") is False  # different title


def test_record_seen_increments_seen_count(store):
    store.record_seen("Story", "Reuters", None)
    store.record_seen("Story", "Reuters", None)
    row = store._conn.execute(
        "SELECT seen_count FROM dedup_index WHERE title_slug = 'story'"
    ).fetchone()
    assert row["seen_count"] == 2


def test_crawl_run_lifecycle(store):
    run_id = store.start_crawl_run(topic="OpenAI", method="get_news")
    store.finish_crawl_run(
        run_id,
        fetched=10,
        new_articles=8,
        skipped_dupes=2,
        status="success",
        duration_seconds=1.23,
    )
    row = store._conn.execute(
        "SELECT * FROM crawl_runs WHERE id = ?", (run_id,)
    ).fetchone()
    assert row["status"] == "success"
    assert row["new_articles"] == 8
    assert row["skipped_dupes"] == 2


def test_dedup_does_not_collapse_reuters_and_bbc(store):
    """Critical PRD/design invariant: Reuters + BBC on the same event stay separate."""
    reuters_id = store.insert_article(_article(publisher_name="Reuters"))
    store.record_seen("OpenAI ships GPT-5", "Reuters", reuters_id)

    bbc_id = store.insert_article(
        _article(publisher_name="BBC News", url="https://bbc.co.uk/news/openai-gpt5")
    )
    store.record_seen("OpenAI ships GPT-5", "BBC News", bbc_id)

    assert store.count_articles() == 2
