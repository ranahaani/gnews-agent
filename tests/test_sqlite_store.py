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
    assert {"articles", "dedup_index", "crawl_runs", "article_observations"}.issubset(names)
    assert "articles_fts" in names


def test_connect_backfills_observations_for_existing_articles(tmp_path):
    """Upgrade path: a v0.1.0 DB has articles but no observation rows.

    Without a baseline snapshot, the next ingest of the same URL would look
    like a revision. Connecting to the file must seed one observation per
    article from the current latest-view row.
    """
    db = tmp_path / "news.db"
    store = SqliteStore(db)
    article_id = store.insert_article(_article())
    store._conn.execute("DELETE FROM article_observations")
    store._conn.commit()
    assert store.list_observations(article_id) == []
    store.close()

    reopened = SqliteStore(db)
    rows = reopened.list_observations(article_id)
    assert len(rows) == 1
    assert rows[0]["title"] == "OpenAI ships GPT-5"
    reopened.close()


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


def test_lookup_article_by_url(store):
    article_id = store.insert_article(_article())
    found = store.get_article_id_by_url("https://reuters.com/article/openai-gpt5")
    assert found == article_id
    assert store.get_article_id_by_url("https://example.com/missing") is None


def test_first_observation_is_recorded(store):
    article_id = store.insert_article(_article())
    created = store.record_observation(
        article_id,
        title="OpenAI ships GPT-5",
        summary="OpenAI today announced the long-anticipated GPT-5.",
        url="https://reuters.com/article/openai-gpt5",
    )
    assert created is True
    rows = store.list_observations(article_id)
    assert len(rows) == 1
    assert rows[0]["title"] == "OpenAI ships GPT-5"


def test_identical_observation_is_idempotent(store):
    article_id = store.insert_article(_article())
    kwargs = dict(
        article_id=article_id,
        title="OpenAI ships GPT-5",
        summary="OpenAI today announced the long-anticipated GPT-5.",
        url="https://reuters.com/article/openai-gpt5",
    )
    assert store.record_observation(**kwargs) is True
    assert store.record_observation(**kwargs) is False
    assert len(store.list_observations(article_id)) == 1


def test_headline_rewrite_appends_observation(store):
    article_id = store.insert_article(_article())
    store.record_observation(
        article_id,
        title="OpenAI ships GPT-5",
        summary="OpenAI today announced the long-anticipated GPT-5.",
        url="https://reuters.com/article/openai-gpt5",
    )
    changed = store.record_observation(
        article_id,
        title="OpenAI unveils GPT-5",
        summary="OpenAI today announced the long-anticipated GPT-5.",
        url="https://reuters.com/article/openai-gpt5",
    )
    assert changed is True
    rows = store.list_observations(article_id)
    assert [r["title"] for r in rows] == ["OpenAI ships GPT-5", "OpenAI unveils GPT-5"]


def test_update_article_snapshot(store):
    article_id = store.insert_article(_article())
    store.update_article(
        article_id,
        title="OpenAI unveils GPT-5",
        summary="Updated lede.",
        url="https://reuters.com/article/openai-gpt5",
    )
    fetched = store.get_article(article_id)
    assert fetched["title"] == "OpenAI unveils GPT-5"
    assert fetched["summary"] == "Updated lede."
    assert store.count_articles() == 1


def test_changes_splits_new_from_revised(store):
    first = store.insert_article(_article())
    store.record_observation(
        first,
        title="OpenAI ships GPT-5",
        summary="OpenAI today announced the long-anticipated GPT-5.",
        url="https://reuters.com/article/openai-gpt5",
    )
    store._conn.execute(
        "UPDATE article_observations SET fetched_at = '2026-01-01 00:00:00' WHERE article_id = ?",
        (first,),
    )
    store._conn.commit()

    store.record_observation(
        first,
        title="OpenAI unveils GPT-5",
        summary="OpenAI today announced the long-anticipated GPT-5.",
        url="https://reuters.com/article/openai-gpt5",
    )
    second = store.insert_article(
        _article(title="Anthropic ships Claude 5", url="https://bbc.co.uk/claude-5")
    )
    store.record_observation(
        second,
        title="Anthropic ships Claude 5",
        summary="Anthropic announced Claude 5.",
        url="https://bbc.co.uk/claude-5",
    )

    payload = store.changes(since_iso="2026-06-01")
    new_ids = {row["id"] for row in payload["new"]}
    revised_ids = {row["id"] for row in payload["revised"]}
    assert second in new_ids
    assert first in revised_ids
    assert first not in new_ids
    assert payload["revised"][0]["previous_title"] == "OpenAI ships GPT-5"
