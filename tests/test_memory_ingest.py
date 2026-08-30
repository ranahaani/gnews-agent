"""NewsMemory.ingest end-to-end: fetch → dedup → embed → store, idempotency, rate-limit handling."""
from __future__ import annotations

import pytest
from gnews.exceptions import RateLimitError

from gnews_agent import NewsMemory, NewsMemoryConfig
from gnews_agent.ingestion.fetcher import Fetcher
from gnews_agent.storage.sqlite_store import SqliteStore
from tests.conftest import FakeEmbedder, FakeVectorStore
from tests.test_fetcher import FakeGNews


@pytest.fixture
def tmp_config(tmp_path):
    return NewsMemoryConfig(
        db_path=tmp_path / "news.db",
        vector_path=tmp_path / "chroma",
        embed_model="fake-embed",
        embed_dim=8,
    )


def _make_memory(config, *, items=None, rate_limit=False):
    fake_gnews = FakeGNews(
        items=items,
        raise_with=RateLimitError("upstream gave up") if rate_limit else None,
    )
    fetcher = Fetcher(gnews_client=fake_gnews, min_interval_seconds=0)
    embedder = FakeEmbedder()
    vector_store = FakeVectorStore()
    sqlite_store = SqliteStore(config.db_path)
    return NewsMemory(
        config=config,
        fetcher=fetcher,
        embedder=embedder,
        sqlite_store=sqlite_store,
        vector_store=vector_store,
    ), vector_store


def test_ingest_writes_article_and_vector(tmp_config):
    memory, vectors = _make_memory(tmp_config)
    result = memory.ingest("OpenAI")
    assert result["OpenAI"]["status"] == "success"
    assert result["OpenAI"]["new"] == 1
    assert result["OpenAI"]["skipped"] == 0
    stats = memory.stats()
    assert stats["total_articles"] == 1
    assert stats["vector_count"] == 1
    assert vectors.records[0]["metadata"]["topic"] == "OpenAI"


def test_second_ingest_is_a_noop(tmp_config):
    memory, vectors = _make_memory(tmp_config)
    memory.ingest("OpenAI")
    result = memory.ingest("OpenAI")
    assert result["OpenAI"]["new"] == 0
    assert result["OpenAI"]["skipped"] == 1
    assert memory.stats()["total_articles"] == 1
    assert len(vectors.records) == 1


def test_reuters_and_bbc_both_stored(tmp_config):
    items = [
        {
            "title": "OpenAI ships GPT-5",
            "url": "https://reuters.com/openai-gpt5",
            "description": "summary",
            "published date": "Mon, 16 Jun 2026 12:00:00 GMT",
            "publisher": {"title": "Reuters", "href": "https://reuters.com"},
        },
        {
            "title": "OpenAI ships GPT-5",
            "url": "https://bbc.co.uk/news/openai-gpt5",
            "description": "summary",
            "published date": "Mon, 16 Jun 2026 12:00:00 GMT",
            "publisher": {"title": "BBC News", "href": "https://bbc.co.uk"},
        },
    ]
    memory, _ = _make_memory(tmp_config, items=items)
    result = memory.ingest("OpenAI")
    assert result["OpenAI"]["new"] == 2
    assert memory.stats()["total_articles"] == 2


def test_rate_limited_topic_records_status(tmp_config):
    memory, _ = _make_memory(tmp_config, rate_limit=True)
    result = memory.ingest("OpenAI")
    assert result["OpenAI"]["status"] == "rate_limited"
    assert memory.stats()["total_articles"] == 0


def test_batch_ingest(tmp_config):
    memory, _ = _make_memory(tmp_config)
    result = memory.ingest(["OpenAI", "Anthropic"])
    assert set(result.keys()) == {"OpenAI", "Anthropic"}
    # Both topics share the fake article — dedup catches the second one.
    assert memory.stats()["total_articles"] == 1


def test_empty_fetch_is_success_with_zero_counts(tmp_config):
    memory, _ = _make_memory(tmp_config, items=[])
    result = memory.ingest("OpenAI")
    assert result["OpenAI"] == {
        "fetched": 0,
        "new": 0,
        "skipped": 0,
        "revised": 0,
        "status": "success",
    }
    assert memory.stats()["total_articles"] == 0


def _item(**overrides):
    base = {
        "title": "OpenAI ships GPT-5",
        "url": "https://reuters.com/openai-gpt5",
        "description": "OpenAI today announced GPT-5.",
        "published date": "Mon, 16 Jun 2026 12:00:00 GMT",
        "publisher": {"title": "Reuters", "href": "https://reuters.com"},
    }
    base.update(overrides)
    return base


def test_second_ingest_reports_revised_zero(tmp_config):
    memory, _ = _make_memory(tmp_config)
    first = memory.ingest("OpenAI")
    second = memory.ingest("OpenAI")
    assert first["OpenAI"]["revised"] == 0
    assert second["OpenAI"]["revised"] == 0
    assert second["OpenAI"]["skipped"] == 1


def test_ab_headline_is_a_revision_not_a_new_row(tmp_config):
    """Same canonical URL, rewritten headline → one article, two observations.

    This is the A/B-test / CMS overwrite case from the Reddit thread: the
    title+publisher key would treat it as a new story, then the url_hash
    UNIQUE constraint used to swallow it as a silent skip.
    """
    memory, vectors = _make_memory(tmp_config, items=[_item()])
    memory.ingest("OpenAI")

    rewritten = _make_memory(
        tmp_config,
        items=[_item(title="OpenAI unveils GPT-5")],
    )[0]
    # Reuse the same sqlite + vectors so the second ingest hits the existing DB.
    rewritten._sqlite = memory._sqlite
    rewritten._vectors = vectors
    result = rewritten.ingest("OpenAI")

    assert result["OpenAI"]["new"] == 0
    assert result["OpenAI"]["revised"] == 1
    assert result["OpenAI"]["skipped"] == 0
    assert memory.stats()["total_articles"] == 1
    article = memory._sqlite.get_article(1)
    assert article["title"] == "OpenAI unveils GPT-5"
    assert len(memory._sqlite.list_observations(1)) == 2
    assert len(vectors.records) == 1
    assert vectors.records[0]["metadata"]["title"] == "OpenAI unveils GPT-5"


def test_summary_edit_on_same_url_is_a_revision(tmp_config):
    memory, _ = _make_memory(tmp_config, items=[_item()])
    memory.ingest("OpenAI")
    follow = _make_memory(
        tmp_config,
        items=[_item(description="OpenAI delayed GPT-5 until next week.")],
    )[0]
    follow._sqlite = memory._sqlite
    follow._vectors = memory._vectors
    result = follow.ingest("OpenAI")
    assert result["OpenAI"]["revised"] == 1
    assert memory.stats()["total_articles"] == 1
    assert memory._sqlite.get_article(1)["summary"] == "OpenAI delayed GPT-5 until next week."


def test_google_news_url_variant_still_dedups(tmp_config):
    """Different Google News URL, same title+publisher → still one row.

    URL is identity only when it matches. Title+publisher remains the
    candidate key for the redirector/locale-variant case.
    """
    memory, _ = _make_memory(tmp_config, items=[_item()])
    memory.ingest("OpenAI")
    variant = _make_memory(
        tmp_config,
        items=[_item(url="https://news.google.com/rss/articles/abc123")],
    )[0]
    variant._sqlite = memory._sqlite
    variant._vectors = memory._vectors
    result = variant.ingest("OpenAI")
    assert result["OpenAI"]["new"] == 0
    assert result["OpenAI"]["revised"] == 0
    assert result["OpenAI"]["skipped"] == 1
    assert memory.stats()["total_articles"] == 1


def test_changes_reports_new_then_revised(tmp_config):
    memory, _ = _make_memory(tmp_config, items=[_item()])
    memory.ingest("OpenAI")
    memory._sqlite._conn.execute(
        "UPDATE article_observations SET fetched_at = '2026-01-01 00:00:00'"
    )
    memory._sqlite._conn.commit()

    follow = _make_memory(
        tmp_config,
        items=[_item(title="OpenAI unveils GPT-5")],
    )[0]
    follow._sqlite = memory._sqlite
    follow._vectors = memory._vectors
    follow.ingest("OpenAI")

    payload = memory.changes(days=30)
    assert payload["revised_count"] == 1
    assert payload["new_count"] == 0
    assert payload["revised"][0]["title"] == "OpenAI unveils GPT-5"
    assert payload["revised"][0]["previous_title"] == "OpenAI ships GPT-5"
