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
