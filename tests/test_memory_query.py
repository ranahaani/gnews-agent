"""NewsMemory.search (semantic + keyword fallback) + timeline."""
from __future__ import annotations

import pytest

from gnews_agent import NewsMemory, NewsMemoryConfig
from gnews_agent.ingestion.fetcher import Fetcher
from gnews_agent.storage.sqlite_store import SqliteStore
from gnews_agent.storage.vector_store import VectorHit

from tests.conftest import FakeEmbedder, FakeVectorStore
from tests.test_fetcher import FakeGNews


class ScriptedVectorStore(FakeVectorStore):
    """FakeVectorStore that returns a scripted query result so search re-rank can be exercised."""

    def __init__(self, *, embed_model="fake-embed", embed_dim=8):
        super().__init__(embed_model=embed_model, embed_dim=embed_dim)
        self.scripted: list[VectorHit] = []

    def query(self, embedding, *, k=10, where=None):
        return list(self.scripted[:k])


@pytest.fixture
def memory(tmp_path):
    cfg = NewsMemoryConfig(
        db_path=tmp_path / "news.db",
        vector_path=tmp_path / "chroma",
        embed_model="fake-embed",
        embed_dim=8,
    )
    items = [
        {
            "title": "OpenAI ships GPT-5",
            "url": "https://reuters.com/openai-gpt5",
            "description": "OpenAI today announced GPT-5.",
            "published date": "Mon, 16 Jun 2026 12:00:00 GMT",
            "publisher": {"title": "Reuters", "href": "https://reuters.com"},
        },
        {
            "title": "Anthropic announces Claude 5",
            "url": "https://bbc.co.uk/news/anthropic-claude5",
            "description": "Anthropic announced Claude 5.",
            "published date": "Sun, 15 Jun 2026 12:00:00 GMT",
            "publisher": {"title": "BBC News", "href": "https://bbc.co.uk"},
        },
    ]
    vectors = ScriptedVectorStore()
    mem = NewsMemory(
        config=cfg,
        fetcher=Fetcher(gnews_client=FakeGNews(items=items), min_interval_seconds=0),
        embedder=FakeEmbedder(),
        sqlite_store=SqliteStore(cfg.db_path),
        vector_store=vectors,
    )
    mem.ingest("OpenAI")
    # Script the next semantic query: GPT-5 article first.
    article_ids = sorted(int(r["article_id"]) for r in vectors.records)
    vectors.scripted = [
        VectorHit(article_id=article_ids[0], score=0.95, metadata={"topic": "OpenAI"}),
        VectorHit(article_id=article_ids[1], score=0.10, metadata={"topic": "OpenAI"}),
    ]
    return mem


def test_semantic_search_returns_top_first(memory):
    hits = memory.search("GPT-5 launch", limit=2)
    assert len(hits) == 2
    assert hits[0]["title"] == "OpenAI ships GPT-5"
    assert hits[0]["search_mode"] == "semantic"
    assert hits[0]["score"] > hits[1]["score"]


def test_keyword_fallback_when_semantic_false(memory):
    hits = memory.search("Claude", semantic=False, limit=5)
    assert any("Claude" in h["title"] for h in hits)
    assert all(h["search_mode"] == "keyword" for h in hits)


def test_timeline_groups_by_day(memory):
    days = memory.timeline("OpenAI")
    assert {"2026-06-15", "2026-06-16"}.issubset({d["date"] for d in days})
    by_day = {d["date"]: d["count"] for d in days}
    assert by_day["2026-06-16"] == 1
    assert by_day["2026-06-15"] == 1


def test_timeline_filters_by_topic(memory):
    days = memory.timeline("Anthropic")  # nothing was ingested under this topic name
    assert days == []
