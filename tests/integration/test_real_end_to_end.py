"""End-to-end: real NewsMemory with real GNews + real embeddings + real Chroma.

This is the demo path from README — `pip install` then ingest + search work
keylessly with zero stubs. brief() requires a key and is covered separately
in test_real_brief.py.
"""
from __future__ import annotations

import pytest

from gnews_agent import NewsMemory, NewsMemoryConfig


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def memory(tmp_path_factory):
    state = tmp_path_factory.mktemp("end_to_end_state")
    cfg = NewsMemoryConfig(
        db_path=state / "news.db",
        vector_path=state / "chroma",
        max_fetch_results=10,
        fetch_min_interval_seconds=0,
    )
    return NewsMemory(config=cfg)


def test_ingest_writes_real_articles(memory):
    result = memory.ingest("artificial intelligence")
    assert result["artificial intelligence"]["status"] == "success"
    assert result["artificial intelligence"]["new"] >= 1
    stats = memory.stats()
    assert stats["total_articles"] >= 1
    assert stats["vector_count"] >= 1
    assert stats["embed_model"] == "all-MiniLM-L6-v2"
    assert stats["embed_dim"] == 384


def test_second_ingest_dedups(memory):
    # First ingest is idempotent — runs whether or not test_ingest ran first.
    memory.ingest("artificial intelligence")
    before = memory.stats()["total_articles"]
    result = memory.ingest("artificial intelligence")
    after = memory.stats()["total_articles"]
    # Second call must not add the same articles back.
    assert after == before, f"second ingest grew store from {before} to {after}"
    assert result["artificial intelligence"]["skipped"] >= 1


def test_semantic_search_returns_real_hits(memory):
    hits = memory.search("artificial intelligence", limit=5, semantic=True)
    assert len(hits) >= 1
    assert all("title" in h and "url" in h for h in hits)
    assert hits[0]["search_mode"] == "semantic"
    # Scores should be in descending order (recency-blended).
    scores = [h["score"] for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_keyword_fallback_real(memory):
    # OpenAI articles are in the store — FTS5 should find them on the
    # plain-text title.
    hits = memory.search("artificial", semantic=False, limit=5)
    assert len(hits) >= 1
    assert all(h["search_mode"] == "keyword" for h in hits)


def test_timeline_real(memory):
    days = memory.timeline("artificial intelligence", days=30)
    # At least one day must have an article.
    assert len(days) >= 1
    assert all(d["count"] >= 1 for d in days)
