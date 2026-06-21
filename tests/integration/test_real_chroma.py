"""ChromaDB backend — real persistence + cross-process survival + model-lock."""
from __future__ import annotations

import pytest

from gnews_agent.exceptions import EmbeddingDimMismatchError
from gnews_agent.storage.backends.chroma import ChromaVectorStore


pytestmark = pytest.mark.integration


def test_upsert_then_query_round_trip(tmp_path):
    store = ChromaVectorStore(
        persist_path=str(tmp_path / "chroma"),
        embed_model="real-st",
        embed_dim=4,
    )
    store.upsert(1, [0.1, 0.2, 0.3, 0.4], metadata={"topic": "OpenAI", "url": "u1", "document": "GPT-5"})
    store.upsert(2, [0.9, 0.8, 0.7, 0.6], metadata={"topic": "Apple", "url": "u2", "document": "Vision Pro"})
    assert store.count() == 2

    # Query closer to row 1 — row 1 should rank first.
    hits = store.query([0.1, 0.2, 0.3, 0.4], k=2)
    assert [h.article_id for h in hits] == [1, 2]
    assert hits[0].score > hits[1].score


def test_persistence_survives_reopen(tmp_path):
    path = str(tmp_path / "chroma")
    store_a = ChromaVectorStore(persist_path=path, embed_model="real-st", embed_dim=4)
    store_a.upsert(7, [0.1, 0.2, 0.3, 0.4], metadata={"topic": "X", "url": "u", "document": "doc"})
    del store_a

    store_b = ChromaVectorStore(persist_path=path, embed_model="real-st", embed_dim=4)
    assert store_b.count() == 1


def test_reopening_with_wrong_model_raises(tmp_path):
    path = str(tmp_path / "chroma")
    ChromaVectorStore(persist_path=path, embed_model="real-st", embed_dim=4)
    with pytest.raises(EmbeddingDimMismatchError):
        ChromaVectorStore(persist_path=path, embed_model="some-other-model", embed_dim=4)


def test_dim_mismatch_on_upsert_raises(tmp_path):
    store = ChromaVectorStore(persist_path=str(tmp_path / "chroma"), embed_model="real-st", embed_dim=4)
    with pytest.raises(EmbeddingDimMismatchError):
        store.upsert(1, [0.1, 0.2, 0.3], metadata={"document": "x"})  # dim=3 not 4


def test_metadata_where_filter(tmp_path):
    store = ChromaVectorStore(persist_path=str(tmp_path / "chroma"), embed_model="real-st", embed_dim=4)
    store.upsert(1, [0.1, 0.2, 0.3, 0.4], metadata={"country": "US", "document": "doc1"})
    store.upsert(2, [0.5, 0.5, 0.5, 0.5], metadata={"country": "UK", "document": "doc2"})
    hits = store.query([0.1, 0.2, 0.3, 0.4], k=5, where={"country": "UK"})
    assert [h.article_id for h in hits] == [2]
