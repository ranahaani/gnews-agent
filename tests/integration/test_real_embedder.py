"""Sentence-transformers embedder — real model load, dim correctness."""
from __future__ import annotations

import pytest

from gnews_agent.ingestion.embedder import SentenceTransformerEmbedder


pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def embedder():
    return SentenceTransformerEmbedder(embed_model="all-MiniLM-L6-v2", embed_dim=384)


def test_embed_returns_384_dim_vectors(embedder):
    vectors = embedder.embed(["OpenAI ships GPT-5", "Apple announces Vision Pro 2"])
    assert len(vectors) == 2
    assert all(len(v) == 384 for v in vectors)


def test_embed_is_normalised(embedder):
    """all-MiniLM with normalize_embeddings=True should produce unit vectors."""
    vectors = embedder.embed(["test sentence"])
    magnitude = sum(x * x for x in vectors[0]) ** 0.5
    assert abs(magnitude - 1.0) < 1e-5


def test_semantically_similar_sentences_score_high(embedder):
    """Sanity: a near-paraphrase should outrank an unrelated sentence."""
    base, near, far = embedder.embed([
        "OpenAI released GPT-5 today",
        "OpenAI unveiled GPT-5 this morning",
        "The Boston Celtics won the basketball game",
    ])
    def dot(a, b): return sum(x * y for x, y in zip(a, b))
    near_sim = dot(base, near)
    far_sim = dot(base, far)
    assert near_sim > far_sim
    assert near_sim > 0.7


def test_empty_input_returns_empty():
    e = SentenceTransformerEmbedder()
    assert e.embed([]) == []
