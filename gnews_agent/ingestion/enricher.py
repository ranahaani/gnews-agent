"""Contextual header + optional full-text enrichment.

The header is a small structured prefix the embedder sees in front of the
article text. It carries topic/publisher/date so semantic search has those
signals baked into the embedding — without it, a query like
"OpenAI announcement in June 2026" only matches on titles that happen to
mention the date.

This is NOT Anthropic's full contextual-retrieval technique (that requires
chunking long docs). News articles in v1 are short enough for a single
embedding per article; this is a metadata prefix, not chunk contextualisation.
"""
from __future__ import annotations

from typing import Any


def build_context_header(article: dict[str, Any]) -> str:
    topic = article.get("topic") or "general"
    publisher = article.get("publisher_name") or "unknown publisher"
    date = article.get("published_date") or "unknown date"
    country = article.get("country") or ""
    suffix = f" | {country}" if country else ""
    return f"[Topic: {topic} | Publisher: {publisher} | Date: {date}{suffix}]"


def enriched_text(article: dict[str, Any]) -> str:
    """Return ``<header>\\n<title>\\n<summary or full_text>``."""
    header = build_context_header(article)
    title = article.get("title") or ""
    body = article.get("full_text") or article.get("summary") or ""
    return f"{header}\n{title}\n{body}".strip()
