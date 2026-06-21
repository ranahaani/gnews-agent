"""Sentiment scoring + optional day-by-day timeline."""
from __future__ import annotations

from collections import defaultdict
from importlib.resources import files
from typing import Any

from gnews_agent.ai.client import LLMClient


def _load_template() -> str:
    return files("gnews_agent.ai.prompts").joinpath("sentiment_v1.txt").read_text(encoding="utf-8")


def _format_articles(articles: list[dict[str, Any]]) -> str:
    return "\n".join(
        f"- {a.get('title')} — {a.get('publisher') or 'unknown'} ({a.get('published_date') or '?'})\n"
        f"  {a.get('summary') or ''}"
        for a in articles
    )


def score_sentiment(
    llm: LLMClient,
    *,
    topic: str,
    articles: list[dict[str, Any]],
) -> dict[str, Any]:
    """Single-shot sentiment score across the supplied corpus."""
    if not articles:
        return {"overall": "neutral", "score": 0.0, "rationale": "no articles", "article_count": 0}
    prompt = _load_template().format(topic=topic, articles_block=_format_articles(articles))
    response = llm.complete_json(prompt)
    response["article_count"] = len(articles)
    return response


def score_sentiment_timeline(
    llm: LLMClient,
    *,
    topic: str,
    articles: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """One sentiment score per calendar day in the corpus."""
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for a in articles:
        day = (a.get("published_at") or a.get("published_date") or "")[:10]
        if day:
            buckets[day].append(a)

    timeline: list[dict[str, Any]] = []
    for day in sorted(buckets):
        scored = score_sentiment(llm, topic=topic, articles=buckets[day])
        scored["date"] = day
        timeline.append(scored)
    return timeline
