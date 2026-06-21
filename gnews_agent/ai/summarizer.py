"""Cited brief generation."""
from __future__ import annotations

from importlib.resources import files
from typing import Any

from gnews_agent.ai.client import LLMClient


def _load_template() -> str:
    return files("gnews_agent.ai.prompts").joinpath("brief_v1.txt").read_text(encoding="utf-8")


def _format_articles(articles: list[dict[str, Any]]) -> str:
    lines: list[str] = []
    for idx, a in enumerate(articles, 1):
        lines.append(
            f"[{idx}] {a.get('title')} — {a.get('publisher') or 'unknown'} "
            f"({a.get('published_date') or 'date unknown'})\n"
            f"    URL: {a.get('url')}\n"
            f"    Summary: {a.get('summary') or ''}"
        )
    return "\n".join(lines)


def make_brief(
    llm: LLMClient,
    *,
    topic: str,
    days: int,
    articles: list[dict[str, Any]],
) -> dict[str, Any]:
    """Render the brief prompt + parse the JSON response."""
    template = _load_template()
    prompt = template.format(
        topic=topic,
        days=days,
        articles_block=_format_articles(articles),
    )
    response = llm.complete_json(prompt)
    response.setdefault("citations", [a.get("url") for a in articles if a.get("url")])
    response["article_count"] = len(articles)
    return response
