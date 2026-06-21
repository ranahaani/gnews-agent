"""Query helpers — date parsing, hydration, re-ranking.

Kept in a dedicated module so ``NewsMemory`` stays an orchestration facade.
"""
from __future__ import annotations

import datetime as _dt
import math
from typing import Any


def parse_date(value: str | None) -> _dt.datetime | None:
    """Best-effort parser for the date formats GNews returns.

    Returns ``None`` when the input is missing or unparseable — callers
    decide what to do with unparseable dates rather than the parser raising.
    """
    if not value:
        return None
    fmts = (
        "%a, %d %b %Y %H:%M:%S %Z",   # RFC822 GNews default ("Mon, 16 Jun 2026 12:00:00 GMT")
        "%a, %d %b %Y %H:%M:%S GMT",
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%SZ",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
    )
    for fmt in fmts:
        try:
            return _dt.datetime.strptime(value, fmt)
        except (ValueError, TypeError):
            continue
    return None


def recency_score(published: _dt.datetime | None, *, half_life_days: float = 3.0) -> float:
    """Exponential decay toward 0 as ``published`` ages, anchored at 1.0 for "now".

    Articles with no parseable date score 0.5 — a neutral midpoint rather
    than being filtered out, since some publishers ship missing dates.
    """
    if published is None:
        return 0.5
    now = _dt.datetime.now(published.tzinfo) if published.tzinfo else _dt.datetime.now()
    age_days = max(0.0, (now - published).total_seconds() / 86400.0)
    return math.exp(-age_days / half_life_days)


def blend(similarity: float, recency: float, *, recency_weight: float = 0.3) -> float:
    """Convex combination of semantic similarity and recency."""
    return (1 - recency_weight) * similarity + recency_weight * recency
