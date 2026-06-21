"""Real GNews fetch — verifies the Fetcher layer works against live RSS."""
from __future__ import annotations

import pytest

from gnews_agent.ingestion.fetcher import Fetcher


pytestmark = pytest.mark.integration


def test_real_fetch_returns_normalised_articles():
    fetcher = Fetcher(language="en", country="US", max_results=5, min_interval_seconds=0)
    result = fetcher.fetch("artificial intelligence")
    assert result.rate_limited is False, result.error
    assert result.error is None
    assert len(result.articles) >= 1
    article = result.articles[0]
    # All four contract fields must be populated by GNews's RSS path.
    assert article["title"]
    assert article["url"]
    assert article["publisher_name"]
    assert article["topic"] == "artificial intelligence"


def test_real_fetch_per_topic_spacing_is_quick_enough():
    """The 1s default would slow CI excessively — we set min=0 elsewhere.
    This test asserts the default value is exactly 1.0 (PRD §I-12)."""
    fetcher = Fetcher()
    assert fetcher._min_interval == 1.0
