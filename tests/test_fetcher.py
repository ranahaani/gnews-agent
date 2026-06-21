"""Fetcher: per-topic spacing, 429 cooldown, dict→article normalisation."""
from __future__ import annotations

import time

import pytest
from gnews.exceptions import RateLimitError

from gnews_agent.ingestion.fetcher import Fetcher


class FakeGNews:
    """Minimal stand-in for ``GNews`` — counts calls, fakes responses, raises on demand."""

    def __init__(self, *, items=None, raise_with=None):
        self.items = items if items is not None else [
            {
                "title": "OpenAI ships GPT-5",
                "url": "https://reuters.com/openai-gpt5",
                "description": "OpenAI today announced GPT-5.",
                "published date": "Mon, 16 Jun 2026 12:00:00 GMT",
                "publisher": {"title": "Reuters", "href": "https://reuters.com"},
            }
        ]
        self.raise_with = raise_with
        self.calls = 0

    def get_news(self, key):
        self.calls += 1
        if self.raise_with:
            raise self.raise_with
        return list(self.items)

    def get_top_news(self):  # unused in current tests but contract-completing
        self.calls += 1
        return list(self.items)


def test_fetch_normalises_publisher_dict():
    fetcher = Fetcher(gnews_client=FakeGNews(), min_interval_seconds=0)
    result = fetcher.fetch("OpenAI")
    assert result.rate_limited is False
    assert result.error is None
    assert len(result.articles) == 1
    a = result.articles[0]
    assert a["publisher_name"] == "Reuters"
    assert a["publisher_href"] == "https://reuters.com"
    assert a["topic"] == "OpenAI"
    assert a["summary"] == "OpenAI today announced GPT-5."


def test_per_topic_spacing_enforced(monkeypatch):
    fake = FakeGNews()
    fetcher = Fetcher(gnews_client=fake, min_interval_seconds=0.05)

    sleeps: list[float] = []
    monkeypatch.setattr(time, "sleep", lambda s: sleeps.append(s))

    fetcher.fetch("OpenAI")
    fetcher.fetch("OpenAI")  # second call within the window must trigger sleep
    fetcher.fetch("Anthropic")  # different topic — no wait

    assert len(sleeps) == 1
    assert 0 < sleeps[0] <= 0.05


def test_rate_limit_cools_down_topic():
    fake = FakeGNews(raise_with=RateLimitError("hard 429"))
    fetcher = Fetcher(gnews_client=fake, min_interval_seconds=0)

    result1 = fetcher.fetch("OpenAI")
    assert result1.rate_limited is True
    assert "hard 429" in (result1.error or "")
    assert fetcher.is_cooled_down("OpenAI")

    # second attempt is a no-op — does not even hit GNews
    fake.calls = 0
    result2 = fetcher.fetch("OpenAI")
    assert result2.rate_limited is True
    assert fake.calls == 0


def test_unknown_method_rejected():
    fetcher = Fetcher(gnews_client=FakeGNews(), min_interval_seconds=0)
    with pytest.raises(ValueError):
        fetcher.fetch("OpenAI", method="get_news_by_telepathy")
