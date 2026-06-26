"""LLM client retry-delay helper — parses provider rate-limit hints."""
from __future__ import annotations

from gnews_agent.ai.client import _retry_delay


def test_groq_rate_limit_message_extracts_wait_plus_pad():
    err = Exception(
        "litellm.RateLimitError: GroqException - "
        '{"error":{"message":"Rate limit reached. Please try again in 45.8s",'
        '"type":"tokens","code":"rate_limit_exceeded"}}'
    )
    delay = _retry_delay(err)
    # 45.8 + 2.0 padding
    assert delay == 47.8


def test_rate_limit_without_explicit_seconds_uses_conservative_default():
    err = Exception("Rate limit exceeded for tokens per minute")
    assert _retry_delay(err) == 30.0


def test_non_rate_limit_returns_short_delay():
    err = Exception("malformed JSON response")
    assert _retry_delay(err) == 1.0
