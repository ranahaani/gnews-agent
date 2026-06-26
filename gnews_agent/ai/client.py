"""Thin LiteLLM wrapper.

Centralises the "do we have an LLM key?" check so ``brief()`` / ``sentiment()``
fail with a clear actionable error instead of an opaque provider exception.
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from typing import Any

from gnews_agent.exceptions import LLMKeyMissingError


logger = logging.getLogger(__name__)


_PROVIDER_ENV = {
    "openai":    "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "groq":      "GROQ_API_KEY",
    "ollama":    None,   # local, no key needed
}


@dataclass(frozen=True)
class LLMConfig:
    provider: str
    model: str
    temperature: float = 0.1
    # 4000 gives the brief prompt (~200-400 words + JSON envelope + per-article
    # citations) headroom on every provider. 1500 caused Groq's llama-3.1-8b
    # to truncate mid-JSON; LiteLLM then surfaced it as
    # "max completion tokens reached before generating a valid document".
    max_tokens: int = 4000


class LLMClient:
    """Wraps ``litellm.completion`` + JSON parsing."""

    def __init__(self, config: LLMConfig) -> None:
        self.config = config

    def _ensure_key(self) -> None:
        env_name = _PROVIDER_ENV.get(self.config.provider)
        if env_name is None:
            return  # provider needs no key (ollama)
        if not os.environ.get(env_name):
            raise LLMKeyMissingError(
                f"{self.config.provider!r} requires {env_name} to be set. "
                f"Either export {env_name}, switch provider via "
                f"NewsMemoryConfig(llm_provider=...), or call the keyless "
                f"methods (search/ingest/timeline) instead."
            )

    def complete_json(self, prompt: str, *, retries: int = 1) -> dict[str, Any]:
        """Send a single-shot chat completion and parse the JSON body.

        Retries once by default on JSON-validation failures and on parse
        errors — small open-source models (Groq Llama 3.1 8B, etc.) sometimes
        emit malformed JSON or truncate mid-document even with
        ``response_format``. One additional attempt resolves most cases in
        practice without doubling latency for the common path.
        """
        self._ensure_key()
        try:
            import litellm
        except ImportError as exc:  # pragma: no cover
            raise ImportError("brief()/sentiment() require litellm — `pip install litellm`") from exc

        model_id = (
            self.config.model
            if "/" in self.config.model
            else f"{self.config.provider}/{self.config.model}"
        )

        last_exc: Exception | None = None
        for attempt in range(retries + 1):
            try:
                response = litellm.completion(
                    model=model_id,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    response_format={"type": "json_object"},
                )
                body = response["choices"][0]["message"]["content"]
                return _parse_json(body)
            except (json.JSONDecodeError, Exception) as exc:
                last_exc = exc
                logger.warning(
                    "LLM JSON call failed (attempt %d/%d): %s",
                    attempt + 1, retries + 1, exc,
                )
                if attempt == retries:
                    raise
                wait = _retry_delay(exc)
                if wait:
                    logger.info("backing off %.1fs before retry", wait)
                    time.sleep(wait)
        # unreachable
        assert last_exc is not None
        raise last_exc


_RATE_LIMIT_SECONDS_RE = re.compile(r"try again in ([\d.]+)\s*s", re.IGNORECASE)


def _retry_delay(exc: Exception) -> float:
    """Pick a sensible backoff for the given exception.

    Per-minute token rate limits (Groq free tier returns "try again in 45.8s")
    are honoured by parsing the suggested wait out of the error message and
    sleeping a couple of seconds past it so the next attempt lands in a fresh
    bucket. Other transient errors (parse failures, generic provider errors)
    fall through to a short fixed delay.
    """
    text = str(exc).lower()
    is_rate_limited = "rate" in text and "limit" in text
    if is_rate_limited:
        match = _RATE_LIMIT_SECONDS_RE.search(text)
        if match:
            try:
                return float(match.group(1)) + 2.0
            except ValueError:
                pass
        return 30.0  # conservative default when the provider doesn't say
    return 1.0


def _parse_json(body: str) -> dict[str, Any]:
    text = body.strip()
    # Strip ```json fences if a provider ignored response_format.
    if text.startswith("```"):
        text = text.strip("`")
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    return json.loads(text)
