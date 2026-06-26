"""Integration-test fixtures.

These tests hit real services — GNews RSS, the sentence-transformers
model download, an LLM provider — so they are marked ``integration`` and
skipped unless ``-m integration`` (or ``-m "integration or not integration"``)
is passed to pytest. They auto-load ``~/ai-me/.env`` so the user's
existing keys (ANTHROPIC_API_KEY, GROQ_API_KEY, GEMINI_API_KEY) flow in.
"""
from __future__ import annotations

import os
from pathlib import Path

import pytest


def _load_env() -> None:
    """Load the nearest .env walking up from the repo root.

    Skipped silently if python-dotenv isn't installed — env vars exported
    in the shell still work fine.
    """
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    here = Path(__file__).resolve()
    for parent in (here.parent, *here.parents):
        candidate = parent / ".env"
        if candidate.is_file():
            load_dotenv(candidate, override=False)
            return


_load_env()


# ---------------------------------------------------------------------------
# provider selection — pick whichever LLM key is set, prefer fastest+cheapest
# ---------------------------------------------------------------------------

_PROVIDER_PREF = [
    # (provider, env var, default model when this provider wins)
    # Anthropic first: claude-3-5-haiku has ~50k TPM on the free tier vs
    # Groq's 6k on llama-3.1-8b — the full integration suite easily
    # blows past 6k TPM when brief() + sentiment() share a minute.
    ("anthropic", "ANTHROPIC_API_KEY", "anthropic/claude-3-5-haiku-latest"),
    ("openai",    "OPENAI_API_KEY",    "openai/gpt-4o-mini"),
    ("gemini",    "GEMINI_API_KEY",    "gemini/gemini-1.5-flash"),
    ("groq",      "GROQ_API_KEY",      "groq/llama-3.1-8b-instant"),
]


def _provider_is_live(model: str) -> bool:
    """Tiny completion to verify the key is actually valid.

    Skips dead/rotated keys so the suite doesn't hard-fail when ``.env``
    has stale credentials for one provider while another is fine.
    """
    try:
        import litellm
        litellm.completion(
            model=model,
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=1,
            temperature=0,
        )
        return True
    except Exception:
        return False


def _selected_provider() -> tuple[str, str, str] | None:
    for provider, env, model in _PROVIDER_PREF:
        if not os.environ.get(env):
            continue
        if _provider_is_live(model):
            return provider, env, model
    return None


@pytest.fixture(scope="session")
def llm_provider_choice():
    pick = _selected_provider()
    if pick is None:
        pytest.skip("no live LLM provider key (probed ANTHROPIC/OPENAI/GEMINI/GROQ)")
    return pick
