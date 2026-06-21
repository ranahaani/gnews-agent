"""Custom exception hierarchy for gnews-agent."""
from __future__ import annotations


class GNewsAgentError(Exception):
    """Base exception for every gnews-agent error."""


class LLMKeyMissingError(GNewsAgentError):
    """Raised when ``brief()`` / ``sentiment()`` is called without a configured LLM key."""


class EmbeddingDimMismatchError(GNewsAgentError):
    """Raised when the configured embedder produces a dimension that conflicts with stored vectors."""


class IngestionError(GNewsAgentError):
    """Raised when an ingestion run fails irrecoverably."""


class WebhookSecurityError(GNewsAgentError):
    """Raised when a webhook URL fails the SSRF allowlist."""
