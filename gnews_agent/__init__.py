"""gnews-agent — persistent, semantic news intelligence layer for AI agents."""
from gnews_agent.config import NewsMemoryConfig
from gnews_agent.exceptions import (
    EmbeddingDimMismatchError,
    GNewsAgentError,
    LLMKeyMissingError,
)
from gnews_agent.memory import NewsMemory

__version__ = "0.1.0"

__all__ = [
    "NewsMemory",
    "NewsMemoryConfig",
    "GNewsAgentError",
    "LLMKeyMissingError",
    "EmbeddingDimMismatchError",
    "__version__",
]
