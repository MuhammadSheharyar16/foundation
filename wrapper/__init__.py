"""Model wrapper package.

The single boundary between lab code and the AI SDK.

No file outside this package imports the SDK.
No credential appears in code, logs, screenshots or committed config.
"""

from .config import WrapperConfig, load_config
from .errors import ConfigurationMissingError, ModelCallError, ModelTimeoutError
from .model_client import chat, embed
from .results import ChatResult, EmbeddingResult

__all__ = [
    "WrapperConfig",
    "load_config",
    "ConfigurationMissingError",
    "ModelCallError",
    "ModelTimeoutError",
    "chat",
    "embed",
    "ChatResult",
    "EmbeddingResult",
]
