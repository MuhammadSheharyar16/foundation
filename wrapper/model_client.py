"""Chat and embedding entry points.

These two functions are the only way lab code reaches a model.

Every implementation here must:
  - authenticate from environment or identity sign-in, never a key in code
  - apply a timeout and raise ModelTimeoutError when exceeded
  - wrap provider exceptions in ModelCallError
  - return the typed results in types.py
  - never log the token, authorization header, prompt body or vectors
"""

from .results import ChatResult, EmbeddingResult


def chat(
    prompt: str,
    *,
    system: str | None = None,
    timeout_s: float | None = None,
) -> ChatResult:
    """Send one chat completion request.

    Args:
        prompt: The user message.
        system: Optional system instruction, kept separate from the user message.
        timeout_s: Overrides the configured default timeout.

    Returns:
        ChatResult with the response text and sanitized call metadata.

    Raises:
        ConfigurationMissingError: Configuration or identity unavailable.
        ModelTimeoutError: Call exceeded the timeout budget.
        ModelCallError: Any other provider failure.
    """
    raise NotImplementedError


def embed(
    texts: list[str],
    *,
    timeout_s: float | None = None,
) -> EmbeddingResult:
    """Embed one or more strings.

    Args:
        texts: Input strings. Vectors are returned in the same order.
        timeout_s: Overrides the configured default timeout.

    Returns:
        EmbeddingResult with one vector per input and sanitized call metadata.

    Raises:
        ConfigurationMissingError: Configuration or identity unavailable.
        ModelTimeoutError: Call exceeded the timeout budget.
        ModelCallError: Any other provider failure.
    """
    raise NotImplementedError
