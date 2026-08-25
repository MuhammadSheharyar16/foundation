"""Typed return values for the wrapper.

Callers depend on these, not on raw SDK response objects.
"""

from dataclasses import dataclass, field


@dataclass(frozen=True)
class ChatResult:
    """Result of a single chat completion call.

    Attributes:
        text: The generated response text.
        request_id: Correlation ID for this call. Safe to log.
        model_alias: Alias used, e.g. "chat-default". Never the raw deployment secret.
        latency_ms: Wall-clock duration of the call.
        prompt_tokens: Input token count, where the provider reports it.
        completion_tokens: Output token count, where the provider reports it.
    """

    text: str
    request_id: str
    model_alias: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None

    def sanitized_metadata(self) -> dict:
        """Return only the fields that are safe to print or write to artifacts.

        Must exclude `text`.
        """
        return {
            "request_id": self.request_id,
            "model_alias": self.model_alias,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
        }


@dataclass(frozen=True)
class EmbeddingResult:
    """Result of a single embedding call.

    Attributes:
        vectors: One vector per input string, in input order.
        request_id: Correlation ID for this call. Safe to log.
        model_alias: Alias used, e.g. "embed-default".
        dimensions: Length of each vector.
        latency_ms: Wall-clock duration of the call.
    """

    vectors: list[list[float]] = field(default_factory=list)
    request_id: str = ""
    model_alias: str = ""
    dimensions: int = 0
    latency_ms: float = 0.0

    def sanitized_metadata(self) -> dict:
        """Return only the fields that are safe to print or write to artifacts.

        Must exclude `vectors`.
        """
        return {
            "request_id": self.request_id,
            "model_alias": self.model_alias,
            "dimensions": self.dimensions,
            "latency_ms": self.latency_ms,
        }
