"""Bridge between lab retrieval code and an embedding backend.

Retrieval code should depend on the `EmbeddingProvider` protocol below, never
on a concrete backend directly. That keeps retrieval logic (chunking,
similarity, ranking) testable with a deterministic fake, and lets the real
`wrapper.embed` be swapped in for production without touching callers.
"""

import asyncio
import hashlib
import math
from typing import Protocol

from wrapper import EmbeddingResult
from wrapper import embed as wrapper_embed


class EmbeddingProvider(Protocol):
    """Anything that can turn text into embedding vectors.

    Mirrors the call shape of `wrapper.embed` so a real provider and a fake
    provider are interchangeable wherever an `EmbeddingProvider` is expected.
    """

    def embed(
        self,
        texts: list[str],
        *,
        timeout_s: float | None = None,
    ) -> EmbeddingResult:
        """Embed one or more strings.

        Args:
            texts: Input strings. Vectors are returned in the same order.
            timeout_s: Overrides the provider's default timeout.

        Returns:
            EmbeddingResult with one vector per input.
        """
        ...


class WrapperEmbeddingProvider:
    """Production `EmbeddingProvider` backed by `wrapper.embed`.

    This is the only place in this module that touches the real AI SDK
    boundary; everything else works against the `EmbeddingProvider` protocol.
    """

    def embed(
        self,
        texts: list[str],
        *,
        timeout_s: float | None = None,
    ) -> EmbeddingResult:
        return wrapper_embed(texts, timeout_s=timeout_s)


class DeterministicFakeEmbeddingProvider:
    """A fake `EmbeddingProvider` with no randomness and no network calls.

    The same input text always produces the same vector, and inputs that
    share words produce vectors that are closer together, which is enough
    to exercise retrieval and similarity logic in tests without a live
    model call.
    """

    def __init__(self, dimensions: int = 16) -> None:
        if dimensions <= 0:
            raise ValueError("dimensions must be a positive integer")
        self.dimensions = dimensions

    def embed(
        self,
        texts: list[str],
        *,
        timeout_s: float | None = None,
    ) -> EmbeddingResult:
        if not isinstance(texts, list) or not all(
            isinstance(text, str) for text in texts
        ):
            raise TypeError("texts must be a list[str]")

        vectors = [self._embed_one(text) for text in texts]
        return EmbeddingResult(
            vectors=vectors,
            request_id="fake-deterministic",
            model_alias="embed-fake",
            dimensions=self.dimensions,
            latency_ms=0.0,
        )

    def _embed_one(self, text: str) -> list[float]:
        """Hash each word into a fixed-size bag-of-words vector, then normalize."""
        vector = [0.0] * self.dimensions
        words = text.lower().split()
        for word in words:
            digest = hashlib.sha256(word.encode("utf-8")).hexdigest()
            index = int(digest, 16) % self.dimensions
            vector[index] += 1.0

        norm = math.sqrt(sum(component * component for component in vector))
        if norm > 0:
            vector = [component / norm for component in vector]
        return vector


async def embed_async(
    texts: list[str],
    provider: EmbeddingProvider,
    *,
    timeout_s: float | None = None,
) -> EmbeddingResult:
    """Embed `texts` via an injected `EmbeddingProvider` without blocking the event loop.

    The provider's `embed` is synchronous (real SDK calls and the fake alike),
    so it runs in a worker thread via `asyncio.to_thread`; callers just await
    this function to get the typed `EmbeddingResult` back.

    Args:
        texts: Input strings. Vectors are returned in the same order.
        provider: Any `EmbeddingProvider` (the deterministic fake in tests,
            `WrapperEmbeddingProvider` in production).
        timeout_s: Overrides the provider's default timeout.

    Returns:
        The `EmbeddingResult` produced by `provider`.
    """
    return await asyncio.to_thread(provider.embed, texts, timeout_s=timeout_s)
