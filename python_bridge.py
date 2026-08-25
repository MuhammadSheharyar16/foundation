"""Bridge between lab retrieval code and an embedding backend.

Retrieval code should depend on the `EmbeddingProvider` protocol below, never
on a concrete backend directly. That keeps retrieval logic (chunking,
similarity, ranking) testable with a deterministic fake, and lets the real
`wrapper.embed` be swapped in for production without touching callers.

Also holds `DeveloperProfile`, the typed shape other lab code should depend
on instead of raw dicts. Profiles are loaded from JSON files under `data/`
and validated eagerly in `__post_init__`, so a malformed profile fails at
load time with a clear message instead of surfacing as a confusing error
further downstream.
"""

import asyncio
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from wrapper import EmbeddingResult
from wrapper import embed as wrapper_embed
from wrapper.model_client import _EMBEDDING_DIMENSIONS, hash_embed_vector


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

    def __init__(self, dimensions: int = _EMBEDDING_DIMENSIONS) -> None:
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
        """Delegates to `wrapper.model_client.hash_embed_vector`, the one real
        implementation of this hashing algorithm, so this fake and the real
        `wrapper.embed` always agree on what "similar" means."""
        return hash_embed_vector(text, self.dimensions)


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


class ProfileValidationError(ValueError):
    """Raised when a profile fails validation.

    Message names the invalid field(s), never the whole raw payload.
    """


@dataclass(frozen=True)
class DeveloperProfile:
    """A developer's current standing and the AI role they are targeting.

    Attributes:
        name: Full name. Must be non-empty after stripping whitespace.
        current_role: Current job title. Must be non-empty after stripping.
        experience_years: Years of professional experience. Must be a
            non-negative number; fractional years (e.g. 6.5) are allowed.
        target_ai_role: The AI role the developer is aiming for. Must be
            non-empty after stripping.
    """

    name: str
    current_role: str
    experience_years: float
    target_ai_role: str

    def __post_init__(self) -> None:
        _require_non_empty_str("name", self.name)
        _require_non_empty_str("current_role", self.current_role)
        _require_non_empty_str("target_ai_role", self.target_ai_role)

        if isinstance(self.experience_years, bool) or not isinstance(
            self.experience_years, (int, float)
        ):
            raise ProfileValidationError("experience_years must be a number")
        if self.experience_years < 0:
            raise ProfileValidationError("experience_years must not be negative")

    def embedding_text(self) -> str:
        """Canonical text for feeding this profile into an `EmbeddingProvider`.

        Used instead of arbitrary sample sentences so embedding tests exercise
        the same typed data the rest of the lab already loads and validates.
        """
        return (
            f"{self.current_role} with {self.experience_years} years of "
            f"experience targeting {self.target_ai_role}"
        )


def _require_non_empty_str(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ProfileValidationError(f"{field_name} must be a non-empty string")


_REQUIRED_FIELDS = ("name", "current_role", "experience_years", "target_ai_role")


def load_profile(path: Path) -> DeveloperProfile:
    """Load and validate one `DeveloperProfile` from a JSON file.

    Args:
        path: Path to a JSON file shaped like
            {"name": ..., "current_role": ..., "experience_years": ...,
             "target_ai_role": ...}.

    Returns:
        A validated DeveloperProfile.

    Raises:
        ProfileValidationError: The file can't be read, isn't valid JSON,
            isn't a JSON object, is missing a required field, or a field
            fails validation.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProfileValidationError(f"Could not read profile file: {path}") from exc

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ProfileValidationError(f"Profile file is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ProfileValidationError(f"Profile file must contain a JSON object: {path}")

    return _build_profile_from_payload(payload)


def _build_profile_from_payload(payload: object) -> DeveloperProfile:
    """Validate one raw profile entry (already-parsed JSON) into a `DeveloperProfile`."""
    if not isinstance(payload, dict):
        raise ProfileValidationError("Profile entry must be a JSON object")

    missing = [field for field in _REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ProfileValidationError(
            f"Profile is missing required field(s): {', '.join(missing)}"
        )

    return DeveloperProfile(
        name=payload["name"],
        current_role=payload["current_role"],
        experience_years=payload["experience_years"],
        target_ai_role=payload["target_ai_role"],
    )


def load_profiles_from_file(
    path: Path, *, indices: list[int] | None = None
) -> list[DeveloperProfile]:
    """Load and validate developer profiles from one JSON file holding a list.

    This is the primary multi-profile loader — the lab's fixtures live
    together in a single `data/profiles.json` array rather than one file per
    profile.

    Args:
        path: Path to a JSON file containing a JSON array of profile objects,
            each shaped like {"name": ..., "current_role": ...,
            "experience_years": ..., "target_ai_role": ...}.
        indices: If given, only these zero-based positions are loaded, in the
            order given. Otherwise every entry is loaded, in file order.

    Returns:
        Validated DeveloperProfile objects, in the order requested.

    Raises:
        ProfileValidationError: The file can't be read, isn't valid JSON,
            isn't a JSON array, a requested index is out of range, or an
            entry (the first one encountered, in the order requested) fails
            validation.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProfileValidationError(f"Could not read profiles file: {path}") from exc

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ProfileValidationError(f"Profiles file is not valid JSON: {path}") from exc

    if not isinstance(payload, list):
        raise ProfileValidationError(f"Profiles file must contain a JSON array: {path}")

    selected_indices = range(len(payload)) if indices is None else indices

    profiles = []
    for index in selected_indices:
        if not (0 <= index < len(payload)):
            raise ProfileValidationError(
                f"Profiles file has no entry at index {index}: {path}"
            )
        profiles.append(_build_profile_from_payload(payload[index]))
    return profiles
