"""Tests for the EmbeddingProvider protocol, its deterministic fake, the
async embedding function, structured wrapper configuration, and the typed
DeveloperProfile model and its JSON loader.

Eight pytest cases: the five required categories, plus three more that round
out invalid-input and configuration coverage:
    - valid input
    - invalid input (from a file, directly on the model, and on the fake provider)
    - async behavior
    - injected fake
    - configuration (missing and valid)
"""

import asyncio
from pathlib import Path

import pytest

from python_bridge import (
    DeterministicFakeEmbeddingProvider,
    DeveloperProfile,
    ProfileValidationError,
    embed_async,
    load_profiles_from_file,
)
from wrapper import ConfigurationMissingError, EmbeddingResult, load_config

_REQUIRED_ENV_VARS = ("MODEL_ENDPOINT", "CHAT_MODEL_ALIAS", "EMBEDDING_MODEL_ALIAS")

PROFILES_FILE = Path(__file__).resolve().parent.parent / "data" / "profiles.json"

# Index layout of data/profiles.json: two valid profiles, then one
# deliberately invalid one (blank current_role, negative experience_years).
_VALID_INDICES = [0, 1]
_INVALID_INDEX = 2


def test_load_profiles_accepts_valid_input() -> None:
    profiles = load_profiles_from_file(PROFILES_FILE, indices=_VALID_INDICES)

    assert [p.name for p in profiles] == ["Hamza Khan", "Muhammad Sheharyar"]
    assert all(isinstance(p, DeveloperProfile) for p in profiles)


def test_load_profiles_rejects_invalid_input() -> None:
    """The third entry in data/profiles.json has a blank current_role and a
    negative experience_years — either should be enough to reject it."""
    with pytest.raises(ProfileValidationError) as excinfo:
        load_profiles_from_file(PROFILES_FILE, indices=[_INVALID_INDEX])

    assert "current_role" in str(excinfo.value) or "experience_years" in str(excinfo.value)


def test_developer_profile_rejects_negative_experience_years_directly() -> None:
    """Invalid input on the model itself, not just via the file loader."""
    with pytest.raises(ProfileValidationError):
        DeveloperProfile(
            name="Test Person",
            current_role="Developer",
            experience_years=-1,
            target_ai_role="AI Engineer",
        )


def test_embed_async_behavior_returns_typed_result() -> None:
    provider = DeterministicFakeEmbeddingProvider()
    profiles = load_profiles_from_file(PROFILES_FILE, indices=_VALID_INDICES)
    texts = [p.embedding_text() for p in profiles]

    result = asyncio.run(embed_async(texts, provider))

    assert isinstance(result, EmbeddingResult)
    assert len(result.vectors) == len(texts)


def test_injected_fake_provider_is_used_instead_of_the_real_one() -> None:
    """Retrieval code depends on the EmbeddingProvider protocol, so a fake
    can be injected in place of WrapperEmbeddingProvider for tests."""
    provider = DeterministicFakeEmbeddingProvider()
    profiles = load_profiles_from_file(PROFILES_FILE, indices=_VALID_INDICES)
    texts = [p.embedding_text() for p in profiles]

    result = provider.embed(texts)

    assert result.model_alias == "embed-fake"  # confirms the fake, not a real call, ran
    assert all(len(vector) == provider.dimensions for vector in result.vectors)


def test_fake_provider_rejects_non_list_input() -> None:
    """Invalid input on the embedding side, not just on DeveloperProfile."""
    provider = DeterministicFakeEmbeddingProvider()

    with pytest.raises(TypeError):
        provider.embed("not a list")  # type: ignore[arg-type]


def test_load_config_raises_safe_error_on_missing_configuration(monkeypatch) -> None:
    for var in _REQUIRED_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CHAT_MODEL_ALIAS", "chat-default")
    monkeypatch.setenv("EMBEDDING_MODEL_ALIAS", "embed-default")
    # MODEL_ENDPOINT is left unset.

    with pytest.raises(ConfigurationMissingError) as excinfo:
        load_config()

    assert "MODEL_ENDPOINT" in str(excinfo.value)


def test_load_config_returns_populated_config_with_valid_env_vars(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_ENDPOINT", "https://example.invalid/api")
    monkeypatch.setenv("CHAT_MODEL_ALIAS", "chat-default")
    monkeypatch.setenv("EMBEDDING_MODEL_ALIAS", "embed-default")
    monkeypatch.delenv("MODEL_TIMEOUT_S", raising=False)

    config = load_config()

    assert config.endpoint == "https://example.invalid/api"
    assert config.chat_model_alias == "chat-default"
    assert config.embedding_model_alias == "embed-default"
    assert config.default_timeout_s == 30.0  # default, since MODEL_TIMEOUT_S was unset
