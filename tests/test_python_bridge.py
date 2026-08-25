"""Tests for the EmbeddingProvider protocol, its deterministic fake, the
async embedding function, and structured wrapper configuration.

Categories covered (per the assignment checklist):
    - valid input
    - invalid input
    - async behavior
    - injected fake
    - missing configuration
"""

import asyncio
import json
from pathlib import Path

import pytest

from python_bridge import DeterministicFakeEmbeddingProvider, EmbeddingProvider, embed_async
from wrapper import ConfigurationMissingError, EmbeddingResult, load_config

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def _load_embedding_sentences() -> list[str]:
    """Real example text from data/sample_questions.json, not arbitrary strings."""
    payload = json.loads((DATA_DIR / "sample_questions.json").read_text(encoding="utf-8"))
    return [item["text"] for item in payload["embedding_sentences"]]


# ---------------------------------------------------------------------------
# Valid input
# ---------------------------------------------------------------------------


def test_fake_provider_with_valid_input_returns_typed_result() -> None:
    provider: EmbeddingProvider = DeterministicFakeEmbeddingProvider()
    sentences = _load_embedding_sentences()  # S1, S2, S3 from the data folder

    result = provider.embed(sentences)

    assert isinstance(result, EmbeddingResult)
    assert len(result.vectors) == len(sentences)
    assert all(len(vector) == provider.dimensions for vector in result.vectors)


def test_fake_provider_is_deterministic_across_calls() -> None:
    provider = DeterministicFakeEmbeddingProvider()
    sentences = _load_embedding_sentences()

    first = provider.embed(sentences)
    second = provider.embed(sentences)

    assert first.vectors == second.vectors


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


def test_fake_provider_rejects_non_list_input() -> None:
    provider = DeterministicFakeEmbeddingProvider()

    with pytest.raises(TypeError):
        provider.embed("not a list")  # type: ignore[arg-type]


def test_fake_provider_rejects_non_string_items() -> None:
    provider = DeterministicFakeEmbeddingProvider()

    with pytest.raises(TypeError):
        provider.embed(["ok", 123])  # type: ignore[list-item]


def test_fake_provider_rejects_non_positive_dimensions() -> None:
    with pytest.raises(ValueError):
        DeterministicFakeEmbeddingProvider(dimensions=0)


# ---------------------------------------------------------------------------
# Async behavior + injected fake
# ---------------------------------------------------------------------------


def test_embed_async_returns_typed_result_using_injected_fake() -> None:
    provider = DeterministicFakeEmbeddingProvider()
    sentences = _load_embedding_sentences()

    result = asyncio.run(embed_async(sentences, provider))

    assert isinstance(result, EmbeddingResult)
    assert result.model_alias == "embed-fake"  # confirms the fake, not a real call, ran
    assert len(result.vectors) == len(sentences)


def test_embed_async_matches_the_sync_fake_result() -> None:
    """The async wrapper must not change what the injected provider produces."""
    provider = DeterministicFakeEmbeddingProvider()
    sentences = _load_embedding_sentences()

    sync_result = provider.embed(sentences)
    async_result = asyncio.run(embed_async(sentences, provider))

    assert async_result.vectors == sync_result.vectors


# ---------------------------------------------------------------------------
# Missing configuration
# ---------------------------------------------------------------------------

_REQUIRED_ENV_VARS = ["MODEL_ENDPOINT", "CHAT_MODEL_ALIAS", "EMBEDDING_MODEL_ALIAS"]


def test_load_config_raises_safe_error_when_a_required_var_is_missing(monkeypatch) -> None:
    for var in _REQUIRED_ENV_VARS:
        monkeypatch.delenv(var, raising=False)
    monkeypatch.setenv("CHAT_MODEL_ALIAS", "chat-default")
    monkeypatch.setenv("EMBEDDING_MODEL_ALIAS", "embed-default")
    # MODEL_ENDPOINT is left unset.

    with pytest.raises(ConfigurationMissingError) as excinfo:
        load_config()

    assert "MODEL_ENDPOINT" in str(excinfo.value)


def test_load_config_raises_when_all_required_vars_are_missing(monkeypatch) -> None:
    for var in _REQUIRED_ENV_VARS:
        monkeypatch.delenv(var, raising=False)

    with pytest.raises(ConfigurationMissingError):
        load_config()


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
