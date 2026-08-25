# Its presence makes pytest add the repo root to sys.path during collection,
# so `tests/*.py` can `import python_bridge` / `import wrapper` regardless of
# how pytest is invoked (`pytest`, `.venv\Scripts\pytest.exe`, or
# `python -m pytest`) or whether the venv is activated.

import pytest

# The one place the deterministic-local-model env var values are defined.
# wrapper.chat/wrapper.embed have no live Foundry/Azure endpoint yet (see
# wrapper/model_client.py), so these are placeholder values naming that
# local model -- not a credential. Every test across tests/*.py that needs
# wrapper.chat()/wrapper.embed() to succeed should depend on the
# `deterministic_model_env` fixture below instead of setting its own env
# vars, so there's exactly one place to change if these values ever change.
# Tests that assert on the resulting `model_alias` should import
# CHAT_MODEL_ALIAS/EMBEDDING_MODEL_ALIAS from here too, rather than
# retyping the literal, e.g.:
#
#     from conftest import CHAT_MODEL_ALIAS
#     def test_x(deterministic_model_env):
#         assert chat("...").sanitized_metadata()["model_alias"] == CHAT_MODEL_ALIAS
MODEL_ENDPOINT = "local://deterministic"
CHAT_MODEL_ALIAS = "chat-local-deterministic-v1"
EMBEDDING_MODEL_ALIAS = "embed-local-deterministic-v1"


@pytest.fixture
def deterministic_model_env(monkeypatch):
    """Sets MODEL_ENDPOINT/CHAT_MODEL_ALIAS/EMBEDDING_MODEL_ALIAS for the
    duration of one test. Use as a fixture argument, e.g.:

        def test_something(deterministic_model_env):
            result = chat("...")
    """
    monkeypatch.setenv("MODEL_ENDPOINT", MODEL_ENDPOINT)
    monkeypatch.setenv("CHAT_MODEL_ALIAS", CHAT_MODEL_ALIAS)
    monkeypatch.setenv("EMBEDDING_MODEL_ALIAS", EMBEDDING_MODEL_ALIAS)
