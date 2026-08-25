# Day 0A - AI Concepts and Python Bridge

**Assignment objective:** Establish the vocabulary and Python mechanics needed to complete the AI build tasks without copying code blindly.

## 1. Configure and verify the environment

| Action | Windows example | macOS/Linux example |
|---|---|---|
| Create environment | `py -m venv .venv` | `python3 -m venv .venv` |
| Activate | `.venv\Scripts\activate` | `source .venv/bin/activate` |
| Install | `pip install -r requirements.txt` | `pip install -r requirements.txt` |
| Verify | `pytest -q` | `pytest -q` |

**Quick setup:** the steps above are also scripted, so a clean checkout can bootstrap and verify itself in one command:

- Windows: `powershell -File setup.ps1` (or `pwsh -File setup.ps1`)
- macOS/Linux: `bash setup.sh`

Each script creates `.venv` if it doesn't exist, installs `requirements.txt`, and runs `pytest -q`.

### Running tests

With the virtual environment activated:

```
pytest -q
```

Without activating (Windows):

```
.venv\Scripts\pytest.exe -q
```

Without activating (macOS/Linux):

```
.venv/bin/pytest -q
```

## 2. Build the Python bridge lab

- [ ] Create a typed `DeveloperProfile` model with name, current role, experience years and target AI role.
- [ ] Load two profiles from JSON and reject one deliberately invalid profile.
- [x] Define an `EmbeddingProvider` Protocol and inject a deterministic fake implementation.
- [x] Create one async function that calls the fake provider and returns a typed result.
- [x] Add structured configuration from environment variables with a safe missing-configuration error.
- [x] Write at least five pytest cases covering valid input, invalid input, async behavior, injected fake and missing configuration.

| Existing experience | Python equivalent to implement |
|---|---|
| C# class / Dart model | Pydantic model or dataclass |
| Interface | Protocol or abstract base class |
| Task / Future | `async def` and `await` |
| xUnit / Flutter test | pytest |
| Dependency injection | Constructor injection with a fake dependency in tests |

### Where things live

- [python_bridge.py](python_bridge.py) — `EmbeddingProvider` Protocol, `DeterministicFakeEmbeddingProvider`, `WrapperEmbeddingProvider`, and the async `embed_async` function.
- [wrapper/config.py](wrapper/config.py) — `WrapperConfig` and `load_config()`, reading `MODEL_ENDPOINT`, `CHAT_MODEL_ALIAS`, `EMBEDDING_MODEL_ALIAS` (required) and `MODEL_TIMEOUT_S` (optional), raising `ConfigurationMissingError` that names the missing/invalid variable and never its value.
- [wrapper/errors.py](wrapper/errors.py) — `ConfigurationMissingError`, `ModelTimeoutError`, `ModelCallError`.
- [tests/test_python_bridge.py](tests/test_python_bridge.py) — the pytest suite for the items above.

## 3. Write the concept baseline

In [concepts.md](concepts.md): LLM, prompt, system instruction, token, context window, temperature, embedding, cosine similarity, chunk, overlap, lexical retrieval, semantic retrieval, RAG, grounding, hallucination, provenance and citation are each explained in plain language with a procurement/supplier example and labeled deterministic, probabilistic or contextual, plus the basic flow (question → retrieve → chunks → prompt → LLM → validate → answer).

## Day 0A acceptance

- [x] A clean checkout can create the environment and run tests from this README.
- [x] All five or more tests pass.
- [x] Invalid typed input fails with a clear validation message.
- [x] The fake provider is injected; no application code creates it internally.
- [x] Concept explanations are original and can be explained without reading the file.
- [x] No secret or production data is present.

**EOD submission:** Open a PR containing the Day 0A files, test output and a short note identifying the three concepts that required the most effort.
