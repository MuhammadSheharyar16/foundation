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

- [x] Create a typed `DeveloperProfile` model with name, current role, experience years and target AI role.
- [x] Load two profiles from JSON and reject one deliberately invalid profile.
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

- [python_bridge.py](python_bridge.py) — one module holding both lab pieces:
  - `EmbeddingProvider` Protocol, `DeterministicFakeEmbeddingProvider`, `WrapperEmbeddingProvider`, and the async `embed_async` function.
  - The `DeveloperProfile` model — with an `embedding_text()` method that turns a profile into the text fed to `EmbeddingProvider` in tests — plus `load_profile()` (one profile from a JSON object file) and `load_profiles_from_file()` (multiple profiles from one JSON array file, optionally by `indices`), raising `ProfileValidationError` that names the invalid/missing field.
- [wrapper/config.py](wrapper/config.py) — `WrapperConfig` and `load_config()`, reading `MODEL_ENDPOINT`, `CHAT_MODEL_ALIAS`, `EMBEDDING_MODEL_ALIAS` (required) and `MODEL_TIMEOUT_S` (optional), raising `ConfigurationMissingError` that names the missing/invalid variable and never its value.
- [wrapper/errors.py](wrapper/errors.py) — `ConfigurationMissingError`, `ModelTimeoutError`, `ModelCallError`.
- [conftest.py](conftest.py) — its presence puts the repo root on `sys.path` so `tests/*.py` can import the modules above no matter how pytest is invoked; also defines the shared `deterministic_model_env` fixture (Day 0B) that every test needing `wrapper.chat`/`wrapper.embed` depends on, plus the `MODEL_ENDPOINT`/`CHAT_MODEL_ALIAS`/`EMBEDDING_MODEL_ALIAS` constants it sets — the one place those values are defined for tests.
- [data/profiles.json](data/profiles.json) — one JSON array with all three `DeveloperProfile` fixtures: `Hamza Khan` and `Muhammad Sheharyar` (valid), then `Ali Nadeem` (deliberately invalid — blank `current_role`, negative `experience_years`). Loaded via `load_profiles_from_file()`; the two valid profiles' text (via `embedding_text()`) is also the embedding input used in the `EmbeddingProvider`/async tests, in place of arbitrary sample sentences.
- [tests/test_python_bridge.py](tests/test_python_bridge.py) — the pytest suite for the `EmbeddingProvider`/async/config items and `DeveloperProfile` above.

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

### EOD note: concepts that required the most effort
1. **Semantic vs. lexical retrieval** — The difficult part was not understanding the definitions, but deciding whether each one should be called deterministic, probabilistic, or contextual. Both methods give the same ranking when the data is the same, so the retrieval step is deterministic. Semantic search can be confusing because it uses embeddings, but that does not automatically make it probabilistic.
2. **Grounding vs. hallucination** — These concepts are closely related. Grounding means the AI answers using the provided documents, while hallucination happens when the AI makes up an answer that is not supported by the documents. It took a few examples to make the difference clear.
3. **Context window vs. chunk vs. overlap** — Each concept is simple on its own, but understanding how they work together is more important. The context window is the maximum amount of text the AI can handle, chunks break large documents into smaller pieces, and overlap helps prevent important information from being lost between chunks. The C001/C002 delivery example makes this easier to understand.

# Day 0B - First AI Calls and Tiny RAG

**Assignment objective:** Make the first approved model and embedding calls, then build the smallest possible evidence-grounded answer flow with citation validation.

There is no live Foundry/Azure endpoint configured for this repo yet — `wrapper.chat`/`wrapper.embed` (see [wrapper/model_client.py](wrapper/model_client.py)) are backed by a deterministic, network-free local model instead of a real SDK call, so the demos below run fully offline and nothing below needs a real credential. Swapping in a real Foundry SDK call later only touches that one file; every demo below keeps working unchanged, since none of them import anything past `wrapper.chat`/`wrapper.embed`.

## 1. Set environment variables

`wrapper/config.py` requires three environment variables before any demo will run: `MODEL_ENDPOINT`, `CHAT_MODEL_ALIAS`, `EMBEDDING_MODEL_ALIAS` (optionally `MODEL_TIMEOUT_S`). Since there's no live endpoint, these are placeholder values naming the deterministic local model — not a credential, and safe to keep in a local, uncommitted `.env` file (already in [.gitignore](.gitignore)).

**Windows PowerShell:**
```powershell
$env:MODEL_ENDPOINT = "local://deterministic"
$env:CHAT_MODEL_ALIAS = "chat-local-deterministic-v1"
$env:EMBEDDING_MODEL_ALIAS = "embed-local-deterministic-v1"
```

**Windows cmd:**
```cmd
set MODEL_ENDPOINT=local://deterministic
set CHAT_MODEL_ALIAS=chat-local-deterministic-v1
set EMBEDDING_MODEL_ALIAS=embed-local-deterministic-v1
```

**macOS/Linux:**
```bash
export MODEL_ENDPOINT="local://deterministic"
export CHAT_MODEL_ALIAS="chat-local-deterministic-v1"
export EMBEDDING_MODEL_ALIAS="embed-local-deterministic-v1"
```

**Optional: a local `.env` file instead.** This repo has no `.env`-loading dependency (`requirements.txt` is just `pytest`), so a `.env` file isn't picked up automatically — create one at the repo root, then load it into the current shell before running a demo:

```
# .env  (not committed — see .gitignore)
MODEL_ENDPOINT=local://deterministic
CHAT_MODEL_ALIAS=chat-local-deterministic-v1
EMBEDDING_MODEL_ALIAS=embed-local-deterministic-v1
```

```powershell
# PowerShell: load .env into the current session
Get-Content .env | ForEach-Object {
    if ($_ -match '^\s*([^#=]+)=(.*)$') { Set-Item "env:$($matches[1].Trim())" $matches[2].Trim() }
}
```

```bash
# macOS/Linux: load .env into the current shell
set -a; source .env; set +a
```

The pytest suite never needs any of this — every test that calls `chat()`/`embed()` depends on the shared `deterministic_model_env` fixture in [conftest.py](conftest.py) (which sets the env vars via `monkeypatch` for the duration of that one test), so `pytest -q` passes on a clean checkout with nothing exported.

## 2. Run the demos

With the three environment variables set (previous section):

```
pytest -q
python model_demo.py
python embedding_demo.py
python tiny_rag.py --question "What is the supplier delivery policy?"
python tiny_rag.py --question "What is the CEO salary?"
```

- `pytest -q` — the full suite (20 cases across `tests/test_python_bridge.py`, `tests/test_embedding_demo.py`, `tests/test_tiny_rag.py`), no environment variables required.
- `model_demo.py` — three-bullet summary of `data/documents/DOC-002-...md` via `wrapper.chat`; prints sanitized metadata only, writes `artifacts/model_run.txt`.
- `embedding_demo.py` — embeds S1/S2/S3 from `data/sample_questions.json`, ranks all three pairs by cosine similarity, confirms S1-S2 ranks above S1-S3, writes `artifacts/embedding_run.txt`.
- `tiny_rag.py --question "..."` — retrieves the top two chunks from `data/chunks.json`, answers with citations or refuses with `INSUFFICIENT_EVIDENCE`; the two commands above are the mandatory pair and, run in that order (Q1 then Q2), leave `artifacts/supported_answer.json` and `artifacts/insufficient_evidence.json` holding their respective results. `tiny_rag.py` also accepts any other question, e.g. `--question "How much notice is needed to avoid a contract renewal?"` or `--question "What is the late payment interest rate?"`, but running one after Q1/Q2 will overwrite whichever of the two required artifact files shares its outcome status — re-run Q1/Q2 last if that happens.

## Day 0B acceptance

- [x] The model call succeeds through the supplied wrapper with sanitized metadata.
- [x] S1-S2 ranks above S1-S3 in the embedding demonstration.
- [x] The supported RAG answer cites only retrieved chunk IDs.
- [x] A fabricated citation is rejected by a test.
- [x] The unanswerable question returns insufficient evidence without citations.
- [x] No agent framework, production data or committed credential is used.

**EOD submission:** Open a PR containing the model, embedding and tiny RAG demos, tests, sanitized outputs and a one-paragraph explanation of retrieval versus generation.

### EOD note: retrieval versus generation
Retrieval and generation are two separate steps, and each has a different job.

Retrieval finds the right information. In `tiny_rag.py`, the question and all chunks from data/`chunks.json` are compared using cosine similarity. The system then keeps only the top two chunks. This means the model can only use those two chunks as evidence.

Generation takes that retrieved information and turns it into a clear answer. `wrapper.chat` receives only the selected chunks, along with their chunk IDs, and creates the response. The chunk IDs make it easy to know where each piece of information came from.

After the answer is created, `validate_citations()` checks the citations. If the model uses a citation that was not part of the two retrieved chunks, the answer is rejected.

If retrieval cannot find useful evidence, the model is not asked to guess. For example, if the question is **"What is the CEO salary?"** and there is no relevant information, the result is `INSUFFICIENT_EVIDENCE`. The same happens if a chunk talks about late payments but does not give the late payment interest rate.
