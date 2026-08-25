# Submission Audit — AICO Pre-Sprint Foundation Assignment

**Audited:** 2026-08-25
**Against:** `AICO_PreSprint_Foundation_Assignment 1.pdf`, v1.0, 24 Aug 2026
**Verified by:** running `pytest -q` in this repo (`.venv/Scripts/python -m pytest -q`) →
**20 passed**, no failures, no environment variables required.

## Bottom line

Nothing required is missing from the code, tests, or docs. Every acceptance checkbox in
the PDF (Day 0A, Day 0B, AICO map, final submission quality checks) is met, and several
items go beyond the minimum (14 tests just for the Day 0A bridge module, a narrow
quantified-evidence refusal guard beyond the base insufficient-evidence case, sanitized
metadata written to artifacts as well as printed). Two things are **not** code gaps but
are worth closing before you call this done — see [Gaps found](#gaps-found).

---

## 1. Required folder structure — present

```
foundation/                    (this repo root = "foundation/")
  README.md                 ✅ present, combines Day 0A + Day 0B instructions and EOD notes
  concepts.md                ✅ present
  python_bridge.py           ✅ present
  model_demo.py               ✅ present
  embedding_demo.py           ✅ present
  tiny_rag.py                 ✅ present
  aico-map.md                 ✅ present
  data/
    documents/                ✅ 3 synthetic docs (DOC-001, DOC-002, DOC-003)
    chunks.json                ✅ 5 chunks, C001–C005, stable IDs
    profiles.json               ✅ 2 valid + 1 deliberately invalid profile
    sample_questions.json        ✅ Q1–Q4, S1–S3, plus planted edge cases
  tests/
    test_python_bridge.py     ✅
    test_embedding_demo.py    ✅
    test_tiny_rag.py          ✅
  artifacts/
    model_run.txt             ✅ sanitized metadata + 3-bullet summary
    embedding_run.txt         ✅ ranked similarity table, no raw vectors
    supported_answer.json     ✅ ANSWERED path
    insufficient_evidence.json ✅ refusal path
  wrapper/  (lead-provided model wrapper boundary)
    config.py, errors.py, model_client.py, results.py, __init__.py  ✅
```

Not part of the PDF's listed structure but present and load-bearing: `conftest.py`
(puts repo root on `sys.path`, defines the shared `deterministic_model_env` fixture),
`setup.ps1` / `setup.sh` (one-command bootstrap), `.env` (gitignored, placeholder
values only, not committed — verified with `git ls-files`).

---

## 2. Day 0A — AI Concepts and Python Bridge

| Acceptance criterion | Status | Evidence |
|---|---|---|
| Clean checkout can create env + run tests from README | ✅ | `setup.ps1`/`setup.sh`; `pytest -q` runs with zero env vars set |
| 5+ tests pass | ✅ | 14 tests in `test_python_bridge.py` alone (valid/invalid input, async, injected fake, config missing/valid) |
| Invalid typed input fails with a clear validation message | ✅ | `ProfileValidationError` names the failing field; tested directly on the model and via the file loader |
| Fake provider injected; app code never creates it internally | ✅ | `EmbeddingProvider` Protocol + `DeterministicFakeEmbeddingProvider`, constructor-injected in tests |
| Concept explanations original, explainable without the file | ✅ | `concepts.md` — plain-language, supplier examples, deterministic/probabilistic/contextual labels for all 17 required terms |
| No secret or production data | ✅ | synthetic data only; `.env` not tracked |

`DeveloperProfile` (name, current_role, experience_years, target_ai_role), the async
`embed_async`, and env-var configuration with a safe missing-configuration error are all
implemented in [python_bridge.py](python_bridge.py) / [wrapper/config.py](wrapper/config.py).

## 3. Day 0B — First AI Calls and Tiny RAG

| Acceptance criterion | Status | Evidence |
|---|---|---|
| Model call succeeds through the wrapper with sanitized metadata | ✅ | `model_demo.py` → `wrapper.chat()` only; prints/writes `request_id`, `model_alias`, `latency_ms`, token counts — never the document body as "metadata" |
| S1-S2 ranks above S1-S3 | ✅ | `artifacts/embedding_run.txt`: S1-S2 = 0.8000, S1-S3 = 0.0000 |
| Supported RAG answer cites only retrieved chunk IDs | ✅ | `validate_citations()`, tested in `test_supported_answer_cites_only_retrieved_chunk_ids` |
| Fabricated citation rejected by a test | ✅ | `test_fabricated_citation_is_rejected` — covers a real-but-not-retrieved chunk ID and a nonexistent one |
| Unanswerable question → `INSUFFICIENT_EVIDENCE`, no citations | ✅ | `artifacts/insufficient_evidence.json` for "What is the CEO salary?"; tested |
| No agent framework, production data, or committed credential | ✅ | no LangChain/Semantic Kernel import anywhere; `wrapper/` is the only model boundary; no key/token in code or `.env` |

Extras beyond the minimum: a second, narrower refusal path
(`_requires_quantified_evidence` in `tiny_rag.py`) that catches a topically-relevant
chunk (late payment) that never states a rate — this maps to the "near-miss gap" the
lead planted in `sample_questions.json`. Timeout handling is real and tested
(`test_chat_raises_timeout_error_when_budget_exceeded`), not just documented.

## 4. `aico-map.md` — AICO mapping

All 11 rows of the required component table are addressed, each explicitly marked
**implemented**, **partially implemented**, or **not implemented (later build day)** —
the PDF only asks that the developer *identify* what plays each role, and this goes
further by being honest about what's simulated vs. real. The four required
explanations (sequence diagram, Mode A vs Mode B, why the model can't call a tool or
query Mode B directly, and one risk each for latency/security/hallucination) are all
present and specific to this codebase rather than generic.

## 5. Submission quality checks (final page)

| Check | Status |
|---|---|
| README gives clean setup + run commands for every demo | ✅ |
| All tests pass from one command | ✅ (`pytest -q` → 20 passed) |
| Sanitized artifacts show both success and refusal paths | ✅ |
| No credential/token/secret/production data | ✅ |
| Full folder in one reviewable PR with small commits | ⚠️ see below |

---

## Gaps found

These are the only two items I'd close before calling this submission-ready — neither
requires new code.

1. **No PR was opened on GitHub.** `git log` shows 11 commits (`init` → `Day0A` fixes →
   `step1`–`step7` → `final step` → `aico-map`) committed straight to `main`, and
   `git branch -a` shows only `main` / `origin/main` — no feature branch, no open or
   merged PR. The PDF's EOD submission instructions for Day 0A, Day 0B, *and* the final
   submission all explicitly say "Open a PR." The commit history itself is small and
   readable (good), but as it stands there's nothing for a reviewer to open on GitHub.
   **Fix:** if this history hasn't been pushed/reviewed yet, push a branch and open one
   PR against `main` covering the whole folder (the final page allows a single PR with
   multiple small commits, so you don't need to unwind history).

2. **`aico-map.md` references a file that doesn't exist.** §1 and §2 both cite
   `final-submission-guide.md §5 Step 2` for a traced 15-step run, but no
   `final-submission-guide.md` exists anywhere in the repo or its history (checked via
   `git log --all -- "*final-submission*"`). Either that file was meant to be part of
   the submission and got dropped, or the reference should be replaced with something
   that actually exists (e.g. inline the trace, or point at the artifact files).
   **Fix:** create the referenced file, or edit the two citations in `aico-map.md` to
   point at real evidence.

## Not a gap, but worth doing before the review gate

The PDF's final page includes a **Developer report template** (What I completed /
commands run / evidence / hours / blockers / 3 concepts clarified / 1 open question /
readiness confidence) for the 15-minute lead review. It isn't listed in the required
folder structure, so its absence isn't a submission gap — but you'll want it filled out
going into the review meeting itself, since the reviewer can ask you to walk through it.
