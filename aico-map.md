# AICO Map — Tiny RAG Lab

This document maps the Day 0B tiny RAG lab (`tiny_rag.py`, `wrapper/`, `data/chunks.json`) onto
the AICO architecture. The lab does not reimplement AICO — it stands in for one small, single-user,
single-lane slice of it. For each component, this says what plays that role here, or says plainly
that nothing does and the piece belongs to a later build day.

Every claim below is backed by a real traced run of
`python tiny_rag.py --question "What is the supplier delivery policy?"` (see §2), not guessed from
the assignment brief.

## 1. Component table

| AICO component | What plays that role in this lab |
|---|---|
| **User and API entry** | The `--question` CLI argument is the request (`_parse_args()`). There is no user/tenant identity and no request ID minted here — `argparse` stands in for an API layer, nothing more. The wrapper mints its own `request_id` per call (e.g. `local-d044de35ec64`), but that belongs to the Model Gateway, not this boundary. |
| **Gate-A (intent/domain classification)** | Not implemented. The script never asks "is this a RAG question, small talk, or something to block" — every `--question` value is treated as a RAG question unconditionally. |
| **Lane Selector** | Not implemented. There is exactly one lane. Which lane you exercise is decided by which Python file you run (`tiny_rag.py` vs `model_demo.py` vs `embedding_demo.py`), not by a runtime decision over one incoming request. |
| **Mode A** | Implemented as `_SYSTEM_INSTRUCTION`. It covers three of the four things Mode A should define: **allowed intent** ("You are a procurement policy assistant"), **citation rule** (keep every `[Cxxx]` tag attached, never invent a chunk ID), and an **answer template** for refusals (say `INSUFFICIENT_EVIDENCE`, cite nothing). Missing: **definitions** — there's no glossary for terms like "vendor" or "delivery window"; the model relies entirely on how the evidence chunks happen to word them. This string never changes between runs, which is what makes it policy — it governs *how* the model behaves, regardless of the question. |
| **Gate-B (permission/tenant/PII/safe-disclosure)** | Not implemented. Nothing checks who is asking or whether a chunk is safe to disclose before retrieval runs — every question can see all five chunks. This only looks safe because the documents were pre-screened as synthetic, not because a control exists in code. |
| **Mode B** | Implemented as `data/chunks.json`, loaded by `load_chunks()`. The five chunks (C001–C005) are the facts the answer is allowed to draw on — "what is true" for this lab. |
| **Gate-C (evidence source, provenance, freshness, completeness)** | Partially implemented. Each chunk carries `source_document` and `section` (provenance). Two real checks run before generation: a zero-similarity refusal (`top_score <= epsilon`) and a narrow quantified-evidence guard (`_requires_quantified_evidence` / `_has_quantified_evidence`) that catches a chunk that's on-topic but never states a number. There is no freshness check — `sample_questions.json` itself notes DOC-001 says 90 days and DOC-002 says 60 days, and nothing here would catch that conflict. |
| **Model Gateway** | Implemented as `wrapper/` (`wrapper.chat()`, `wrapper.embed()`). Every demo file imports only these two functions — never `wrapper.model_client` internals or an SDK directly. It's currently backed by a deterministic, offline stub instead of a live endpoint (no credentials exist yet), but the boundary is real: swapping in a live SDK call later touches one file, and nothing that calls `wrapper.chat`/`wrapper.embed` has to change. |
| **Gate-D (final citation, quality, disclosure, latency-budget)** | Partially implemented. `parse_citations()` + `validate_citations()` is real and tested — a citation naming a chunk that wasn't actually retrieved gets dropped, not passed through (`tests/test_tiny_rag.py` covers this). Latency is measured (`latency_ms`) but never compared against a budget. There's no disclosure or quality scoring beyond the citation check. |
| **Tool Registry** | Not implemented — later build day. There are no tools or function calls in this lab, so none of a registry's jobs apply yet: no **definitions** to register, no **schemas** to validate, no **versions** to pin, no **ownership/policy** to check. |
| **MCP Gateway** | Not implemented — later build day, and explicitly excluded by the assignment rules (no agent framework). There's no **transport** carrying a tool call anywhere, so there's nothing for **controlled execution** to sandbox or approve. |
| **Observability** | Partially implemented. Sanitized metadata (`request_id`, `model_alias`, `latency_ms`, token counts) is printed and written to `artifacts/model_run.txt` / `embedding_run.txt`; the RAG result (`status`, `citations`, `retrieved_chunk_ids`) goes to `artifacts/supported_answer.json` / `insufficient_evidence.json`. There is no central log store and no correlation across runs — each artifact is a standalone snapshot. |

## 2. Sequence

```
user (--question)
  -> Gate-A         [not implemented: no classification, single hardcoded lane]
  -> Lane Selector  [not implemented: always treated as a RAG question]
  -> Gate-B         [not implemented: no permission/tenant/PII check]
  -> retrieve Mode B   [load_chunks() + rank_chunks() -- embed question + all
                        chunks via wrapper.embed, cosine-rank, keep top 2]
  -> Gate-C         [partial: zero-similarity refusal + quantified-evidence
                      guard run; provenance recorded; no freshness check]
  -> Model Gateway  [wrapper.chat(evidence_prompt, system=Mode A + question)]
  -> Gate-D         [partial: parse_citations() + validate_citations() enforced
                      and tested; no latency-budget enforcement]
  -> answer         [RagResult: answer, citations, retrieved_chunk_ids, status]
  -> audit          [partial: sanitized console print + artifacts/*.json;
                      no central log or cross-run correlation]
```

The console output only shows the start (the question) and the end (status/citations/answer).
Everything from Gate-A through Gate-D happens silently in between — which is exactly why several
rows above say "not implemented" rather than "runs but isn't visible."

### Traced run, step by step

`run()` only prints the question and the final result. To see what happens in between, the same
functions `answer_question()` calls were run directly, in order, with a print after each one — no
logic changed, only observed. Every value below is copied from that real run.

| # | Step | What happened | AICO mapping |
|---|---|---|---|
| 1 | Question received | `question = "What is the supplier delivery policy?"` | User and API entry |
| 2 | Mode B loaded | `load_chunks()` reads 5 chunk IDs: `['C001', 'C002', 'C003', 'C004', 'C005']` | Mode B |
| 3 | Question embedded | `wrapper.embed([question])` → `request_id=local-1dc1b157bbd2`, `dimensions=512`, `latency_ms=0.199` | Model Gateway |
| 4 | Chunks embedded | `wrapper.embed(chunks)` → `request_id=local-f7a36946f8b9`, `latency_ms=2.176` | Model Gateway |
| 5 | All 5 chunks ranked | Cosine similarity to the question: `C002=0.2545, C001=0.2395, C004=0.2282, C005=0.1485, C003=0.0845` | retrieve Mode B |
| 6 | Top 2 kept | `retrieved_chunk_ids = ['C002', 'C001']` — C002 (late/notify) edges out C001 (base delivery window) | retrieve Mode B |
| 7 | Zero-similarity guard | `top_score = 0.2545 > 1e-9` → not refused. (The CEO-salary question scores `0.0` and is refused here instead — see `artifacts/insufficient_evidence.json`.) | Gate-C |
| 8 | Quantified-evidence guard | Doesn't apply — the question contains none of `rate`/`percent`/`percentage` | Gate-C |
| 9 | Evidence prompt assembled | Every sentence of C002 then C001 tagged with its chunk ID → 11 tagged sentences | retrieve Mode B → Model Gateway |
| 10 | System prompt assembled | `_SYSTEM_INSTRUCTION` + `"QUESTION: {question}"` → 390 characters, kept separate from the evidence | Mode A |
| 11 | Model called | `wrapper.chat(...)` → `request_id=local-bd7fb244f1d4`, `latency_ms=0.378`, `prompt_tokens=217`, `completion_tokens=93` | Model Gateway |
| 12 | Raw answer returned | Three `[Cxxx]`-tagged bullets, two from C002 and one from C001 | Model Gateway output |
| 13 | Citations parsed | `parse_citations()` extracts `['C002', 'C001']` | Gate-D |
| 14 | Citations validated | `validate_citations(['C002', 'C001'], ['C002', 'C001'])` → both survive, nothing dropped | Gate-D |
| 15 | Final result | `status = "ANSWERED"`, `citations = ['C002', 'C001']` — matches `artifacts/supported_answer.json` | answer → audit |

What this confirms:

- **Gate-A / Lane Selector / Gate-B genuinely run nothing.** There's no step between #1 and #2
  where classification, lane choice, or a permission check could have happened.
- **Gate-C is real but narrow.** Steps #7 and #8 are actual pass/fail checks. The same trace for
  `"What is the CEO salary?"` refuses at step #7 (`top_score = 0.0`); for `"What is the late
  payment interest rate?"` it passes step #7 but refuses at step #8. Both cases are covered by
  `tests/test_tiny_rag.py`.
- **Gate-D is a real, tested control.** Step #14 is a no-op here because the model behaved, but
  `tests/test_tiny_rag.py::test_fabricated_citation_is_rejected` exercises the case where it isn't
  — a citation outside `retrieved_chunk_ids` gets dropped.
- **Observability is a snapshot, not a log.** Steps #3, #4, and #11 each mint their own
  `request_id`; nothing correlates them into one trace ID for the overall question.

## 3. Mode A vs. Mode B

Mode A (`_SYSTEM_INSTRUCTION`) is the same string on every run, no matter the question or which
chunks get retrieved. It's policy: cite tags, refuse when evidence doesn't support an answer, never
invent a chunk ID.

Mode B (`chunks.json`) is the opposite — it changes meaning depending on what's retrieved for
*this* question. Ask about delivery and Mode B contributes C001/C002; ask about renewal notice and
it contributes C003.

They vary independently: reword Mode A's citation rule and Mode B's facts don't change; swap which
chunks are in Mode B and Mode A's instruction still applies exactly as written. That's because they
answer two different questions — Mode A answers "how am I allowed to behave," Mode B answers "what
am I allowed to claim."

## 4. Why the model can't touch a tool or Mode B directly

`wrapper.chat()` never receives a database handle, a file path, or a callable — only an
already-assembled prompt string and a system string, and it returns text. Every step that touches
Mode B (loading `chunks.json`, embedding, ranking, picking the top two, validating citations
afterward) happens in plain Python, before the model is called and after it returns — entirely
outside the model's reach.

If the model could query Mode B or invoke a tool directly, that separation disappears: there'd be
no fixed point where Gate-C or Gate-D could run. The gates only work because they sit on a path the
model doesn't choose and can't skip. Once the model can reach evidence or a tool on its own, a
fabricated citation or an out-of-scope query has no checkpoint left to catch it.

## 5. Implemented, simulated, or later

- **Implemented and tested:** Mode A instruction; Mode B evidence file; the Model Gateway boundary
  (`wrapper.chat`/`wrapper.embed` only); citation validation (with a fabricated-citation test); the
  zero-similarity and quantified-evidence refusal guards; sanitized observability metadata.
- **Simulated — true only because the lab is small:** Gate-B's "nothing sensitive leaks" outcome,
  which holds only because the documents were pre-screened, not because a permission check runs;
  the single fixed lane standing in for a real Lane Selector decision.
- **Belongs to a later build day:** Gate-A classification, real Gate-B enforcement,
  freshness/conflict detection in Gate-C, latency-budget enforcement in Gate-D, the Tool Registry,
  the MCP Gateway, multi-tenant User/API entry, and centralized/correlated observability.

## 6. One risk each

- **Latency:** `answer_question()` re-embeds all five chunks from scratch on every call — no
  caching across questions. Invisible at 5 chunks, but wouldn't scale to a real corpus without a
  real Gate-D latency budget.
- **Security:** with no Gate-B, nothing stops a sensitive chunk from being retrieved and quoted
  back verbatim. The lab is safe only because the documents are synthetic — that gap goes live the
  moment real documents replace them.
- **Hallucination:** the quantified-evidence guard only fires on the keywords `rate`/`percent`/
  `percentage`. It catches "what is the late payment interest rate?" but isn't a general
  fact-checker — a chunk stating the *wrong* number instead of no number would slip through
  untouched.
