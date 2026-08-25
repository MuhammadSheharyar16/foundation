# AICO Map — Tiny RAG Lab

This maps the Day 0B tiny RAG lab (`tiny_rag.py`, `wrapper/`, `data/chunks.json`) onto the
AICO architecture. The lab is a single-user, offline, single-lane exercise — it does not
reimplement AICO, it stands in for a small slice of it. For each component below I say
what plays that role here, and I'm explicit when nothing does and the component belongs
to a later build day. The reasoning is backed by one real traced run of
`python tiny_rag.py --question "What is the supplier delivery policy?"` (see the 15-step
trace at the end of §2) rather than guessed from the assignment brief.

## 1. Component table

| AICO component | What plays that role in this lab |
|---|---|
| **User and API entry** | The `--question` CLI argument (`_parse_args()` in `tiny_rag.py`) is the request. There is no user/tenant identity and no request ID minted at this boundary — `argparse` stands in for an API layer, nothing more. The wrapper mints its own `request_id` per call (e.g. `local-d044de35ec64`), but that's a Model Gateway concern, not this one. |
| **Gate-A (intent/domain classification)** | Not implemented. The script never asks "is this a RAG question, small talk, or something to block" — it treats every `--question` value as a RAG question unconditionally. |
| **Lane Selector** | Not implemented. There is exactly one lane. Which "lane" you exercise is decided by which Python file you run (`tiny_rag.py` vs `model_demo.py` vs `embedding_demo.py`), not by a runtime decision over one incoming request. |
| **Mode A** | Implemented as `_SYSTEM_INSTRUCTION` in `tiny_rag.py`. It's a fixed string that never changes between runs: only use the supplied evidence, keep each sentence's `[Cxxx]` tag attached, say `INSUFFICIENT_EVIDENCE` and cite nothing if the evidence doesn't answer the question. That's policy — it governs *how* the model is allowed to behave regardless of what question comes in. |
| **Gate-B (permission/tenant/PII/safe-disclosure)** | Not implemented. Nothing checks who is asking or whether a chunk is safe to disclose before retrieval runs — every question can see every one of the five chunks. This only looks safe today because the lead pre-screened the synthetic documents, not because a control exists in code. |
| **Mode B** | Implemented as `data/chunks.json`, loaded by `load_chunks()`. The five chunks (C001–C005) are the authoritative facts the answer is allowed to draw on — this is "what is true" for the lab. |
| **Gate-C (evidence source, provenance, freshness, completeness)** | Partially implemented. Each chunk carries `source_document` and `section` (provenance), and `answer_question()` runs two real checks before generating: a zero-similarity refusal (`top_score <= epsilon`) and a narrow quantified-evidence guard (`_requires_quantified_evidence` / `_has_quantified_evidence`) that catches a topically-relevant chunk that never states a number. There is no freshness check — `sample_questions.json`'s own planted note records that DOC-001 says 90 days' notice and DOC-002 says 60 days, and nothing here would detect or resolve that conflict. |
| **Model Gateway** | Implemented as `wrapper/` (`wrapper.chat()`, `wrapper.embed()`). Every demo file imports only those two functions, never `wrapper.model_client` internals or an SDK. It's currently backed by a deterministic, offline extractive summarizer and a hash-based embedder instead of a live Foundry endpoint (no credentials exist to swap in yet) — but the boundary is real: swapping the internals for a live SDK call later touches one file and nothing that calls `wrapper.chat`/`wrapper.embed` needs to change. |
| **Gate-D (final citation, quality, disclosure, latency-budget)** | Partially implemented. `parse_citations()` + `validate_citations()` is a real, tested control — a citation naming a chunk that wasn't actually retrieved is dropped, not passed through (`tests/test_tiny_rag.py` constructs exactly this case). Latency is measured (`latency_ms` on every result) but nothing is compared against a budget or fails a slow call. There's no disclosure or quality scoring beyond the citation check. |
| **Tool Registry** | Not implemented — later build day. There are no tools or function calls in this lab for a registry to govern. |
| **MCP Gateway** | Not implemented — later build day, and explicitly excluded by the assignment's working rules (no agent framework). No tool transport exists here to gate. |
| **Observability** | Partially implemented. Sanitized metadata (`request_id`, `model_alias`, `latency_ms`, token counts) is printed to console and written to `artifacts/model_run.txt` / `embedding_run.txt`; the RAG result (`status`, `citations`, `retrieved_chunk_ids`) is written to `artifacts/supported_answer.json` / `insufficient_evidence.json`. There is no central log store and no correlation across separate runs — each artifact is a standalone snapshot of one call. |

## 2. Sequence

```
user (--question)
  -> Gate-A         [not implemented: single hardcoded lane, no classification]
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

I traced this against a real run of `python tiny_rag.py --question "What is the supplier
delivery policy?"` step by step rather than assuming it from the diagram in the assignment
brief — the printed console output only shows the very start (the question) and the very
end (status/citations/answer); everything from Gate-A through Gate-D happens silently
between those two printed points, which is exactly why several of the labels above say
"not implemented" instead of "runs but isn't visible."

### Traced run, step by step

`tiny_rag.py`'s own `run()` only prints the question, then the final `status` /
`retrieved_chunk_ids` / `citations` / `answer` — everything in between happens silently.
To see it, the same functions `answer_question()` calls (`load_chunks`, `wrapper.embed`,
`rank_chunks`, `_requires_quantified_evidence`, `build_evidence_prompt`, `wrapper.chat`,
`parse_citations`, `validate_citations`) were called directly, in the same order, with a
print after each one — no logic was changed, only observed. Every value below is
copy-pasted from that real run, not reconstructed from reading the code.

| # | Step | What happened | AICO mapping |
|---|---|---|---|
| 1 | Question received | `question = "What is the supplier delivery policy?"` | User and API entry |
| 2 | Mode B loaded | `load_chunks()` reads `data/chunks.json` → 5 chunk IDs: `['C001', 'C002', 'C003', 'C004', 'C005']` | Mode B |
| 3 | Question embedded | `wrapper.embed([question])` → `request_id=local-1dc1b157bbd2`, `model_alias=embed-local-deterministic-v1`, `dimensions=512`, `latency_ms=0.199` | Model Gateway |
| 4 | Chunks embedded | `wrapper.embed(list(chunks.values()))` → `request_id=local-f7a36946f8b9`, same `model_alias`/`dimensions`, `latency_ms=2.176` | Model Gateway |
| 5 | All 5 chunks ranked | `rank_chunks()` cosine-ranks every chunk against the question vector: `C002=0.2545, C001=0.2395, C004=0.2282, C005=0.1485, C003=0.0845` | retrieve Mode B |
| 6 | Top 2 kept | `retrieved_chunk_ids = ['C002', 'C001']` (`_TOP_K = 2`) — C002 (late/notify) edges out C001 (base delivery window) because the question shares more content words with it | retrieve Mode B |
| 7 | Zero-similarity guard | `top_score = 0.2545 > 1e-9` → not refused. (Compare: the CEO-salary question scores `0.0` here and is refused at this exact step — see `artifacts/insufficient_evidence.json`) | Gate-C |
| 8 | Quantified-evidence guard | `_requires_quantified_evidence(question) = False` — the question contains none of `rate`/`percent`/`percentage`, so this guard doesn't apply (it exists for questions like "What is the late payment interest rate?") | Gate-C |
| 9 | Evidence prompt assembled | `build_evidence_prompt()` tags every sentence of C002 then C001 with its chunk ID → 11 tagged sentences, starting `"[C002] If a vendor expects to miss the five working day window, it must notify Meridian in writing at least 48 hours before the agreed delivery date..."` | retrieve Mode B → Model Gateway boundary |
| 10 | System prompt assembled | `_SYSTEM_INSTRUCTION` (the citation rule) + `"QUESTION: {question}"` concatenated → 390 characters, kept entirely separate from the evidence prompt | Mode A |
| 11 | Model called | `wrapper.chat(evidence_prompt, system=system)` → `request_id=local-bd7fb244f1d4`, `model_alias=chat-local-deterministic-v1`, `latency_ms=0.378`, `prompt_tokens=217`, `completion_tokens=93` | Model Gateway |
| 12 | Raw answer returned | Three `[Cxxx]`-tagged bullets, two from C002 and one from C001 (full text in `artifacts/supported_answer.json`) | Model Gateway output |
| 13 | Citations parsed | `parse_citations()` extracts `['C002', 'C001']` from the raw answer text | Gate-D |
| 14 | Citations validated | `validate_citations(['C002', 'C001'], ['C002', 'C001'])` → both survive, nothing dropped — every citation the model produced was actually retrieved | Gate-D |
| 15 | Final result | `status = "ANSWERED"`, `citations = ['C002', 'C001']` — matches `artifacts/supported_answer.json` exactly | answer → audit |

What this confirms:

- **Gate-A / Lane Selector / Gate-B genuinely run nothing** — there's no step between #1
  and #2 above where intent classification, lane choice, or a permission check could have
  happened. This is first-hand confirmation of what the component table above claims about
  those three rows, not an assumption from the sequence diagram.
- **Gate-C is real but narrow** — steps #7 and #8 are actual guard checks with real
  pass/fail outcomes, not decoration. Re-running this same trace for `"What is the CEO
  salary?"` refuses at step #7 (`top_score = 0.0`); re-running it for `"What is the late
  payment interest rate?"` passes step #7 but refuses at step #8. Both are covered by
  `tests/test_tiny_rag.py`.
- **Gate-D is a real, tested control, not a formality** — step #14 is a no-op here
  because the model happened to behave, but `tests/test_tiny_rag.py::
  test_fabricated_citation_is_rejected` exercises the case where it isn't a no-op (a
  citation naming a chunk outside `retrieved_chunk_ids` gets dropped).
- **Observability is a snapshot, not a log** — steps #3, #4, and #11 each mint their own
  `request_id`; nothing here correlates them into one trace ID for the overall question,
  which is exactly the Observability gap the component table above already calls out.

## 3. Why Mode A tells the system how to operate while Mode B provides what is true

Mode A (`_SYSTEM_INSTRUCTION`) is the same string on every single run of `tiny_rag.py`,
no matter what question is asked or which chunks get retrieved. It's policy: cite tags,
refuse when the evidence doesn't support an answer, never invent a chunk ID. Mode B
(`chunks.json`) is the opposite — it's the part that changes meaning depending on what's
actually retrieved for *this* question. Ask about delivery and Mode B contributes C001/C002;
ask about renewal notice and it contributes C003. If I swapped Mode A's wording for a
differently-phrased citation rule, the facts in Mode B wouldn't change at all. If I swapped
which chunks are in Mode B, the instruction in Mode A would still apply exactly as written
to whatever new evidence showed up. They vary independently because they answer two
different questions — Mode A answers "how am I allowed to behave," Mode B answers "what am
I allowed to claim."

## 4. Why the model cannot directly execute a tool or query Mode B

`wrapper.chat()` never receives a database handle, a file path, or a callable — it
receives one already-assembled prompt string (`build_evidence_prompt()`'s output) and a
system string, and returns text. Every step that touches Mode B — loading `chunks.json`,
embedding, ranking, picking the top two, and afterward validating citations against
`retrieved_chunk_ids` — happens in plain Python before the model is called and after it
returns, entirely outside anything the model can influence. If the model could query Mode B
or invoke a tool directly instead, that separation disappears: there would be no fixed point
between "what the model asked for" and "what it's allowed to have" where Gate-C or Gate-D
could actually run. The gates in this design only work because they sit on a path the model
does not choose and cannot skip — the moment the model can reach evidence or a tool on its
own, a fabricated citation or an out-of-scope query has no checkpoint left to be caught at.

## 5. What's implemented, what's simulated, what's later

- **Implemented and enforced by a test:** Mode A instruction; Mode B evidence file; the
  Model Gateway boundary (`wrapper.chat`/`wrapper.embed` only); citation validation
  (`validate_citations`, with a fabricated-citation test); the zero-similarity and
  quantified-evidence refusal guards; sanitized observability metadata.
- **Simulated / true only because the lab is small:** Gate-B's "nothing sensitive gets
  disclosed" outcome, achieved because the lead pre-screened the synthetic documents, not
  because a permission check runs; the single fixed lane standing in for a Lane Selector
  decision.
- **Belongs to a later build day:** Gate-A intent classification, real Gate-B enforcement,
  freshness/conflict detection in Gate-C, latency-budget enforcement in Gate-D, the Tool
  Registry, the MCP Gateway, multi-tenant User/API entry, and centralized/correlated
  observability across runs.

## 6. One risk each

- **Latency:** `answer_question()` re-embeds all five chunks from scratch on every single
  call — there's no caching of chunk vectors across questions. That's invisible at 5
  chunks but would not scale to a real corpus without a real Gate-D latency budget backing
  it up.
- **Security:** with no Gate-B, nothing in code stops a sensitive chunk from being
  retrieved and quoted back verbatim in an answer. The lab is safe only because the
  documents were pre-screened as synthetic — the moment real documents replaced them,
  this gap would be live.
- **Hallucination:** the quantified-evidence guard only fires on a specific keyword trigger
  (`rate`/`percent`/`percentage`). It correctly catches "what is the late payment interest
  rate?" (C004 discusses late payment but never states a rate), but it is not a general
  fact-checker — a differently-shaped gap (e.g. a chunk that states a wrong number instead
  of no number) would not be caught by anything in this lab.
