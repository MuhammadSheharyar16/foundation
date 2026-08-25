# AICO Map — Tiny RAG Lab

This maps the Day 0B tiny RAG lab (`tiny_rag.py`, `wrapper/`, `data/chunks.json`) onto the
AICO architecture. The lab is a single-user, offline, single-lane exercise — it does not
reimplement AICO, it stands in for a small slice of it. For each component below I say
what plays that role here, and I'm explicit when nothing does and the component belongs
to a later build day. The reasoning is backed by one real traced run of
`python tiny_rag.py --question "What is the supplier delivery policy?"` (see
`final-submission-guide.md` §5 Step 2 for the full 15-step trace) rather than guessed from
the assignment brief.

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
delivery policy?"` step by step rather than assuming it from the diagram in the
assignment brief — the printed console output only shows the very start (the question)
and the very end (status/citations/answer); everything from Gate-A through Gate-D happens
silently between those two printed points, which is exactly why several of the labels
above say "not implemented" instead of "runs but isn't visible."

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
