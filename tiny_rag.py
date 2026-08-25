"""Tiny manual RAG: retrieve, ground, answer, cite -- or refuse.

Retrieval: embed the question and all five chunks in data/chunks.json,
rank by cosine similarity, keep the top two as `retrieved_chunk_ids`.

Generation: build an EVIDENCE prompt where every sentence of the top two
chunks is tagged with its own chunk ID (e.g. "[C001] Suppliers must ...");
call `wrapper.chat` with that as `prompt` and a SYSTEM instruction (the
citation rule) plus the QUESTION folded into `system`. Only ever imports
from `wrapper` -- never `wrapper.model_client` directly and never an SDK.

Why the question rides in `system`, not `prompt`: `chat()`'s deterministic
"model" (see wrapper/model_client.py) extracts sentences out of whatever it
is given as `prompt` and only uses `system` as a keyword-relevance nudge, so
the question would risk being extracted back out verbatim as if it were the
"answer" if it sat in `prompt` alongside the evidence. Keeping it in `system`
lets it bias which evidence sentences get picked without ever being
extractable itself -- so every extracted sentence always carries its own
`[Cxxx]` tag, which is what makes citation parsing exact rather than
best-effort.

Grounding decision, in order:
  1. Zero retrieval signal (e.g. "What is the CEO salary?" -- nothing in the
     corpus is about that) -> INSUFFICIENT_EVIDENCE, no model call needed.
  2. A narrow guard for rate/percentage questions (see
     `_requires_quantified_evidence`): retrieval can find a topically related
     chunk that never actually states a number (the payment-terms chunk
     discusses "late payment" and "interest" but never quantifies a rate) --
     if the top-evidence chunk has no percentage in it, refuse rather than
     let the extractive "model" present some on-topic-but-unquantified
     sentence as though it answered the question.
  3. Otherwise: generate, parse `[Cxxx]` citations out of the answer, and
     validate every citation against `retrieved_chunk_ids` -- a citation that
     is not one of the two retrieved chunks is dropped, never passed through.
"""

import argparse
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from wrapper import ConfigurationMissingError, ModelCallError, ModelTimeoutError, chat, embed
from wrapper.model_client import _split_sentences as _wrapper_split_sentences

_REPO_ROOT = Path(__file__).resolve().parent
_CHUNKS_PATH = _REPO_ROOT / "data" / "chunks.json"
_ARTIFACTS_DIR = _REPO_ROOT / "artifacts"
_TOP_K = 2
_ZERO_SIMILARITY_EPSILON = 1e-9

_SYSTEM_INSTRUCTION = (
    "You are a procurement policy assistant. Only use the evidence below. "
    "Every sentence you rely on is already tagged with its chunk ID, like "
    "[C001] -- keep that tag attached to any sentence you use, and never "
    "invent a chunk ID that isn't in the evidence. If the evidence does not "
    "answer the question, say INSUFFICIENT_EVIDENCE and cite nothing."
)

# Narrow guard, not general numeric fact-checking -- see the module
# docstring above. Only fires for questions that specifically ask for a
# rate/percentage.
_QUANTIFIED_ANSWER_TRIGGER_WORDS = frozenset({"rate", "percent", "percentage"})
_WORD_PATTERN = re.compile(r"[a-z]+")
_CITATION_PATTERN = re.compile(r"\[([A-Za-z0-9]+)\]")

_INSUFFICIENT_EVIDENCE_ANSWER = "INSUFFICIENT_EVIDENCE"


@dataclass(frozen=True)
class RagResult:
    """Typed result of one tiny_rag question."""

    question: str
    status: str  # "ANSWERED" or "INSUFFICIENT_EVIDENCE"
    answer: str
    citations: list[str]
    retrieved_chunk_ids: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def load_chunks(path: Path = _CHUNKS_PATH) -> dict[str, str]:
    """Loads data/chunks.json into {chunk_id: text}, in file order."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {chunk["chunk_id"]: chunk["text"] for chunk in payload["chunks"]}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def rank_chunks(
    question_vector: list[float], chunk_vectors: dict[str, list[float]]
) -> list[tuple[str, float]]:
    """All chunk IDs ranked by cosine similarity to `question_vector`, highest first."""
    scored = [(cid, _cosine_similarity(question_vector, vec)) for cid, vec in chunk_vectors.items()]
    return sorted(scored, key=lambda item: -item[1])


def _requires_quantified_evidence(question: str) -> bool:
    words = set(_WORD_PATTERN.findall(question.lower()))
    return bool(words & _QUANTIFIED_ANSWER_TRIGGER_WORDS)


def _has_quantified_evidence(text: str) -> bool:
    return "%" in text or "percent" in text.lower()


def _tag_chunk_sentences(chunk_id: str, chunk_text: str) -> list[str]:
    """Splits `chunk_text` the same way `wrapper.model_client` will re-split the
    assembled evidence prompt, tagging each resulting sentence with its chunk
    ID so the tag survives extraction and always lands on the right sentence.
    """
    return [f"[{chunk_id}] {sentence}" for sentence in _wrapper_split_sentences(chunk_text)]


def build_evidence_prompt(retrieved: list[tuple[str, str]]) -> str:
    """`retrieved` is [(chunk_id, chunk_text), ...] in ranked order."""
    tagged_sentences: list[str] = []
    for chunk_id, chunk_text in retrieved:
        tagged_sentences.extend(_tag_chunk_sentences(chunk_id, chunk_text))
    return " ".join(tagged_sentences)


def parse_citations(answer_text: str) -> list[str]:
    """Extracts `[Cxxx]`-style tags from `answer_text`, deduplicated, in order
    of first appearance."""
    seen: dict[str, None] = {}
    for match in _CITATION_PATTERN.finditer(answer_text):
        seen.setdefault(match.group(1), None)
    return list(seen)


def validate_citations(citations: list[str], retrieved_chunk_ids: list[str]) -> list[str]:
    """Drops any citation that is not one of `retrieved_chunk_ids`, rather than
    passing a fabricated citation through. Order is preserved."""
    retrieved_set = set(retrieved_chunk_ids)
    return [citation for citation in citations if citation in retrieved_set]


def _refuse(question: str, retrieved_chunk_ids: list[str]) -> RagResult:
    return RagResult(
        question=question,
        status="INSUFFICIENT_EVIDENCE",
        answer=_INSUFFICIENT_EVIDENCE_ANSWER,
        citations=[],
        retrieved_chunk_ids=retrieved_chunk_ids,
    )


def answer_question(question: str, chunks: dict[str, str] | None = None) -> RagResult:
    """Runs the full retrieve -> ground -> generate -> validate pipeline for one question.

    Raises:
        ConfigurationMissingError, ModelTimeoutError, ModelCallError: propagated
            from `wrapper.embed`/`wrapper.chat` untouched, so callers can apply
            the same safe-error handling used by the other demos.
    """
    chunks = load_chunks() if chunks is None else chunks
    chunk_ids = list(chunks)

    question_vector = embed([question]).vectors[0]
    chunk_vectors = dict(zip(chunk_ids, embed(list(chunks.values())).vectors))

    ranked = rank_chunks(question_vector, chunk_vectors)
    retrieved = ranked[:_TOP_K]
    retrieved_chunk_ids = [chunk_id for chunk_id, _ in retrieved]
    top_chunk_id, top_score = ranked[0]

    if top_score <= _ZERO_SIMILARITY_EPSILON:
        return _refuse(question, retrieved_chunk_ids)

    if _requires_quantified_evidence(question) and not _has_quantified_evidence(chunks[top_chunk_id]):
        return _refuse(question, retrieved_chunk_ids)

    evidence_prompt = build_evidence_prompt([(cid, chunks[cid]) for cid in retrieved_chunk_ids])
    system = f"{_SYSTEM_INSTRUCTION}\n\nQUESTION: {question}"
    result = chat(evidence_prompt, system=system)

    citations = validate_citations(parse_citations(result.text), retrieved_chunk_ids)
    if not citations:
        return _refuse(question, retrieved_chunk_ids)

    return RagResult(
        question=question,
        status="ANSWERED",
        answer=result.text,
        citations=citations,
        retrieved_chunk_ids=retrieved_chunk_ids,
    )


def _artifact_path_for(status: str) -> Path:
    filename = "supported_answer.json" if status == "ANSWERED" else "insufficient_evidence.json"
    return _ARTIFACTS_DIR / filename


def _write_artifact(result: RagResult) -> Path:
    _ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    path = _artifact_path_for(result.status)
    path.write_text(json.dumps(result.to_dict(), indent=2) + "\n", encoding="utf-8")
    return path


def run(question: str) -> int:
    try:
        result = answer_question(question)
    except ConfigurationMissingError as exc:
        print(f"RAG call skipped - configuration missing: {exc}")
        return 1
    except ModelTimeoutError as exc:
        print(f"RAG call skipped - timed out: {exc}")
        return 1
    except ModelCallError as exc:
        print(f"RAG call failed: {exc}")
        return 1

    print(f"question: {result.question}")
    print(f"status: {result.status}")
    print(f"retrieved_chunk_ids: {result.retrieved_chunk_ids}")
    print(f"citations: {result.citations}")
    print("answer:")
    print(result.answer)

    written_path = _write_artifact(result)
    print()
    print(f"Wrote result to {written_path.relative_to(_REPO_ROOT)}")
    return 0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tiny cited RAG over data/chunks.json.")
    parser.add_argument("--question", required=True, help="The question to answer.")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(_parse_args().question))
