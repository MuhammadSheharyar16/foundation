"""Embedding similarity lab.

Loads S1/S2/S3 from data/sample_questions.json (the fixture, not hand-copied
text), embeds them through `wrapper.embed`, ranks all three pairs by cosine
similarity, and demonstrates that S1-S2 (same meaning, different words)
ranks above S1-S3 (an unrelated sentence) even though S1 and S2 share almost
no literal words.

Only ever imports from `wrapper` — never `wrapper.model_client` directly and
never an SDK. Console output and the artifact never include the vectors
themselves, per the assignment's "record model alias, dimensions and
execution time without printing the vector itself" rule — only sanitized
metadata and the ranked similarity table.
"""

import json
import math
from pathlib import Path

from wrapper import ConfigurationMissingError, ModelCallError, ModelTimeoutError, embed

_REPO_ROOT = Path(__file__).resolve().parent
_SAMPLE_QUESTIONS_PATH = _REPO_ROOT / "data" / "sample_questions.json"
_ARTIFACT_PATH = _REPO_ROOT / "artifacts" / "embedding_run.txt"
_SENTENCE_IDS = ["S1", "S2", "S3"]

# Adapted from concepts.md's "Semantic Retrieval" entry, applied to this pair.
_WHY_EXPLANATION = (
    "S1 and S2 describe the same real-world event -- a vehicle/car engine/motor "
    "problem -- using different words ('vehicle'/'car', 'engine'/'motor', "
    "'supplier'/'vendor', 'problem'/'issue'), while S3 is about an unrelated "
    "topic (a contract renewal date). Semantic retrieval ranks by meaning, not "
    "literal wording, so S1-S2 must score higher than S1-S3 even though S1 and "
    "S2 share almost no words in common word-for-word. Lexical (keyword-only) "
    "retrieval would miss this relationship entirely."
)


def _load_sentences() -> dict[str, str]:
    """Reads the embedding_sentences fixture -- S1/S2/S3 are never hand-copied."""
    payload = json.loads(_SAMPLE_QUESTIONS_PATH.read_text(encoding="utf-8"))
    return {item["id"]: item["text"] for item in payload["embedding_sentences"]}


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _ranked_pairs(vectors: dict[str, list[float]]) -> list[tuple[str, str, float]]:
    """All unique pairs among _SENTENCE_IDS, ranked highest similarity first."""
    pairs = [
        (a, b, _cosine_similarity(vectors[a], vectors[b]))
        for i, a in enumerate(_SENTENCE_IDS)
        for b in _SENTENCE_IDS[i + 1 :]
    ]
    return sorted(pairs, key=lambda item: -item[2])


def _format_table(ranked: list[tuple[str, str, float]]) -> str:
    header = f"{'pair':<8}{'similarity':>10}"
    rows = [header, "-" * len(header)]
    rows += [f"{a + '-' + b:<8}{score:>10.4f}" for a, b, score in ranked]
    return "\n".join(rows)


def _write_artifact(*, metadata: dict, ranked: list[tuple[str, str, float]]) -> None:
    _ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"model_alias: {metadata['model_alias']}",
        f"dimensions: {metadata['dimensions']}",
        f"latency_ms: {metadata['latency_ms']}",
        "",
        "ranked_similarity:",
        _format_table(ranked),
        "",
        "why_s1_s2_ranks_above_s1_s3:",
        _WHY_EXPLANATION,
    ]
    _ARTIFACT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> int:
    sentences = _load_sentences()
    texts = [sentences[sentence_id] for sentence_id in _SENTENCE_IDS]

    try:
        result = embed(texts)
    except ConfigurationMissingError as exc:
        print(f"Embedding call skipped - configuration missing: {exc}")
        return 1
    except ModelTimeoutError as exc:
        print(f"Embedding call skipped - timed out: {exc}")
        return 1
    except ModelCallError as exc:
        print(f"Embedding call failed: {exc}")
        return 1

    metadata = result.sanitized_metadata()
    print("Embedding call succeeded. Sanitized metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    vectors = dict(zip(_SENTENCE_IDS, result.vectors))
    ranked = _ranked_pairs(vectors)
    print()
    print("Ranked similarity:")
    print(_format_table(ranked))

    sim_s1_s2 = _cosine_similarity(vectors["S1"], vectors["S2"])
    sim_s1_s3 = _cosine_similarity(vectors["S1"], vectors["S3"])
    print()
    if sim_s1_s2 > sim_s1_s3:
        print(f"Confirmed: sim(S1,S2)={sim_s1_s2:.4f} > sim(S1,S3)={sim_s1_s3:.4f}")
    else:
        print(f"UNEXPECTED: sim(S1,S2)={sim_s1_s2:.4f} did not rank above sim(S1,S3)={sim_s1_s3:.4f}")
        return 1

    print()
    print("Why S1-S2 ranks above S1-S3:")
    print(_WHY_EXPLANATION)

    _write_artifact(metadata=metadata, ranked=ranked)
    print()
    print(f"Wrote sanitized run to {_ARTIFACT_PATH.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
