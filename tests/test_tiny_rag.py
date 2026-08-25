"""Tests for tiny_rag.py: citation grounding and the insufficient-evidence
refusal path.

Covers the three Day 0B mandatory checks: a supported answer cites only
retrieved chunks, a fabricated citation is rejected, and the absent question
returns INSUFFICIENT_EVIDENCE with no citations -- plus a data-integrity
sanity check on the chunk fixture itself.
"""

from tiny_rag import answer_question, load_chunks, validate_citations


def _set_required_env(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_ENDPOINT", "local://deterministic")
    monkeypatch.setenv("CHAT_MODEL_ALIAS", "chat-local-deterministic-v1")
    monkeypatch.setenv("EMBEDDING_MODEL_ALIAS", "embed-local-deterministic-v1")


def test_supported_answer_cites_only_retrieved_chunk_ids(monkeypatch) -> None:
    _set_required_env(monkeypatch)

    result = answer_question("What is the supplier delivery policy?")

    assert result.status == "ANSWERED"
    assert result.citations  # at least one grounded citation
    assert set(result.citations) <= set(result.retrieved_chunk_ids)
    assert set(result.retrieved_chunk_ids) == {"C001", "C002"}


def test_absent_question_returns_insufficient_evidence_with_no_citations(monkeypatch) -> None:
    _set_required_env(monkeypatch)

    result = answer_question("What is the CEO salary?")

    assert result.status == "INSUFFICIENT_EVIDENCE"
    assert result.citations == []


def test_fabricated_citation_is_rejected() -> None:
    """A citation naming a chunk that was never retrieved -- whether it's a
    real chunk ID from elsewhere in the corpus or one that doesn't exist at
    all -- must never survive validation."""
    retrieved_chunk_ids = ["C001", "C002"]

    # A real chunk ID, just not one of the two retrieved for this question.
    assert validate_citations(["C001", "C003"], retrieved_chunk_ids) == ["C001"]

    # A chunk ID that doesn't exist anywhere in the corpus.
    assert validate_citations(["C999"], retrieved_chunk_ids) == []

    # Every citation fabricated -> drops to no citations at all, not a partial pass.
    assert validate_citations(["C999", "C888"], retrieved_chunk_ids) == []


def test_load_chunks_returns_all_five_known_ids() -> None:
    chunks = load_chunks()

    assert set(chunks) == {"C001", "C002", "C003", "C004", "C005"}
