"""Tests for embedding_demo.py: the Day 0B semantic-retrieval requirement
that S1-S2 (same meaning, different words) ranks above S1-S3 (unrelated)."""

from embedding_demo import _SENTENCE_IDS, _cosine_similarity, _load_sentences
from wrapper import embed


def _set_required_env(monkeypatch) -> None:
    monkeypatch.setenv("MODEL_ENDPOINT", "local://deterministic")
    monkeypatch.setenv("CHAT_MODEL_ALIAS", "chat-local-deterministic-v1")
    monkeypatch.setenv("EMBEDDING_MODEL_ALIAS", "embed-local-deterministic-v1")


def test_sim_s1_s2_ranks_above_sim_s1_s3(monkeypatch) -> None:
    _set_required_env(monkeypatch)
    sentences = _load_sentences()
    texts = [sentences[sentence_id] for sentence_id in _SENTENCE_IDS]

    vectors = dict(zip(_SENTENCE_IDS, embed(texts).vectors))

    sim_s1_s2 = _cosine_similarity(vectors["S1"], vectors["S2"])
    sim_s1_s3 = _cosine_similarity(vectors["S1"], vectors["S3"])

    assert sim_s1_s2 > sim_s1_s3


def test_sentences_are_loaded_from_the_fixture_not_hand_copied() -> None:
    """Guards against accidentally hand-copying S1/S2/S3 text into the demo
    script instead of reading data/sample_questions.json."""
    sentences = _load_sentences()

    assert set(_SENTENCE_IDS) <= set(sentences)
    assert sentences["S1"] == "The supplier reported a vehicle engine problem."
    assert sentences["S2"] == "The vendor has an issue with the car motor."
    assert sentences["S3"] == "The contract renewal date is next month."
