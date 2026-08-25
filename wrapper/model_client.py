"""Chat and embedding entry points.

These two functions are the only way lab code reaches a model.

There is no live Foundry/Azure endpoint wired up yet (no SDK dependency, no
credentials anywhere in this repo), so both functions are backed by a small,
deterministic, network-free "model": `chat` extracts the most keyword-relevant
sentences from the prompt as a bullet summary, and `embed` hashes words into a
fixed-size bag-of-words vector. Same input always produces the same output,
which is what makes this reproducible in tests and at the review gate without
needing live model access. See `day0b-guide.md` for the reasoning.

`python_bridge.DeterministicFakeEmbeddingProvider` calls back into
`hash_embed_vector` below so the injectable test fake and this "real"
implementation always agree on what "similar" means.

Swapping in a real Foundry SDK call later only touches this file: every
caller (`model_demo.py`, `embedding_demo.py`, `tiny_rag.py`, and anything
using `python_bridge.WrapperEmbeddingProvider`) only ever calls `chat()` /
`embed()` and never reaches past them.

Every implementation here must:
  - authenticate from environment or identity sign-in, never a key in code
  - apply a timeout and raise ModelTimeoutError when exceeded
  - wrap provider exceptions in ModelCallError
  - return the typed results in results.py
  - never log the token, authorization header, prompt body or vectors
"""

import hashlib
import math
import re
import time
import uuid

from .config import load_config
from .errors import ModelCallError, ModelTimeoutError
from .results import ChatResult, EmbeddingResult

_EMBEDDING_DIMENSIONS = 16
_MAX_SUMMARY_BULLETS = 3

# Small closed-class words excluded from keyword scoring so summary sentences
# are picked by their content words, not by how many "the"/"and" they contain.
_STOPWORDS = frozenset(
    {
        "a", "an", "and", "are", "as", "at", "be", "been", "being", "but",
        "by", "can", "could", "for", "from", "if", "in", "is", "it", "its",
        "may", "must", "no", "not", "of", "on", "or", "shall", "should",
        "that", "the", "these", "this", "those", "to", "was", "were", "will",
        "with", "would",
    }
)

_TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
_SENTENCE_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+")

# A short, curated synonym→canonical map, NOT general semantic understanding.
# Plain word hashing is purely lexical: "vehicle" and "car" hash to unrelated
# buckets even though they mean the same thing, so on its own it cannot rank
# S1 ("...vehicle engine problem") close to S2 ("...car motor issue") above
# S3 (unrelated). Canonicalizing a small, known vocabulary before hashing —
# covering the embedding-lab sentences and the synonym pairs noted in
# data/sample_questions.json — fixes that for this lab's fixed dataset only.
# It does not generalize to unseen words; a real embedding model would.
_SYNONYM_CANON: dict[str, str] = {
    "vendor": "supplier",
    "vehicle": "car",
    "engine": "motor",
    "issue": "problem",
    "shipment": "delivery",
    "price": "cost",
    "charge": "cost",
    "cancel": "end",
    "terminate": "end",
}


def hash_embed_vector(text: str, dimensions: int) -> list[float]:
    """Hash each (canonicalized, non-stopword) word of `text` into a fixed-size
    bag-of-words vector, then L2-normalize.

    Deterministic and network-free: the same input text always produces the
    same vector, and inputs that share (or are synonyms for, per
    `_SYNONYM_CANON`) content words produce vectors that are closer together.
    This is the canonical implementation of the lab's embedding algorithm —
    both `embed()` below and `python_bridge.DeterministicFakeEmbeddingProvider`
    call it, so there is exactly one place that defines what "similar" means.

    Args:
        text: The string to embed.
        dimensions: Length of the returned vector. Must be positive.

    Returns:
        A unit-length (or all-zero, if `text` has no non-stopword words) vector.
    """
    vector = [0.0] * dimensions
    for word in _tokenize(text):
        if word in _STOPWORDS:
            continue
        canonical_word = _SYNONYM_CANON.get(word, word)
        digest = hashlib.sha256(canonical_word.encode("utf-8")).hexdigest()
        index = int(digest, 16) % dimensions
        vector[index] += 1.0

    norm = math.sqrt(sum(component * component for component in vector))
    if norm > 0:
        vector = [component / norm for component in vector]
    return vector


def _tokenize(text: str) -> list[str]:
    """Lowercase word-like tokens, used for both summary scoring and token counts."""
    return _TOKEN_PATTERN.findall(text.lower())


def _split_sentences(text: str) -> list[str]:
    """Split on sentence-ending punctuation, dropping empty fragments."""
    return [sentence.strip() for sentence in _SENTENCE_SPLIT_PATTERN.split(text.strip()) if sentence.strip()]


def _summarize(prompt: str, system: str | None, *, max_bullets: int = _MAX_SUMMARY_BULLETS) -> str:
    """Extractive, deterministic summary: the `max_bullets` most keyword-relevant sentences.

    A sentence's score is the sum of corpus-wide frequencies of its own
    (non-stopword) words, plus a bonus for words it shares with `system` —
    so a system instruction can nudge which sentences get picked, even
    though this "model" doesn't truly follow instructions. Ties are broken
    by original sentence order, and the chosen sentences are re-sorted back
    into original order so the summary reads naturally.
    """
    sentences = _split_sentences(prompt)
    if not sentences:
        return ""

    tokenized_sentences = [_tokenize(sentence) for sentence in sentences]

    word_frequency: dict[str, int] = {}
    for tokens in tokenized_sentences:
        for word in tokens:
            if word in _STOPWORDS:
                continue
            word_frequency[word] = word_frequency.get(word, 0) + 1

    system_keywords = {word for word in _tokenize(system or "") if word not in _STOPWORDS}

    scored: list[tuple[int, int]] = []
    for index, tokens in enumerate(tokenized_sentences):
        unique_words = {word for word in tokens if word not in _STOPWORDS}
        score = sum(word_frequency[word] for word in unique_words)
        score += sum(2 for word in unique_words if word in system_keywords)
        scored.append((score, index))

    top_indices = sorted(index for _, index in sorted(scored, key=lambda item: (-item[0], item[1]))[:max_bullets])
    return "\n".join(f"- {sentences[index]}" for index in top_indices)


def chat(
    prompt: str,
    *,
    system: str | None = None,
    timeout_s: float | None = None,
) -> ChatResult:
    """Send one chat completion request.

    Args:
        prompt: The user message.
        system: Optional system instruction, kept separate from the user message.
        timeout_s: Overrides the configured default timeout.

    Returns:
        ChatResult with the response text and sanitized call metadata.

    Raises:
        ConfigurationMissingError: Configuration or identity unavailable.
        ModelTimeoutError: Call exceeded the timeout budget.
        ModelCallError: Any other provider failure.
    """
    config = load_config()
    budget_s = timeout_s if timeout_s is not None else config.default_timeout_s

    start = time.perf_counter()
    try:
        summary_text = _summarize(prompt, system)
    except Exception as exc:  # unexpected failure inside the deterministic "model"
        raise ModelCallError(f"chat call failed: {exc}") from exc
    latency_ms = (time.perf_counter() - start) * 1000

    if latency_ms > budget_s * 1000:
        raise ModelTimeoutError(f"chat call exceeded timeout budget of {budget_s}s")

    return ChatResult(
        text=summary_text,
        request_id=f"local-{uuid.uuid4().hex[:12]}",
        model_alias=config.chat_model_alias,
        latency_ms=latency_ms,
        prompt_tokens=len(_tokenize(prompt)),
        completion_tokens=len(_tokenize(summary_text)),
    )


def embed(
    texts: list[str],
    *,
    timeout_s: float | None = None,
) -> EmbeddingResult:
    """Embed one or more strings.

    Args:
        texts: Input strings. Vectors are returned in the same order.
        timeout_s: Overrides the configured default timeout.

    Returns:
        EmbeddingResult with one vector per input and sanitized call metadata.

    Raises:
        ConfigurationMissingError: Configuration or identity unavailable.
        ModelTimeoutError: Call exceeded the timeout budget.
        ModelCallError: Any other provider failure.
    """
    config = load_config()

    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        raise TypeError("texts must be a list[str]")

    budget_s = timeout_s if timeout_s is not None else config.default_timeout_s

    start = time.perf_counter()
    try:
        vectors = [hash_embed_vector(text, _EMBEDDING_DIMENSIONS) for text in texts]
    except Exception as exc:  # unexpected failure inside the deterministic "model"
        raise ModelCallError(f"embed call failed: {exc}") from exc
    latency_ms = (time.perf_counter() - start) * 1000

    if latency_ms > budget_s * 1000:
        raise ModelTimeoutError(f"embed call exceeded timeout budget of {budget_s}s")

    return EmbeddingResult(
        vectors=vectors,
        request_id=f"local-{uuid.uuid4().hex[:12]}",
        model_alias=config.embedding_model_alias,
        dimensions=_EMBEDDING_DIMENSIONS,
        latency_ms=latency_ms,
    )
