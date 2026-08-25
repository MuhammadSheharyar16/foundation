"""First approved model call.

Reads one synthetic supplier policy document, asks for a three-bullet
summary through `wrapper.chat`, and demonstrates the safe-print/safe-error
discipline required before Day 1.

Only ever imports from `wrapper` — never `wrapper.model_client` directly and
never an SDK. So that's why `wrapper.chat` is currently backed
by a deterministic local "model" rather than a live endpoint.

Console output is sanitized metadata ONLY (request ID, model alias, latency,
token counts) — never the document text and never a credential (there isn't
one to leak: this repo has no key anywhere). The three-bullet summary itself
is the demo's intended output, so it's written — together with the metadata
that produced it — to artifacts/model_run.txt.
"""

from pathlib import Path

from wrapper import ConfigurationMissingError, ModelCallError, ModelTimeoutError, chat

_REPO_ROOT = Path(__file__).resolve().parent
_DOCUMENT_PATH = _REPO_ROOT / "data" / "documents" / "DOC-002-contract-delivery-renewal-termination.md"
_ARTIFACT_PATH = _REPO_ROOT / "artifacts" / "model_run.txt"
_SYSTEM_INSTRUCTION = "Summarize in exactly three bullets."


def _extract_prose(markdown_text: str) -> str:
    """Strip markdown headers, metadata lines and horizontal rules.

    Keeps only the flowing prose paragraphs, so the extractive summarizer in
    `wrapper.model_client` scores real sentences instead of tripping over
    markdown syntax (e.g. a heading like "## 2. Delivery obligations" would
    otherwise look like a sentence ending after "2.").
    """
    prose_lines = []
    for line in markdown_text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            continue
        if stripped.startswith("**") and ":**" in stripped:
            continue  # metadata lines, e.g. "**Version:** 1.4"
        if set(stripped) == {"-"}:
            continue  # horizontal rule
        prose_lines.append(stripped)
    return " ".join(prose_lines)


def _write_artifact(*, document_name: str, metadata: dict, summary: str) -> None:
    _ARTIFACT_PATH.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        f"source_document: {document_name}",
        f"system_instruction: {_SYSTEM_INSTRUCTION}",
        *(f"{key}: {value}" for key, value in metadata.items()),
        "",
        "summary:",
        summary,
    ]
    _ARTIFACT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run() -> int:
    try:
        document_text = _DOCUMENT_PATH.read_text(encoding="utf-8")
    except OSError as exc:
        print(f"Could not read document {_DOCUMENT_PATH.name}: {exc}")
        return 1

    prompt = _extract_prose(document_text)

    try:
        result = chat(prompt, system=_SYSTEM_INSTRUCTION)
    except ConfigurationMissingError as exc:
        print(f"Model call skipped - configuration missing: {exc}")
        return 1
    except ModelTimeoutError as exc:
        print(f"Model call skipped - timed out: {exc}")
        return 1
    except ModelCallError as exc:
        print(f"Model call failed: {exc}")
        return 1

    metadata = result.sanitized_metadata()
    print("Model call succeeded. Sanitized metadata:")
    for key, value in metadata.items():
        print(f"  {key}: {value}")

    _write_artifact(document_name=_DOCUMENT_PATH.name, metadata=metadata, summary=result.text)
    print(f"Wrote sanitized run (metadata + summary) to {_ARTIFACT_PATH.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(run())
