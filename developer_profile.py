"""Typed developer profile model and JSON loading.

`DeveloperProfile` is the typed shape other lab code should depend on instead
of raw dicts. Profiles are loaded from JSON files under `data/` and validated
eagerly in `__post_init__`, so a malformed profile fails at load time with a
clear message instead of surfacing as a confusing error further downstream.
"""

import json
from dataclasses import dataclass
from pathlib import Path


class ProfileValidationError(ValueError):
    """Raised when a profile fails validation.

    Message names the invalid field(s), never the whole raw payload.
    """


@dataclass(frozen=True)
class DeveloperProfile:
    """A developer's current standing and the AI role they are targeting.

    Attributes:
        name: Full name. Must be non-empty after stripping whitespace.
        current_role: Current job title. Must be non-empty after stripping.
        experience_years: Years of professional experience. Must be a
            non-negative number; fractional years (e.g. 6.5) are allowed.
        target_ai_role: The AI role the developer is aiming for. Must be
            non-empty after stripping.
    """

    name: str
    current_role: str
    experience_years: float
    target_ai_role: str

    def __post_init__(self) -> None:
        _require_non_empty_str("name", self.name)
        _require_non_empty_str("current_role", self.current_role)
        _require_non_empty_str("target_ai_role", self.target_ai_role)

        if isinstance(self.experience_years, bool) or not isinstance(
            self.experience_years, (int, float)
        ):
            raise ProfileValidationError("experience_years must be a number")
        if self.experience_years < 0:
            raise ProfileValidationError("experience_years must not be negative")


def _require_non_empty_str(field_name: str, value: object) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ProfileValidationError(f"{field_name} must be a non-empty string")


_REQUIRED_FIELDS = ("name", "current_role", "experience_years", "target_ai_role")


def load_profile(path: Path) -> DeveloperProfile:
    """Load and validate one `DeveloperProfile` from a JSON file.

    Args:
        path: Path to a JSON file shaped like
            {"name": ..., "current_role": ..., "experience_years": ...,
             "target_ai_role": ...}.

    Returns:
        A validated DeveloperProfile.

    Raises:
        ProfileValidationError: The file can't be read, isn't valid JSON,
            isn't a JSON object, is missing a required field, or a field
            fails validation.
    """
    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ProfileValidationError(f"Could not read profile file: {path}") from exc

    try:
        payload = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ProfileValidationError(f"Profile file is not valid JSON: {path}") from exc

    if not isinstance(payload, dict):
        raise ProfileValidationError(f"Profile file must contain a JSON object: {path}")

    missing = [field for field in _REQUIRED_FIELDS if field not in payload]
    if missing:
        raise ProfileValidationError(
            f"Profile is missing required field(s): {', '.join(missing)}"
        )

    return DeveloperProfile(
        name=payload["name"],
        current_role=payload["current_role"],
        experience_years=payload["experience_years"],
        target_ai_role=payload["target_ai_role"],
    )


def load_profiles(paths: list[Path]) -> list[DeveloperProfile]:
    """Load and validate multiple profiles, in order.

    Raises on the first invalid profile encountered. Callers that need to
    load the valid ones while separately rejecting a bad one should call
    `load_profile` per path themselves, as the tests in this repo do.
    """
    return [load_profile(path) for path in paths]
