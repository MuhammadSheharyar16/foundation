"""Tests for the typed DeveloperProfile model and its JSON loader.

Loads real profile files from data/ — two valid, one deliberately invalid —
rather than constructing arbitrary dicts inline.
"""

import json
from pathlib import Path

import pytest

from developer_profile import (
    DeveloperProfile,
    ProfileValidationError,
    load_profile,
    load_profiles,
)

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


# ---------------------------------------------------------------------------
# Valid input
# ---------------------------------------------------------------------------


def test_load_profile_accepts_valid_profile_hamza() -> None:
    profile = load_profile(DATA_DIR / "profile_hamza.json")

    assert isinstance(profile, DeveloperProfile)
    assert profile.name == "Hamza Khan"
    assert profile.current_role == "Software Developer"
    assert profile.experience_years == 5
    assert profile.target_ai_role == "AI Engineer"


def test_load_profile_accepts_valid_profile_sheharyar() -> None:
    profile = load_profile(DATA_DIR / "profile_sheharyar.json")

    assert profile.name == "Muhammad Sheharyar"
    assert profile.experience_years == 6.5
    assert profile.target_ai_role == "AI Platform Engineer"


def test_load_profiles_loads_both_valid_profiles_in_order() -> None:
    profiles = load_profiles(
        [DATA_DIR / "profile_hamza.json", DATA_DIR / "profile_sheharyar.json"]
    )

    assert [p.name for p in profiles] == ["Hamza Khan", "Muhammad Sheharyar"]
    assert all(isinstance(p, DeveloperProfile) for p in profiles)


# ---------------------------------------------------------------------------
# Invalid input
# ---------------------------------------------------------------------------


def test_load_profile_rejects_the_deliberately_invalid_profile() -> None:
    """data/profile_ali_invalid.json has a blank current_role and a negative
    experience_years — either should be enough to reject it."""
    with pytest.raises(ProfileValidationError) as excinfo:
        load_profile(DATA_DIR / "profile_ali_invalid.json")

    assert "current_role" in str(excinfo.value) or "experience_years" in str(excinfo.value)


def test_developer_profile_rejects_negative_experience_years_directly() -> None:
    with pytest.raises(ProfileValidationError):
        DeveloperProfile(
            name="Test Person",
            current_role="Developer",
            experience_years=-1,
            target_ai_role="AI Engineer",
        )


def test_load_profile_rejects_missing_required_field(tmp_path) -> None:
    incomplete = tmp_path / "profile_missing_field.json"
    incomplete.write_text(
        json.dumps({"name": "No Role", "experience_years": 2, "target_ai_role": "AI Engineer"}),
        encoding="utf-8",
    )

    with pytest.raises(ProfileValidationError) as excinfo:
        load_profile(incomplete)

    assert "current_role" in str(excinfo.value)


def test_load_profile_rejects_invalid_json(tmp_path) -> None:
    broken = tmp_path / "profile_broken.json"
    broken.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ProfileValidationError):
        load_profile(broken)


def test_load_profiles_stops_at_the_first_invalid_profile() -> None:
    with pytest.raises(ProfileValidationError):
        load_profiles(
            [DATA_DIR / "profile_hamza.json", DATA_DIR / "profile_ali_invalid.json"]
        )
