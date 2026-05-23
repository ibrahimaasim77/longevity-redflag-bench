"""STUB — owner: Anderson. Apply ONE red flag counterfactually to a base profile.

Changes only the surface (inject the bio team's clinical phrase); the rest of the
profile is untouched. The perturbed profile NEVER reuses the real linked outcome —
its ground truth is the red flag's relative effect (Layer B).
"""

from __future__ import annotations

from src.redflags.redflags import RedFlag


def apply_red_flag(base_profile_text: str, flag: RedFlag) -> str:
    """base profile + one red flag -> perturbed profile text. TODO(anderson)."""
    raise NotImplementedError("Anderson: inject flag.phrase into the profile, surface-only.")
