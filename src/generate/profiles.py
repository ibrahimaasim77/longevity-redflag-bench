"""STUB — owner: Anderson. Turn a cohort row into a retrieval-resistant feature vector.

Strip NHANES identifiers (SEQN, cycle codes) and variable codes; render clean clinical
values; bin/jitter continuous values within clinically-safe ranges (seeded, reproducible).
Output is the patient-description string that goes into the prompt.
"""

from __future__ import annotations

import random

from src import config


def render_profile(row: dict, rng: random.Random | None = None) -> str:
    """cohort row -> clean, jittered, identifier-free profile text. TODO(anderson)."""
    raise NotImplementedError("Anderson: render + strip + jitter. Keep it reproducible (config.SEED).")


def profile_id_for(row: dict) -> str:
    """Stable opaque id (NOT the SEQN) for provenance + split-leakage checks."""
    raise NotImplementedError("Anderson: hash row to an opaque base_profile_id.")
