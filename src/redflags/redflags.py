"""Loader for the bio team's red-flag table (tasks/redflags.csv).

This table is triple-duty (task-authoring-worksheet.md):
  1. Layer-B ground truth (direction + HR band) for pairwise / set-generation tasks
  2. The keyword-reactive trap definitions
  3. The knowledge base the deterministic reasoning scorer checks claims against
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from typing import Dict, List

from src import config


@dataclass
class RedFlag:
    key: str                       # machine id, e.g. "current_smoker"
    phrase: str                    # clinical phrase injected into the profile
    what_it_tests: str
    should_moderate: List[str]     # what a context-aware model must weigh
    direction: str                 # increase_risk | decrease_risk | no_change
    hr_band: str                   # e.g. "2.0-3.0"
    citation: str
    is_protective_control: bool = False


def load_redflags(path=None) -> Dict[str, RedFlag]:
    path = path or (config.TASKS_DIR / "redflags.csv")
    flags: Dict[str, RedFlag] = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if not row.get("key"):
                continue
            flags[row["key"]] = RedFlag(
                key=row["key"].strip(),
                phrase=row["phrase"].strip(),
                what_it_tests=row.get("what_it_tests", "").strip(),
                should_moderate=[s.strip() for s in row.get("should_moderate", "").split("|") if s.strip()],
                direction=row.get("direction", "").strip(),
                hr_band=row.get("hr_band", "").strip(),
                citation=row.get("citation", "").strip(),
                is_protective_control=str(row.get("is_protective_control", "")).strip().lower() in ("1", "true", "yes"),
            )
    return flags
