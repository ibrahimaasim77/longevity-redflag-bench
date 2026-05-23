"""STUB — owner: Anderson. Phrasing variation for Diversity (build-plan.md §3).

Same underlying item rendered as clinical / lay / tabular wording so the benchmark
can't be solved by surface-pattern shortcuts. Keep ground truth identical across
variants of the same item.
"""

from __future__ import annotations

from enum import Enum


class Phrasing(str, Enum):
    clinical = "clinical"
    lay = "lay"
    tabular = "tabular"


def render(profile_text: str, question: str, style: Phrasing) -> str:
    """Render the prompt in the given style. TODO(anderson)."""
    raise NotImplementedError("Anderson: 3 phrasing templates; GT stays constant.")
