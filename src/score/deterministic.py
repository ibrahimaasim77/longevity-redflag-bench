"""Bonus track — the PROGRAMMATIC half of the reasoning-verification scorer
(grading-rubric-spec.md §1). Runs before the LLM judge, needs no API calls, and is
what makes this a "programmatic scoring function" rather than just "ask Claude".

Extracts (factor -> asserted direction) claims from a reasoning trace and checks them
against the bio red-flag table. Fires objective, citable flags.

This is a working skeleton: the claim-extraction is keyword/regex based. TODO(anderson):
expand the lexicon and add the fabricated-quantity + false-attribution checks as you
see real traces. Bio team owns the correctness criteria (context_cases.yaml).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List

from src.redflags.redflags import RedFlag

# direction lexicon
_INCREASE = re.compile(r"\b(increase|raise|elevat|worse|higher|reduc\w* survival|shorten)\w*", re.I)
_DECREASE = re.compile(r"\b(decrease|lower|reduc\w* risk|protect|improv|better|longer survival)\w*", re.I)
_HR_NUM = re.compile(r"\b(hazard ratio|hr|relative risk|rr)\s*[:=]?\s*\d+(\.\d+)?", re.I)


@dataclass
class DeterministicResult:
    flags: List[str] = field(default_factory=list)
    detail: Dict[str, str] = field(default_factory=dict)

    @property
    def clean(self) -> bool:
        return not self.flags


def check_trace(reasoning: str, red_flag: RedFlag) -> DeterministicResult:
    """Check one reasoning trace against the ground-truth direction of its red flag."""
    res = DeterministicResult()
    text = reasoning or ""

    # 1. directional contradiction: did the model assert the opposite of the truth?
    asserts_increase = bool(_INCREASE.search(text))
    asserts_decrease = bool(_DECREASE.search(text))
    truth = red_flag.direction
    if truth == "increase_risk" and asserts_decrease and not asserts_increase:
        res.flags.append("directional_contradiction")
        res.detail["directional_contradiction"] = f"{red_flag.key}: trace implies decrease, truth=increase"
    if truth == "decrease_risk" and asserts_increase and not asserts_decrease:
        res.flags.append("directional_contradiction")
        res.detail["directional_contradiction"] = f"{red_flag.key}: trace implies increase, truth=decrease"

    # 2. fabricated quantity: an HR/RR number stated for a flag we only have a band for
    if _HR_NUM.search(text) and not red_flag.hr_band:
        res.flags.append("fabricated_effect_size")

    # 3. context omission: a `should_moderate` factor never mentioned
    missing = [m for m in red_flag.should_moderate
               if m and not re.search(re.escape(m.split()[0]), text, re.I)]
    if red_flag.should_moderate and len(missing) == len(red_flag.should_moderate):
        res.flags.append("context_ignored")
        res.detail["context_ignored"] = "none of should_moderate mentioned: " + ", ".join(red_flag.should_moderate)

    return res
