"""Answer extraction from raw model output. Robust by design — a 9B model formats
poorly. Every parse returns a structured result with a `failure_type`; it never raises.

The `failure_type` field is also the data behind the parse-success metric
(grading-rubric-spec.md §4) that the LongevityBench paper says it lacks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import List, Optional

from schema.records import AnswerFormat


@dataclass
class ParsedAnswer:
    answer: object                 # str | float | list[str] | None
    reasoning: Optional[str]
    failure_type: Optional[str]    # None=ok | "malformed" | "refusal" | "missing"
    raw: str

    @property
    def ok(self) -> bool:
        return self.failure_type is None


_REFUSAL = re.compile(r"\b(i (cannot|can't|won't)|as an ai|unable to)\b", re.I)


def _maybe_json(text: str) -> Optional[dict]:
    # tolerate code fences and trailing commas
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    blob = re.sub(r",\s*([}\]])", r"\1", m.group(0))
    try:
        return json.loads(blob)
    except Exception:
        return None


def parse(raw: str, fmt: AnswerFormat) -> ParsedAnswer:
    text = (raw or "").strip()
    if not text:
        return ParsedAnswer(None, None, "missing", raw)
    if _REFUSAL.search(text) and len(text) < 240:
        return ParsedAnswer(None, None, "refusal", raw)

    obj = _maybe_json(text)
    reasoning = obj.get("reasoning") if obj else None
    candidate = obj.get("answer") if obj else None

    try:
        if fmt == AnswerFormat.binary:
            ans = _extract_binary(candidate, text)
        elif fmt in (AnswerFormat.ternary, AnswerFormat.multiple_choice):
            ans = _extract_choice(candidate, text)
        elif fmt == AnswerFormat.pairwise:
            ans = _extract_pairwise(candidate, text)
        elif fmt == AnswerFormat.regression:
            ans = _extract_number(candidate, text)
        elif fmt == AnswerFormat.set_generation:
            ans = _extract_set(candidate, text)
        else:
            ans = candidate
    except Exception:
        return ParsedAnswer(None, reasoning, "malformed", raw)

    if ans is None:
        return ParsedAnswer(None, reasoning, "malformed", raw)
    return ParsedAnswer(ans, reasoning, None, raw)


# --- per-format extractors (extend as you see real outputs) ----------------- #
def _extract_binary(candidate, text):
    s = str(candidate if candidate is not None else text).lower()
    if re.search(r"\b(yes|survives?|alive|true|1)\b", s):
        return "yes"
    if re.search(r"\b(no|dies?|deceased|false|0)\b", s):
        return "no"
    return None


def _extract_choice(candidate, text):
    s = str(candidate if candidate is not None else text).lower()
    for label in ("low", "medium", "moderate", "high"):
        if re.search(rf"\b{label}\b", s):
            return "medium" if label == "moderate" else label
    return None


def _extract_pairwise(candidate, text):
    s = str(candidate if candidate is not None else text).upper()
    m = re.search(r"\b(A|B)\b", s)
    return m.group(1) if m else None


def _extract_number(candidate, text):
    if isinstance(candidate, (int, float)):
        return float(candidate)
    m = re.search(r"-?\d+(\.\d+)?", str(candidate if candidate is not None else text))
    return float(m.group(0)) if m else None


def _extract_set(candidate, text) -> Optional[List[str]]:
    if isinstance(candidate, list):
        return [str(x).strip() for x in candidate]
    # fall back to comma/newline separated
    items = re.split(r"[,\n;]", str(candidate if candidate is not None else text))
    items = [i.strip(" -*•").lower() for i in items if i.strip()]
    return items or None
