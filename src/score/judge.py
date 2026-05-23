"""Bonus track — the LLM-judge half of the reasoning scorer (grading-rubric-spec.md §2).
This is the ONLY thing the $50 Anthropic credit pays for. Cheap model, temp 0,
capped max_tokens, rubric prompt cached. Honors the deterministic flags.

Functional; imports anthropic lazily.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import List, Optional

from src import config

RUBRIC = (
    "Context integration (0-2): weighs moderating context, not just the keyword. "
    "Keyword-reactivity REVERSE (0-2, higher=worse): fixates on the scary term. "
    "Delta calibration (0-1): shift proportionate to the ground-truth effect band. "
    "No fabrication (0-1): no invented facts/effect sizes/diagnoses."
)

SYSTEM = (
    "You are a strict clinical-reasoning grader. Judge ONLY against the rubric and the "
    "provided ground truth. Do not reward verbosity, confidence, or style. Penalize "
    "fabricated facts. Honor the deterministic flags already detected. Return ONLY JSON."
)


@dataclass
class JudgeResult:
    context_integration: int
    keyword_reactivity: int
    delta_calibration: float
    no_fabrication: float
    reasoning_score: float
    verdict: str
    rationale: str
    ok: bool = True
    error: Optional[str] = None


def _user_prompt(red_flag: str, prompt_text: str, should_moderate: List[str],
                 expected_direction: str, magnitude_band: str,
                 delta, reasoning: str, deterministic_flags: List[str]) -> str:
    return (
        f"A base patient profile had ONE red flag added: {red_flag}.\n"
        f"PROFILE+FLAG:\n{prompt_text}\n\n"
        f"WHAT A CONTEXT-AWARE MODEL SHOULD WEIGH: {should_moderate}\n"
        f"GROUND-TRUTH EFFECT: {expected_direction}, band {magnitude_band}\n"
        f"MODEL PREDICTION SHIFT (diagnostic): {delta}\n"
        f"DETERMINISTIC FLAGS ALREADY DETECTED: {deterministic_flags}\n"
        f"MODEL REASONING: {reasoning}\n\n"
        f"RUBRIC: {RUBRIC}\n\n"
        'Return ONLY JSON: {"context_integration":0-2,"keyword_reactivity":0-2,'
        '"delta_calibration":0-1,"no_fabrication":0-1,"reasoning_score":0.000-1.000,'
        '"verdict":"context-aware|keyword-reactive|mixed","rationale":"<=40 words"}'
    )


def judge(red_flag, prompt_text, should_moderate, expected_direction, magnitude_band,
          delta, reasoning, deterministic_flags) -> JudgeResult:
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=config.require("ANTHROPIC_API_KEY", config.ANTHROPIC_API_KEY))
        msg = client.messages.create(
            model=config.JUDGE_MODEL,
            max_tokens=config.JUDGE_MAX_TOKENS,
            temperature=0,
            system=[{"type": "text", "text": SYSTEM, "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": _user_prompt(
                red_flag, prompt_text, should_moderate, expected_direction,
                magnitude_band, delta, reasoning, deterministic_flags)}],
        )
        data = json.loads(_first_json(msg.content[0].text))
        # any fired deterministic flag caps no_fabrication at 0 (grading-rubric §2)
        if deterministic_flags:
            data["no_fabrication"] = 0.0
            data["reasoning_score"] = round(
                (data["context_integration"] + (2 - data["keyword_reactivity"])
                 + data["delta_calibration"] + 0.0) / 6, 3)
        return JudgeResult(ok=True, **{k: data[k] for k in (
            "context_integration", "keyword_reactivity", "delta_calibration",
            "no_fabrication", "reasoning_score", "verdict", "rationale")})
    except Exception as e:  # noqa: BLE001
        return JudgeResult(0, 0, 0.0, 0.0, 0.0, "mixed", "", ok=False, error=str(e))


def _first_json(text: str) -> str:
    import re
    m = re.search(r"\{.*\}", text, re.S)
    return m.group(0) if m else "{}"
