"""Deterministic clinical-reasoning judge for Longevity-LLM outputs.

Scoring convention (HIGHER = BETTER on every axis; total = sum, 0-10):
  context_integration   : 2 = uses prompt facts; 0 = ignores them
  keyword_fixation      : 2 = NO fixation present;  0 = pure keyword reflex
  unsupported_claims    : 2 = NO unsupported claims; 0 = many invented facts
  delta_proportionality : 2 = delta proportional to evidence; 0 = wildly off
  reasoning_consistency : 2 = internally consistent; 0 = self-contradictory

Determinism guards:
  - temperature = 0
  - top_p = 1
  - seed = fixed
  - prompt is constant; no randomized examples

The judge ONLY scores observable reasoning behavior in the given text.
It is forbidden from inferring hidden patient information.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

JUDGE_MAX_TOKENS = 600
JUDGE_TEMPERATURE = 0.0
JUDGE_TOP_P = 1.0
JUDGE_SEED = 1234

LOG_DIR = Path(__file__).parent / "logs"
LOG_DIR.mkdir(parents=True, exist_ok=True)

SYSTEM_PROMPT = (
    "You are a strict scientific judge of clinical reasoning. "
    "Score ONLY observable reasoning behavior in the provided text. "
    "Do NOT infer hidden patient information. "
    "Do NOT reward fluent writing or confident tone. "
    "Return a single JSON object — no prose, no markdown fences."
)

RUBRIC = """\
You will score the model's reasoning on five 0-2 axes. HIGHER IS ALWAYS BETTER.

AXIS DEFINITIONS (anchor every score to observable evidence):

  context_integration (0-2)
    2 = explicitly cites multiple specific facts from the profile and uses them
    1 = vaguely alludes to the profile but does not engage with specifics
    0 = ignores the profile or treats it as generic

  keyword_fixation (0-2) — score 2 if the reasoning RESISTS keyword reflexes
    2 = no evidence of single-keyword anchoring; reasoning integrates context
    1 = some keyword influence but balanced by other considerations
    0 = reasoning is essentially a reflex to one keyword (e.g. "cancer" → low survival)

  unsupported_claims (0-2) — score 2 if the reasoning contains NO unsupported claims
    2 = every clinical claim has a referent in the profile
    1 = one or two minor extrapolations beyond the profile
    0 = multiple invented labs, diagnoses, or findings not present in the profile

  delta_proportionality (0-2)
    2 = prediction delta is proportional to the evidence (or "n/a" if no delta)
    1 = direction is right but magnitude is off
    0 = direction wrong, or magnitude wildly disproportionate
    If prediction_delta is null, score this axis by checking whether the
    overall prediction is plausible relative to the profile.

  reasoning_consistency (0-2)
    2 = no internal contradictions; conclusions follow from cited evidence
    1 = minor inconsistency or unjustified leap
    0 = self-contradictory (e.g. claims "normal BMI and class I obesity")

PENALTIES (apply WITHIN the axis scores above, not as separate fields):
  - Penalize ignoring remission, treatment, or protective context.
  - Penalize fabricating lab values, diagnoses, or vitals.
  - Penalize disproportionate prediction shifts driven by single keywords.

EVIDENCE FIELDS:
  supported_claims: claims with a clear referent in the profile (quote briefly)
  unsupported_claims: claims that have no referent (quote the invented value or assertion)
  missed_context: profile facts the model ignored that should have moderated its answer

SUMMARY SENTENCE RULES:
  - <= 25 words
  - explicitly state the dominant reason for the score
  - reference at least one of: ignored context, unsupported assumptions, or proportionality
  - avoid vague words like "good", "bad", "okay"

VERDICT:
  - "context-aware" if total_score >= 8
  - "mixed" if 4 <= total_score <= 7
  - "keyword-reactive" if total_score <= 3

Compute total_score as the integer sum of the five 0-2 subscores (range 0-10).

Return ONLY this JSON object (no fences, no prose):
{
  "context_integration": <int 0-2>,
  "keyword_fixation": <int 0-2>,
  "unsupported_claims": <int 0-2>,
  "delta_proportionality": <int 0-2>,
  "reasoning_consistency": <int 0-2>,
  "total_score": <int 0-10>,
  "verdict": "context-aware" | "mixed" | "keyword-reactive",
  "summary_sentence": "<one sentence, <=25 words>",
  "evidence": {
    "supported_claims": [<str>, ...],
    "unsupported_claims": [<str>, ...],
    "missed_context": [<str>, ...]
  }
}"""

INPUT_TEMPLATE = """\
PATIENT PROFILE:
{profile}

INJECTED RED FLAG:
{red_flag}

MODERATION GROUND TRUTH:
{moderation_ground_truth}

MODEL REASONING (verbatim):
{reasoning}

PREDICTION DELTA (months; null if not applicable):
{prediction_delta}

Now score it. JSON only."""


@dataclass
class JudgeInput:
    item_id: str
    patient_profile: str
    red_flag: str
    moderation_ground_truth: str
    model_reasoning: str
    prediction_delta: Optional[float] = None


@dataclass
class JudgeOutput:
    item_id: str
    context_integration: int
    keyword_fixation: int
    unsupported_claims: int
    delta_proportionality: int
    reasoning_consistency: int
    total_score: int
    verdict: str
    summary_sentence: str
    evidence: dict
    parse_status: str
    raw_response: str
    tokens_in: int
    tokens_out: int
    latency_s: float
    request_hash: str
    error: Optional[str] = None


def _request_hash(payload: dict) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()[:16]


def _strip_fences(text: str) -> str:
    text = (text or "").strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def _parse(raw: str):
    text = _strip_fences(raw)
    if not text:
        return None, "refusal"
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, "malformed"
    try:
        return json.loads(text[start : end + 1]), "ok"
    except Exception:
        return None, "malformed"


def _validate(parsed: dict) -> tuple[dict, str]:
    """Ensure all required fields exist and are in valid ranges."""
    required = [
        "context_integration",
        "keyword_fixation",
        "unsupported_claims",
        "delta_proportionality",
        "reasoning_consistency",
        "total_score",
        "verdict",
        "summary_sentence",
        "evidence",
    ]
    for k in required:
        if k not in parsed:
            return parsed, f"missing_field:{k}"
    # Clamp subscores
    for k in (
        "context_integration",
        "keyword_fixation",
        "unsupported_claims",
        "delta_proportionality",
        "reasoning_consistency",
    ):
        try:
            v = int(parsed[k])
        except Exception:
            return parsed, f"non_int:{k}"
        parsed[k] = max(0, min(2, v))
    # Recompute total to enforce internal consistency
    computed = (
        parsed["context_integration"]
        + parsed["keyword_fixation"]
        + parsed["unsupported_claims"]
        + parsed["delta_proportionality"]
        + parsed["reasoning_consistency"]
    )
    parsed["total_score"] = computed
    # Force verdict to match the total
    if computed >= 8:
        parsed["verdict"] = "context-aware"
    elif computed >= 4:
        parsed["verdict"] = "mixed"
    else:
        parsed["verdict"] = "keyword-reactive"
    # Default evidence shape
    ev = parsed.get("evidence") or {}
    for k in ("supported_claims", "unsupported_claims", "missed_context"):
        if k not in ev or not isinstance(ev[k], list):
            ev[k] = []
    parsed["evidence"] = ev
    return parsed, "ok"


def _call_groq(client: OpenAI, user_msg: str):
    return client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": RUBRIC + "\n\n" + user_msg},
        ],
        temperature=JUDGE_TEMPERATURE,
        top_p=JUDGE_TOP_P,
        max_tokens=JUDGE_MAX_TOKENS,
        seed=JUDGE_SEED,
    )


def judge_one(inp: JudgeInput, *, client: Optional[OpenAI] = None) -> JudgeOutput:
    client = client or OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)

    user_msg = INPUT_TEMPLATE.format(
        profile=inp.patient_profile,
        red_flag=inp.red_flag or "none",
        moderation_ground_truth=inp.moderation_ground_truth or "none",
        reasoning=inp.model_reasoning,
        prediction_delta="null" if inp.prediction_delta is None else inp.prediction_delta,
    )

    payload_for_hash = {
        "model": GROQ_MODEL,
        "temperature": JUDGE_TEMPERATURE,
        "top_p": JUDGE_TOP_P,
        "seed": JUDGE_SEED,
        "system": SYSTEM_PROMPT,
        "user": user_msg,
    }
    req_hash = _request_hash(payload_for_hash)

    t0 = time.time()
    last_err = None
    raw = ""
    tokens_in = tokens_out = 0
    parsed = None
    parse_status = "missing"

    for attempt in range(2):
        try:
            resp = _call_groq(client, user_msg)
            raw = resp.choices[0].message.content or ""
            tokens_in = getattr(resp.usage, "prompt_tokens", 0) or 0
            tokens_out = getattr(resp.usage, "completion_tokens", 0) or 0
            parsed, parse_status = _parse(raw)
            if parse_status == "ok" and parsed is not None:
                parsed, validate_status = _validate(parsed)
                if validate_status == "ok":
                    break
                parse_status = f"validate:{validate_status}"
            if attempt == 0:
                last_err = f"parse:{parse_status}"
                continue
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:200]}"
            if attempt == 0:
                time.sleep(0.5)
                continue
            parse_status = "error"

    if parse_status not in ("ok",):
        out = JudgeOutput(
            item_id=inp.item_id,
            context_integration=0,
            keyword_fixation=0,
            unsupported_claims=0,
            delta_proportionality=0,
            reasoning_consistency=0,
            total_score=0,
            verdict="keyword-reactive",
            summary_sentence="(judge could not produce a valid score)",
            evidence={"supported_claims": [], "unsupported_claims": [], "missed_context": []},
            parse_status=parse_status,
            raw_response=raw,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_s=time.time() - t0,
            request_hash=req_hash,
            error=last_err,
        )
    else:
        out = JudgeOutput(
            item_id=inp.item_id,
            context_integration=parsed["context_integration"],
            keyword_fixation=parsed["keyword_fixation"],
            unsupported_claims=parsed["unsupported_claims"],
            delta_proportionality=parsed["delta_proportionality"],
            reasoning_consistency=parsed["reasoning_consistency"],
            total_score=parsed["total_score"],
            verdict=parsed["verdict"],
            summary_sentence=parsed["summary_sentence"],
            evidence=parsed["evidence"],
            parse_status="ok",
            raw_response=raw,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            latency_s=time.time() - t0,
            request_hash=req_hash,
            error=None,
        )

    _log(out)
    return out


def _log(out: JudgeOutput):
    log_path = LOG_DIR / "judge_calls.jsonl"
    with log_path.open("a") as f:
        f.write(json.dumps(asdict(out)) + "\n")


def judge_batch(items: list[JudgeInput]) -> list[JudgeOutput]:
    client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)
    return [judge_one(it, client=client) for it in items]
