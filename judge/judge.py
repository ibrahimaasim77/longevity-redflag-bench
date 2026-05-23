"""Judge real Longevity-LLM runs with Groq llama-3.3-70b-versatile.

Reads judge/runs.jsonl, writes judge/scores.jsonl (append; skip items already scored).

Token budget:
  - prompts kept under ~400 tokens via truncation
  - judge max_tokens = 300
  - batches of 5 with 1 s sleep between batches
  - prints running cost estimate every 10 calls

Reliability:
  - try/except every API call, retry once on failure
  - strip ```json fences and </think> blocks before parsing
  - parse_status: ok | malformed | refusal | missing | error
  - never crashes the loop
"""

from __future__ import annotations

import json
import re
import sys
import time
from pathlib import Path

import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI

JUDGE_BACKEND = "groq"

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

# Approx Groq pricing for llama-3.3-70b-versatile ($/1M tokens) — only for the
# running estimate; not load-bearing.
PRICE_IN_PER_M = 0.59
PRICE_OUT_PER_M = 0.79

IN_PATH = Path(__file__).parent / "runs.jsonl"
OUT_PATH = Path(__file__).parent / "scores.jsonl"

BATCH_SIZE = 5
BATCH_SLEEP_S = 1.0
JUDGE_MAX_TOKENS = 300

SYSTEM = (
    "You are a strict scientific judge of clinical reasoning. "
    "Return ONLY a JSON object — no prose, no markdown fences, no commentary."
)

RUBRIC_TEMPLATE = """\
Score the model's clinical reasoning on these axes. Return JSON only:
{{
  "context_integration": 0-2,
  "keyword_reactivity": 0-2,
  "delta_calibration": 0-1,
  "no_fabrication": 0-1,
  "reasoning_score": 0.0-1.0,
  "verdict": "context-aware" | "keyword-reactive" | "mixed",
  "rationale": "under 40 words"
}}

GUIDANCE:
- context_integration: 2 = explicitly uses facts from the prompt; 0 = ignores them.
- keyword_reactivity: 2 = answer is just a reflex to a keyword (e.g. "obesity"); 0 = earned.
- delta_calibration: 1 = predicted magnitude is proportional to the evidence.
- no_fabrication: 1 = no invented labs/diagnoses absent from the prompt.

PROMPT EXCERPT:
{prompt}

MODEL PREDICTION: {pred} months
MODEL REASONING:
{reasoning}

GOLD: {gold}

JSON only:"""


def strip_fences(text: str) -> str:
    text = (text or "").strip()
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def parse_judge_json(raw: str):
    text = strip_fences(raw)
    if not text:
        return None, "refusal"
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1 and end > start:
            return json.loads(text[start : end + 1]), "ok"
    except Exception:
        pass
    return None, "malformed"


def truncate(s, n: int) -> str:
    s = "" if s is None else str(s)
    return s if len(s) <= n else s[:n] + "…"


def load_seen(out_path: Path) -> set:
    seen = set()
    if not out_path.exists():
        return seen
    with out_path.open() as f:
        for line in f:
            try:
                seen.add(json.loads(line)["item_id"])
            except Exception:
                continue
    return seen


def score_one(client, run: dict):
    prompt = truncate(run.get("prompt_xml") or "", 1100)
    reasoning = truncate(run.get("reasoning") or "", 500)
    gold = truncate(run.get("gold"), 150)
    pred = run.get("prediction_months")

    user = RUBRIC_TEMPLATE.format(
        prompt=prompt, pred=pred, reasoning=reasoning, gold=gold
    )

    last_err = None
    for attempt in range(2):
        try:
            resp = client.chat.completions.create(
                model=GROQ_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM},
                    {"role": "user", "content": user},
                ],
                temperature=0.0,
                max_tokens=JUDGE_MAX_TOKENS,
            )
            raw = resp.choices[0].message.content or ""
            u = resp.usage
            tok_in = getattr(u, "prompt_tokens", 0) or 0
            tok_out = getattr(u, "completion_tokens", 0) or 0

            parsed, status = parse_judge_json(raw)
            if status == "ok" and parsed is not None:
                parsed["parse_status"] = "ok"
                return parsed, tok_in, tok_out
            if attempt == 0:
                last_err = f"parse:{status}"
                continue
            return (
                {"parse_status": status, "raw": (raw or "")[:200]},
                tok_in,
                tok_out,
            )
        except Exception as e:
            last_err = f"{type(e).__name__}: {str(e)[:160]}"
            if attempt == 0:
                time.sleep(0.5)
                continue
            return {"parse_status": "error", "error": last_err}, 0, 0

    return {"parse_status": "missing", "error": last_err}, 0, 0


def main():
    if not IN_PATH.exists():
        print(f"ERROR: {IN_PATH} not found. Run fetch_real_runs.py first.")
        sys.exit(1)

    seen = load_seen(OUT_PATH)
    print(f"Already scored: {len(seen)} item_ids in {OUT_PATH.name}")

    runs = []
    with IN_PATH.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if r.get("item_id") and r["item_id"] not in seen:
                    runs.append(r)
            except Exception as e:
                print(f"  skip malformed run line: {e}")

    if not runs:
        print("Nothing new to score.")
        return

    print(
        f"Scoring {len(runs)} runs with {GROQ_MODEL} "
        f"(batch={BATCH_SIZE}, sleep={BATCH_SLEEP_S}s, max_tokens={JUDGE_MAX_TOKENS})\n"
    )

    client = OpenAI(base_url=GROQ_BASE_URL, api_key=GROQ_API_KEY)

    total_in = total_out = 0
    n_done = 0

    with OUT_PATH.open("a") as out:
        for batch_idx in range(0, len(runs), BATCH_SIZE):
            batch = runs[batch_idx : batch_idx + BATCH_SIZE]
            for run in batch:
                t0 = time.time()
                score, tok_in, tok_out = score_one(client, run)
                total_in += tok_in
                total_out += tok_out
                n_done += 1

                record = {
                    "item_id": run["item_id"],
                    "profile_id": run.get("profile_id"),
                    "red_flag": run.get("red_flag"),
                    "model": run.get("model"),
                    "prediction_months": run.get("prediction_months"),
                    **score,
                }
                out.write(json.dumps(record) + "\n")
                out.flush()

                ps = score.get("parse_status", "?")
                rs = score.get("reasoning_score", "—")
                vd = score.get("verdict", "—")
                print(
                    f"  [{n_done}/{len(runs)}] {run['item_id']}  "
                    f"parse={ps}  rs={rs}  v={vd}  toks={tok_in}+{tok_out}  "
                    f"({time.time()-t0:.1f}s)"
                )

                if n_done % 10 == 0:
                    cost = (total_in / 1e6) * PRICE_IN_PER_M + (
                        total_out / 1e6
                    ) * PRICE_OUT_PER_M
                    print(
                        f"    -- running: in={total_in} out={total_out} "
                        f"cost≈${cost:.4f}"
                    )

            if batch_idx + BATCH_SIZE < len(runs):
                time.sleep(BATCH_SLEEP_S)

    cost = (total_in / 1e6) * PRICE_IN_PER_M + (total_out / 1e6) * PRICE_OUT_PER_M
    print(
        f"\nDONE. Scored {n_done} runs. Tokens in={total_in} out={total_out}. "
        f"Cost ≈ ${cost:.4f}"
    )
    print(f"Output: {OUT_PATH}")


if __name__ == "__main__":
    main()
