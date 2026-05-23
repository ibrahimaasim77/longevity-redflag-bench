"""Fetch 10 real NHANES survival profiles from LongevityBench, run them through
Longevity-LLM, and save predictions + reasoning to judge/runs.jsonl.

Known model quirks (see /test.py probe):
  - Model emits a hidden <think>...</think> trace before its answer — strip it.
  - Thinking mode burns 200-600 tokens before any answer — use max_tokens=1200.
  - Model sometimes hallucinates patient details — judge will catch this.
  - No `reasoning_content` field exists; everything lives in `message.content`.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import os
from dotenv import load_dotenv
load_dotenv()
from datasets import load_dataset
from openai import OpenAI

LONGEVITY_BASE_URL = os.environ.get("LONGEVITY_BASE_URL", "https://swchnq0ekc3scmqw.us-east-2.aws.endpoints.huggingface.cloud/v1")
LONGEVITY_API_KEY = os.environ.get("HF_TOKEN", "")
LONGEVITY_MODEL = "longevity-llm"

REASONING_SUFFIX = "\n\nAfter your answer, explain your reasoning in 2 sentences."

OUT_PATH = Path(__file__).parent / "runs.jsonl"
SEED = 42
N = 10
MAX_TOKENS = 1200


def split_thinking_and_answer(raw: str):
    """Longevity-LLM emits a <think>...</think> block then a terse final answer.
    The thinking IS the reasoning — keep it. The final answer is just a number.

    Returns (thinking_text, answer_text). Either may be empty.
    """
    raw = raw or ""
    if "</think>" in raw:
        head, _, tail = raw.partition("</think>")
        # Drop the opening <think> tag if present
        thinking = re.sub(r"^.*?<think>\s*", "", head, flags=re.DOTALL).strip()
        # If no <think> tag was emitted, head IS the thinking
        if not thinking:
            thinking = head.strip()
        return thinking, tail.strip()
    return "", raw.strip()


def extract_prediction_months(text: str):
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:months?|mo\b)", text, flags=re.IGNORECASE)
    if m:
        return float(m.group(1))
    m = re.search(r"\b(\d{1,4}(?:\.\d+)?)\b", text)
    if m:
        return float(m.group(1))
    return None


def main():
    print("Loading insilicomedicine/longebench (benchmark/eval) ...")
    ds = load_dataset("insilicomedicine/longebench", "benchmark", split="eval")
    print(f"  total rows: {len(ds)}")

    # Show available columns once to make schema visible in console
    print(f"  columns: {ds.column_names}")

    # NHANES survival/mortality rows are labeled "NHANES Mortality / Regression"
    # in this dataset (1363 rows). The user's spec said "survival" — same thing.
    nhanes = ds.filter(
        lambda r: "nhanes" in (r.get("display_name") or "").lower()
        and "mortality" in (r.get("display_name") or "").lower()
        and r.get("format") == "regression"
    )
    print(f"  NHANES mortality/regression rows: {len(nhanes)}")

    if len(nhanes) == 0:
        raise SystemExit("No NHANES survival/regression rows matched the filter.")

    n = min(N, len(nhanes))
    sample = nhanes.shuffle(seed=SEED).select(range(n))
    print(f"  sampled {n} profiles (seed={SEED})\n")

    client = OpenAI(base_url=LONGEVITY_BASE_URL, api_key=LONGEVITY_API_KEY)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    tok_in = tok_out = 0
    n_ok = n_fail = 0

    with OUT_PATH.open("w") as f:
        for idx, row in enumerate(sample):
            messages = list(row["messages"])
            gold = messages[-1]["content"]
            input_msgs = [dict(m) for m in messages[:-1]]

            # Append reasoning request to the LAST user message
            last = dict(input_msgs[-1])
            last["content"] = last["content"] + REASONING_SUFFIX
            input_msgs[-1] = last

            lb_id = row.get("lb_id") or f"unknown-{idx}"
            print(f"[{idx+1}/{n}] {lb_id}  display={row.get('display_name','?')[:60]}")
            try:
                resp = client.chat.completions.create(
                    model=LONGEVITY_MODEL,
                    messages=input_msgs,
                    temperature=0.0,
                    max_tokens=MAX_TOKENS,
                )
                raw = resp.choices[0].message.content or ""
                u = resp.usage
                tok_in += u.prompt_tokens
                tok_out += u.completion_tokens
                print(f"    tokens: in={u.prompt_tokens} out={u.completion_tokens}")
            except Exception as e:
                print(f"    FAILED: {type(e).__name__}: {str(e)[:200]}")
                n_fail += 1
                continue

            thinking, answer = split_thinking_and_answer(raw)
            pred = extract_prediction_months(answer) or extract_prediction_months(thinking)
            # Reasoning = the <think> trace (where the model actually reasons).
            # Fall back to the answer text if no thinking was emitted.
            reasoning = (thinking or answer)[:1500]

            print(f"    prediction_months={pred}  reasoning_len={len(reasoning)}  answer={answer[:60]!r}")
            print(f"    gold={gold[:80]!r}")

            record = {
                "item_id": f"{lb_id}-{idx}",
                "profile_id": lb_id,
                "red_flag": "none",
                "model": "longevity-llm",
                "base_prediction_months": None,
                "prediction_months": pred,
                "delta": None,
                "reasoning": reasoning,
                "prompt_xml": json.dumps(input_msgs),
                "gold": gold,
            }
            f.write(json.dumps(record) + "\n")
            n_ok += 1

    print(f"\nWrote {n_ok} runs to {OUT_PATH}  (failures: {n_fail})")
    print(f"Total Longevity-LLM tokens: in={tok_in} out={tok_out}")


if __name__ == "__main__":
    main()
