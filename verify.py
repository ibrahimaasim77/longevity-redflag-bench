"""End-to-end verification. ONE Longevity-LLM call + ONE Groq call.

Confirms:
  1. Longevity-LLM endpoint is the one the user supplied.
  2. The model's <think> block contains fabricated labs not in the prompt.
  3. Groq llama-3.3-70b-versatile responds correctly.
"""
from __future__ import annotations

import json
import os
import re
from dotenv import load_dotenv
load_dotenv()
from datasets import load_dataset
from openai import OpenAI

LONGEVITY_URL = os.environ.get("LONGEVITY_BASE_URL", "https://swchnq0ekc3scmqw.us-east-2.aws.endpoints.huggingface.cloud/v1")
LONGEVITY_KEY = os.environ.get("HF_TOKEN", "")
LONGEVITY_MODEL = os.environ.get("LONGEVITY_MODEL", "longevity-llm")

GROQ_URL = "https://api.groq.com/openai/v1"
GROQ_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"

print("=" * 70)
print("STEP 1 — confirm endpoint identity")
print("=" * 70)
print(f"Longevity-LLM base_url: {LONGEVITY_URL}")
print(f"Longevity-LLM model:    {LONGEVITY_MODEL}")
print(f"Groq base_url:          {GROQ_URL}")
print(f"Groq model:             {GROQ_MODEL}")

print("\n" + "=" * 70)
print("STEP 2 — pull ONE real NHANES patient (same seed as runs.jsonl)")
print("=" * 70)
ds = load_dataset("insilicomedicine/longebench", "benchmark", split="eval")
nhanes = ds.filter(
    lambda r: "nhanes" in (r.get("display_name") or "").lower()
    and "mortality" in (r.get("display_name") or "").lower()
    and r.get("format") == "regression"
)
sample = nhanes.shuffle(seed=42).select(range(1))
row = sample[0]
msgs = list(row["messages"])
prompt_text = msgs[-2]["content"] if len(msgs) >= 2 else msgs[-1]["content"]
gold = msgs[-1]["content"]
print(f"lb_id={row['lb_id']}  gold_survival_months={gold}")
print("\n--- ACTUAL PATIENT (from HuggingFace dataset) ---")
print(prompt_text[:1400])

print("\n" + "=" * 70)
print("STEP 3 — call Longevity-LLM, capture FULL raw response")
print("=" * 70)
client = OpenAI(base_url=LONGEVITY_URL, api_key=LONGEVITY_KEY)
input_msgs = [dict(m) for m in msgs[:-1]]
input_msgs[-1] = dict(input_msgs[-1])
input_msgs[-1]["content"] += "\n\nAfter your answer, explain your reasoning in 2 sentences."

resp = client.chat.completions.create(
    model=LONGEVITY_MODEL,
    messages=input_msgs,
    temperature=0.0,
    max_tokens=1200,
)
raw = resp.choices[0].message.content or ""
u = resp.usage
print(f"tokens: prompt={u.prompt_tokens} completion={u.completion_tokens}")
print(f"\n--- RAW MODEL OUTPUT (verbatim, no stripping) ---")
print(raw)

print("\n" + "=" * 70)
print("STEP 4 — hallucination diff: labs in <think> vs labs in prompt")
print("=" * 70)

# Extract labs from prompt (real)
def find_labs(text):
    pat = r"([A-Z][A-Za-z\s\-]{2,40}?)\s*\(([^)]+)\)\s*(\d+\.?\d*)"
    return {m.group(1).strip().lower(): (m.group(3), m.group(2)) for m in re.finditer(pat, text)}

real_labs = find_labs(prompt_text)
print(f"Prompt has {len(real_labs)} labeled labs. A few examples:")
for k, (v, unit) in list(real_labs.items())[:6]:
    print(f"  REAL: {k} = {v} ({unit})")

# Pull labs the model mentions in its thinking
thinking = raw.split("</think>")[0] if "</think>" in raw else raw
mentions = re.findall(
    r"\b(BMI|HbA1c|CRP|albumin|cholesterol|HDL|LDL|glucose|creatinine|hemoglobin|"
    r"obesity|diabetes|hypertension)\b[^.]{0,80}?(\d+\.?\d*)",
    thinking, flags=re.IGNORECASE,
)
print(f"\nModel <think> mentions {len(mentions)} numeric clinical claims:")
for term, val in mentions[:10]:
    print(f"  CLAIM: {term} ≈ {val}")
print("\n(Compare these to the real labs above. Any mismatch = fabrication.)")

print("\n" + "=" * 70)
print("STEP 5 — call Groq judge with the real prompt + model's reasoning")
print("=" * 70)
groq = OpenAI(base_url=GROQ_URL, api_key=GROQ_KEY)
judge_user = (
    "Return JSON only: "
    '{"context_integration": 0-2, "no_fabrication": 0-1, '
    '"verdict": "context-aware|keyword-reactive|mixed", "rationale": "<40 words"}\n\n'
    f"PROMPT:\n{prompt_text[:900]}\n\nMODEL REASONING:\n{thinking[:900]}\n\nJSON:"
)
gr = groq.chat.completions.create(
    model=GROQ_MODEL,
    messages=[
        {"role": "system", "content": "You are a strict judge. Return JSON only."},
        {"role": "user", "content": judge_user},
    ],
    temperature=0.0,
    max_tokens=200,
)
print(f"Groq tokens: prompt={gr.usage.prompt_tokens} completion={gr.usage.completion_tokens}")
print(f"Groq verdict:\n{gr.choices[0].message.content}")

print("\n" + "=" * 70)
print("DONE")
print("=" * 70)
