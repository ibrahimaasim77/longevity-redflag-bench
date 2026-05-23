"""Honest diagnostic: is the model really this bad, or are we using it wrong?

Tests:
  1. What model does the endpoint actually claim to be?
  2. With the dataset's original system prompt, does the hallucination still happen?
  3. With thinking disabled, does it still happen?
  4. With a fully empty input ("hello"), what does it say?
"""
from __future__ import annotations
import os
import requests
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from datasets import load_dataset

BASE_URL = os.environ.get("LONGEVITY_BASE_URL", "https://swchnq0ekc3scmqw.us-east-2.aws.endpoints.huggingface.cloud/v1")
API_KEY = os.environ.get("HF_TOKEN", "")
MODEL = "longevity-llm"

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)

print("=" * 70)
print("TEST 1 — What does the endpoint say it is?")
print("=" * 70)
try:
    r = requests.get(f"{BASE_URL}/models", headers={"Authorization": f"Bearer {API_KEY}"}, timeout=15)
    print(r.json())
except Exception as e:
    print(f"  /models failed: {e}")

print("\n" + "=" * 70)
print("TEST 2 — Trivial prompt: 'hello'")
print("=" * 70)
r = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "hello"}],
    temperature=0.0,
    max_tokens=400,
)
print(r.choices[0].message.content)

print("\n" + "=" * 70)
print("TEST 3 — Pull dataset's OWN system prompt + a patient, send verbatim")
print("=" * 70)
ds = load_dataset("insilicomedicine/longebench", "benchmark", split="eval")
nhanes = ds.filter(
    lambda r: "nhanes" in (r.get("display_name") or "").lower()
    and "mortality" in (r.get("display_name") or "").lower()
    and r.get("format") == "regression"
)
row = nhanes.shuffle(seed=42).select(range(1))[0]
msgs = list(row["messages"])
print(f"  patient: {row['lb_id']}  gold={msgs[-1]['content']}")
print(f"  system prompt the dataset uses:")
print(f"    {msgs[0]['content']!r}"[:400])

# Send exactly what the dataset ships — no modification
r = client.chat.completions.create(
    model=MODEL,
    messages=[dict(m) for m in msgs[:-1]],  # drop the gold answer
    temperature=0.0,
    max_tokens=1200,
)
out = r.choices[0].message.content or ""
print(f"\n  tokens: in={r.usage.prompt_tokens} out={r.usage.completion_tokens}")
print(f"\n  raw output:\n{out}")

print("\n" + "=" * 70)
print("TEST 4 — Thinking disabled (Qwen template flag)")
print("=" * 70)
try:
    r = client.chat.completions.create(
        model=MODEL,
        messages=[dict(m) for m in msgs[:-1]],
        temperature=0.0,
        max_tokens=400,
        extra_body={"chat_template_kwargs": {"enable_thinking": False}},
    )
    out = r.choices[0].message.content or ""
    print(f"  tokens: in={r.usage.prompt_tokens} out={r.usage.completion_tokens}")
    print(f"  raw output:\n{out}")
except Exception as e:
    print(f"  endpoint rejected thinking=False: {type(e).__name__}: {str(e)[:200]}")

print("\n" + "=" * 70)
print("TEST 5 — Just the BMI question, ZERO labs, see what it invents")
print("=" * 70)
r = client.chat.completions.create(
    model=MODEL,
    messages=[{"role": "user", "content": "BMI is 25. Predict survival in months."}],
    temperature=0.0,
    max_tokens=600,
)
print(r.choices[0].message.content)
