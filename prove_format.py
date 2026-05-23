"""Prove we're not sending the wrong format. Same patient, three send paths.

Path A — verbatim dataset row (no modifications). This is exactly what Insilico
         ships. If A hallucinates, the model is the source.
Path B — what fetch_real_runs.py sends (with appended reasoning request).
Path C — raw HTTPS POST showing the exact JSON wire payload.
"""
from __future__ import annotations
import json
import os
import re
import requests
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
from datasets import load_dataset

BASE = os.environ.get("LONGEVITY_BASE_URL", "https://swchnq0ekc3scmqw.us-east-2.aws.endpoints.huggingface.cloud/v1")
KEY = os.environ.get("HF_TOKEN", "")
MODEL = "longevity-llm"

client = OpenAI(base_url=BASE, api_key=KEY)

ds = load_dataset("insilicomedicine/longebench", "benchmark", split="eval")
nhanes = ds.filter(
    lambda r: "nhanes" in (r.get("display_name") or "").lower()
    and "mortality" in (r.get("display_name") or "").lower()
    and r.get("format") == "regression"
)
row = nhanes.shuffle(seed=42).select(range(1))[0]
msgs = list(row["messages"])
gold = msgs[-1]["content"]
print(f"Patient: {row['lb_id']}  gold_months={gold}")
print(f"BMI in prompt: ", end="")
m = re.search(r"BMI\s+(\d+\.?\d*)", msgs[-2]["content"])
print(m.group(1) if m else "?")
real_hba1c = re.search(r"Glycohemoglobin.*?(\d+\.?\d*)", msgs[-2]["content"])
print(f"HbA1c in prompt: {real_hba1c.group(1) if real_hba1c else '?'}")


def quick_check(answer_text):
    """Return list of (claim, value) from the thinking block."""
    return re.findall(
        r"\b(BMI|HbA1c|glycohemoglobin|CRP|albumin|cholesterol|HDL|glucose|"
        r"obesity|hypertension|diabetes)\b[^.]{0,60}?(\d+\.?\d*)",
        answer_text, flags=re.IGNORECASE,
    )


print("\n" + "=" * 70)
print("PATH A — verbatim dataset row (literally what Insilico published)")
print("=" * 70)
print(f"messages we send: system + user, NO modifications")
r = client.chat.completions.create(
    model=MODEL,
    messages=[dict(m) for m in msgs[:-1]],
    temperature=0.0,
    max_tokens=1200,
)
raw_a = r.choices[0].message.content or ""
think_a = raw_a.partition("</think>")[0]
answer_a = raw_a.partition("</think>")[2].strip()
print(f"  in={r.usage.prompt_tokens} out={r.usage.completion_tokens}")
print(f"  answer: {answer_a!r}")
print(f"  claims in thinking: {quick_check(think_a)[:8]}")


print("\n" + "=" * 70)
print("PATH B — our fetch_real_runs.py format (system + user + reasoning suffix)")
print("=" * 70)
msgs_b = [dict(m) for m in msgs[:-1]]
msgs_b[-1] = dict(msgs_b[-1])
msgs_b[-1]["content"] += "\n\nAfter your answer, explain your reasoning in 2 sentences."
r = client.chat.completions.create(model=MODEL, messages=msgs_b, temperature=0.0, max_tokens=1200)
raw_b = r.choices[0].message.content or ""
think_b = raw_b.partition("</think>")[0]
answer_b = raw_b.partition("</think>")[2].strip()
print(f"  in={r.usage.prompt_tokens} out={r.usage.completion_tokens}")
print(f"  answer: {answer_b!r}")
print(f"  claims in thinking: {quick_check(think_b)[:8]}")


print("\n" + "=" * 70)
print("PATH C — raw HTTPS POST. Showing the EXACT JSON we put on the wire.")
print("=" * 70)
payload = {
    "model": MODEL,
    "messages": [dict(m) for m in msgs[:-1]],
    "temperature": 0.0,
    "max_tokens": 1200,
}
print("JSON payload (first 800 chars):")
print(json.dumps(payload, indent=2)[:800] + "...")
http = requests.post(
    f"{BASE}/chat/completions",
    headers={"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"},
    data=json.dumps(payload),
    timeout=120,
)
print(f"\nHTTP status: {http.status_code}")
data = http.json()
raw_c = data["choices"][0]["message"]["content"]
answer_c = raw_c.partition("</think>")[2].strip()
print(f"  in={data['usage']['prompt_tokens']} out={data['usage']['completion_tokens']}")
print(f"  answer: {answer_c!r}")


print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)
print(f"  Path A (dataset verbatim):  answer={answer_a!r}")
print(f"  Path B (our reasoning ask): answer={answer_b!r}")
print(f"  Path C (raw HTTP):          answer={answer_c!r}")
print(f"  Gold:                       answer={gold!r}")
print()
print("If A, B, and C agree → we're sending it correctly; the model is just like this.")
print("If A is fine but B/C aren't → it's our pipeline.")
