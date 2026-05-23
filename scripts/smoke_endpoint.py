"""RUNNABLE NOW (hour 0). Confirms the Longevity-LLM HF endpoint answers and shows how
it formats output. Run this BEFORE building anything else — the shared credential may
be cold/rate-limited.

    cp .env.example .env   # fill HF_TOKEN
    python scripts/smoke_endpoint.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config
from src.model.client import chat

PROBE = [
    {"role": "system", "content": "You are a clinical longevity model. Answer concisely."},
    {"role": "user", "content": (
        "Patient: age 67, male, never-smoker, SBP 130, BMI 26. "
        "Will this person survive at least 10 years? Reply with JSON "
        '{"answer":"yes|no","reasoning":"..."}.')},
]


def main():
    print(f"base_url = {config.LONGEVITY_BASE_URL}")
    print(f"model    = {config.LONGEVITY_MODEL}")
    print(f"HF_TOKEN = {'set' if config.HF_TOKEN else 'MISSING — fill .env'}\n")
    res = chat(PROBE, max_tokens=200)
    if not res.ok:
        print(f"FAIL after retries: {res.error}")
        print("\nTroubleshooting: endpoint may be scaled-to-zero (wait ~1 min and retry), "
              "token wrong/expired, or LONGEVITY_MODEL should be the endpoint's model id.")
        sys.exit(1)
    print(f"OK ({res.latency_s:.1f}s). Raw response:\n{'-'*60}\n{res.content}\n{'-'*60}")
    print("\nNote how it formats the answer — tune src/model/parse.py to match.")


if __name__ == "__main__":
    main()
