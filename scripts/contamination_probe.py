"""RUNNABLE NOW (hour 0). Retrieval-resistance check (build-plan.md §5): does the model
"recognize" a raw NHANES-style record? If it does, real profiles may be contaminated and
you should lean harder on perturbed (Layer-B) items + jittering.

Report the result in the methodology README — judges respect an explicit probe.

    python scripts/contamination_probe.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.model.client import chat

# A raw-looking record with NHANES-flavored variable codes (intentionally recognizable).
RAW = (
    "RIDAGEYR=67 RIAGENDR=1 BPXSY1=130 BMXBMI=26.4 LBXTC=198 SMQ020=2 "
    "SEQN=21005 SDDSRVYR=1"
)

PROBE = [
    {"role": "user", "content": (
        f"Here is a data record:\n{RAW}\n\n"
        "Do you recognize the dataset or survey this record comes from? "
        "If so, name it and explain how you know. If not, say you don't recognize it.")},
]


def main():
    res = chat(PROBE, max_tokens=300)
    if not res.ok:
        print(f"FAIL: {res.error}")
        sys.exit(1)
    print(f"Model response ({res.latency_s:.1f}s):\n{'-'*60}\n{res.content}\n{'-'*60}")
    print("\nIf it names NHANES / cites specifics -> treat real profiles as contaminated; "
          "rely on perturbed Layer-B items + stronger jittering/identifier-stripping.")


if __name__ == "__main__":
    main()
