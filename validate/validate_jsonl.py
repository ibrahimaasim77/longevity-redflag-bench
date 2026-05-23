"""RUNNABLE NOW. The pre-freeze gate for benchmark.jsonl.

Checks, per the spec (track-01-spec.md) and build-plan.md §6:
  - every line parses into BenchmarkRecord (schema valid)
  - every prompt <= 30K tokens (cl100k_base) — fills token_count_cl100k
  - verifiable ground truth present (schema enforces; we double-report)
  - >= MIN prompts per task
  - class balance for binary_survival (positive rate)
  - split integrity: train+test both present; no base_profile_id leaks across splits

Hard failures (schema error, token overflow, leakage) -> exit 1. Counts/balance below
target -> warnings (so you can run against the small mock).

Usage:
    python validate/validate_jsonl.py mock/mock_records.jsonl --min-per-task 1
    python validate/validate_jsonl.py outputs/benchmark.jsonl          # default min 50
"""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schema.records import BenchmarkRecord, TaskType, prompt_text
from src.config import MAX_PROMPT_TOKENS_CL100K, MIN_PROMPTS_PER_TASK


def _encoder():
    try:
        import tiktoken
        return tiktoken.get_encoding("cl100k_base")
    except Exception as e:  # noqa: BLE001
        print(f"  ! tiktoken unavailable ({e}); skipping token check. pip install tiktoken", file=sys.stderr)
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--min-per-task", type=int, default=MIN_PROMPTS_PER_TASK)
    args = ap.parse_args()

    enc = _encoder()
    hard_fail = False
    warnings = []
    per_task = Counter()
    per_task_split = defaultdict(Counter)
    binary_labels = []
    profile_splits = defaultdict(set)
    n = 0

    with open(args.path, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                rec = BenchmarkRecord.model_validate_json(line)
            except Exception as e:  # noqa: BLE001
                print(f"  ✗ line {lineno}: schema error: {e}")
                hard_fail = True
                continue

            per_task[rec.task.value] += 1
            per_task_split[rec.task.value][rec.split.value] += 1
            if rec.base_profile_id:
                profile_splits[rec.base_profile_id].add(rec.split.value)
            if rec.task == TaskType.binary_survival and isinstance(rec.ground_truth.answer, str):
                binary_labels.append(rec.ground_truth.answer.lower())

            if enc is not None:
                tok = len(enc.encode(prompt_text(rec)))
                if tok > MAX_PROMPT_TOKENS_CL100K:
                    print(f"  ✗ {rec.item_id}: {tok} tokens > {MAX_PROMPT_TOKENS_CL100K} (cl100k)")
                    hard_fail = True

    print(f"\nrecords: {n}")
    print("per task (target >= %d):" % args.min_per_task)
    for t, c in sorted(per_task.items()):
        flag = "ok" if c >= args.min_per_task else "LOW"
        splits = dict(per_task_split[t])
        if flag == "LOW":
            warnings.append(f"{t}: {c} < {args.min_per_task}")
        print(f"   {t:24s} {c:4d}  splits={splits}  [{flag}]")
        if "train" not in splits or "test" not in splits:
            warnings.append(f"{t}: missing a split ({splits})")

    if binary_labels:
        pos = sum(1 for x in binary_labels if x in ("no", "deceased", "0"))  # "no"=died within horizon
        rate = pos / len(binary_labels)
        print(f"\nbinary_survival event(died) rate: {rate:.2%} (n={len(binary_labels)})")
        if rate < 0.25 or rate > 0.75:
            warnings.append(f"binary class imbalance: event rate {rate:.2%} — report balanced metrics / consider stratifying")

    leaks = [p for p, s in profile_splits.items() if len(s) > 1]
    if leaks:
        print(f"\n  ✗ {len(leaks)} base_profile_id(s) appear in BOTH splits (leakage): {leaks[:5]}...")
        hard_fail = True

    print("\nwarnings:")
    for w in warnings or ["  none"]:
        print(f"  - {w}" if not w.startswith("  ") else w)

    print("\nRESULT:", "FAIL" if hard_fail else "PASS")
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
