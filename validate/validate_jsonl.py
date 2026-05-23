"""RUNNABLE NOW. Pre-freeze gate for benchmark.jsonl (LongevityBench format).

Checks (track-01-spec.md + build-plan.md §6):
  - every line parses into BenchmarkRecord (schema valid; gold is trailing assistant turn;
    metric matches format; metadata is valid JSON)
  - every prompt (messages minus gold) <= 30K tokens cl100k_base
  - >= MIN rows per task (task = lb_id; rows of a task share lb_id)
  - covariate split present (metadata.split): train + test both present per task
  - no metadata.base_profile_id leaks across splits
  - class balance for binary tasks (gold-letter distribution)

Hard failures (schema error, token overflow, leakage) -> exit 1. Counts/balance -> warnings.

    python validate/validate_jsonl.py mock/mock_records.jsonl --min-per-task 1
    python validate/validate_jsonl.py outputs/benchmark.jsonl
"""

import argparse
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schema.records import BenchmarkRecord, Format, prompt_text
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
    task_name = {}
    per_task_split = defaultdict(Counter)
    binary_gold = defaultdict(Counter)
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

            meta = rec.meta()
            per_task[rec.lb_id] += 1
            task_name[rec.lb_id] = f"{rec.pool} [{rec.format.value}/{rec.metric.value}]"
            split = meta.get("split")
            if split:
                per_task_split[rec.lb_id][split] += 1
            pid = meta.get("base_profile_id")
            if pid and split:
                profile_splits[pid].add(split)
            if rec.format == Format.binary:
                binary_gold[rec.lb_id][rec.gold().upper()] += 1

            if enc is not None:
                tok = len(enc.encode(prompt_text(rec)))
                if tok > MAX_PROMPT_TOKENS_CL100K:
                    print(f"  ✗ {rec.lb_id}: {tok} tokens > {MAX_PROMPT_TOKENS_CL100K} (cl100k)")
                    hard_fail = True

    print(f"\nrecords: {n}")
    print("per task (lb_id; target >= %d rows):" % args.min_per_task)
    for t, c in sorted(per_task.items()):
        splits = dict(per_task_split[t])
        flag = "ok" if c >= args.min_per_task else "LOW"
        if flag == "LOW":
            warnings.append(f"{t}: {c} < {args.min_per_task}")
        print(f"   {t}  {task_name[t]:42s} {c:4d}  splits={splits}  [{flag}]")
        if splits and ("train" not in splits or "test" not in splits):
            warnings.append(f"{t}: missing a split ({splits})")
        elif not splits:
            warnings.append(f"{t}: no metadata.split set")

    for t, gold in binary_gold.items():
        total = sum(gold.values())
        top = max(gold.values()) / total if total else 0
        print(f"\nbinary gold balance {t}: {dict(gold)}  (majority {top:.0%})")
        if top > 0.75:
            warnings.append(f"{t}: gold-label imbalance (majority {top:.0%}) — report balanced metrics / rebalance")

    leaks = [p for p, s in profile_splits.items() if len(s) > 1]
    if leaks:
        print(f"\n  ✗ {len(leaks)} base_profile_id(s) in BOTH splits (leakage): {leaks[:5]}")
        hard_fail = True

    print("\nwarnings:")
    for w in warnings or ["none"]:
        print(f"  - {w}")

    print("\nRESULT:", "FAIL" if hard_fail else "PASS")
    sys.exit(1 if hard_fail else 0)


if __name__ == "__main__":
    main()
