"""Orchestration: generate -> validate -> (optional) evaluate -> score.
Wiring is here; the workstream bodies live in their stub modules. Run pieces as they
come online. TODO(anderson): connect once cohort + generators are implemented.

    python scripts/run_all.py --generate     # build outputs/benchmark.jsonl
    python scripts/run_all.py --evaluate      # run Longevity-LLM over the benchmark
    python scripts/run_all.py --score         # metrics + bonus reasoning scorer
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import config


def generate():
    """cohort = build_cohort(); records = sum(gen(cohort) ...); write outputs/benchmark.jsonl"""
    raise NotImplementedError("Anderson: wire build_cohort + generators + redflags + matched effects.")


def evaluate():
    """For each record: chat(messages) -> parse(fmt) -> store answer + reasoning + parse_status."""
    raise NotImplementedError("Anderson: run model over benchmark.jsonl into outputs/eval.jsonl.")


def score():
    """metrics per task vs baselines; bonus = deterministic.check_trace + judge.judge; report kappa."""
    raise NotImplementedError("Anderson: compute metrics + bonus reasoning scores into outputs/scores.json.")


if __name__ == "__main__":
    config.OUTPUTS_DIR.mkdir(exist_ok=True)
    ap = argparse.ArgumentParser()
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--score", action="store_true")
    a = ap.parse_args()
    if a.generate: generate()
    if a.evaluate: evaluate()
    if a.score: score()
    if not any([a.generate, a.evaluate, a.score]):
        ap.print_help()
