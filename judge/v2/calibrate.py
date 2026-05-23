"""Run judge_v2 on calibration.jsonl, compute Cohen's kappa on verdict
and quadratic-weighted kappa on each subscore. Print disagreement table."""

import json
from pathlib import Path

import numpy as np
from sklearn.metrics import cohen_kappa_score

from judge_v2 import JudgeInput, judge_batch

CAL_PATH = Path(__file__).parent / "calibration.jsonl"
VERDICT_MAP = {"keyword-reactive": 0, "mixed": 1, "context-aware": 2}
SUBSCORES = [
    "context_integration",
    "keyword_fixation",
    "unsupported_claims",
    "delta_proportionality",
    "reasoning_consistency",
]


def load_calibration():
    rows = []
    with CAL_PATH.open() as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def build_inputs(rows):
    return [
        JudgeInput(
            item_id=r["id"],
            patient_profile=r["patient_profile"],
            red_flag=r["red_flag"],
            moderation_ground_truth=r["moderation_ground_truth"],
            model_reasoning=r["model_reasoning"],
            prediction_delta=r.get("prediction_delta"),
        )
        for r in rows
    ]


def main():
    rows = load_calibration()
    inputs = build_inputs(rows)
    print(f"Running judge_v2 on {len(inputs)} calibration items...")
    outputs = judge_batch(inputs)

    human_verdicts = []
    judge_verdicts = []
    human_subs = {k: [] for k in SUBSCORES}
    judge_subs = {k: [] for k in SUBSCORES}
    disagreements = []

    for row, out in zip(rows, outputs):
        hl = row["human_label"]
        hv = hl["verdict"]
        jv = out.verdict

        human_verdicts.append(VERDICT_MAP[hv])
        judge_verdicts.append(VERDICT_MAP[jv])

        for k in SUBSCORES:
            human_subs[k].append(hl[k])
            judge_subs[k].append(getattr(out, k))

        h_total = sum(hl[k] for k in SUBSCORES)
        j_total = out.total_score
        if hv != jv:
            disagreements.append(
                {
                    "id": row["id"],
                    "human_verdict": hv,
                    "judge_verdict": jv,
                    "human_total": h_total,
                    "judge_total": j_total,
                    "delta": abs(h_total - j_total),
                    "judge_summary": out.summary_sentence,
                }
            )

    verdict_kappa = cohen_kappa_score(
        human_verdicts, judge_verdicts, weights="quadratic"
    )
    print(f"\n{'='*60}")
    print(f"VERDICT KAPPA (quadratic-weighted): {verdict_kappa:.3f}")
    print(f"{'='*60}")

    print(f"\n{'Subscore':<25} {'QW Kappa':>10}")
    print("-" * 37)
    for k in SUBSCORES:
        kappa = cohen_kappa_score(
            human_subs[k], judge_subs[k], weights="quadratic"
        )
        print(f"{k:<25} {kappa:>10.3f}")

    disagreements.sort(key=lambda d: d["delta"], reverse=True)
    print(f"\n{'='*60}")
    print(f"DISAGREEMENTS: {len(disagreements)} / {len(rows)}")
    print(f"{'='*60}")
    for d in disagreements[:10]:
        print(
            f"  {d['id']}: human={d['human_verdict']} judge={d['judge_verdict']} "
            f"(H:{d['human_total']} J:{d['judge_total']}) — {d['judge_summary']}"
        )

    agree = sum(1 for h, j in zip(human_verdicts, judge_verdicts) if h == j)
    print(f"\nVerdict agreement: {agree}/{len(rows)} ({100*agree/len(rows):.0f}%)")


if __name__ == "__main__":
    main()
