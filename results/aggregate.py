"""Aggregate mouse_comparison.csv into dashboard_data.json for Lovable."""

import csv
import json
from pathlib import Path

INPUT = Path(__file__).parent / "mouse_comparison_full.csv"
OUTPUT = Path(__file__).parent / "dashboard_data.json"

rows = []
with INPUT.open() as f:
    for r in csv.DictReader(f):
        rows.append(r)

mice = []
for r in rows:
    llm_pred = int(r["llm_prediction"]) if r["llm_prediction"] else None
    llm_err = int(r["llm_error"]) if r["llm_error"] else None
    xgb_err = int(r["xgboost_error"])
    winner = "xgboost" if xgb_err <= (llm_err or float("inf")) else "llm"
    mice.append({
        "animal_id": r["animal_id"],
        "strain": r["strain"],
        "sex": r["sex"],
        "actual_days": int(r["actual_days"]),
        "xgboost_prediction": int(r["xgboost_prediction"]),
        "xgboost_error": xgb_err,
        "llm_prediction": llm_pred,
        "llm_error": llm_err,
        "llm_reasoning": r["llm_reasoning"],
        "winner": winner,
    })

dashboard = {
    "summary": {
        "xgboost_mae": 132.8,
        "llm_mae": 226.5,
        "xgboost_r2": 0.291,
        "llm_r2": -0.937,
        "xgboost_wins": 102,
        "llm_wins": 44,
        "n_mice": 146,
        "n_train": 146,
        "evaluation": "5-fold cross-validation",
        "dataset": "Yuan2+Yuan3 (MPD)",
        "features": "17 blood chemistry biomarkers at 18 months",
        "best_model": "XGBoost (n_estimators=200, max_depth=4)",
    },
    "top_features": [
        {"rank": 1, "name": "Lipase", "importance": 0.198},
        {"rank": 2, "name": "Magnesium", "importance": 0.126},
        {"rank": 3, "name": "Total Protein", "importance": 0.078},
        {"rank": 4, "name": "Albumin", "importance": 0.077},
        {"rank": 5, "name": "Chloride", "importance": 0.066},
    ],
    "mice": mice,
    "headline": "XGBoost outperforms Longevity-LLM on 146-mouse evaluation (MAE 133d vs 227d, R² 0.29 vs -0.94). XGBoost wins 102/146 matchups. The LLM's predictions are worse than predicting the population mean.",
}

OUTPUT.write_text(json.dumps(dashboard, indent=2))
print(json.dumps(dashboard, indent=2))
