"""Compare XGBoost vs Longevity-LLM on 5 test-set mice."""

import json
import pickle
import re
import time

import pandas as pd
from openai import OpenAI

DATA_PATH = "data/mouse_ml_dataset.csv"
MODEL_PATH = "mouse/models/yuan_best_model.pkl"  # Yuan3-only XGBoost
PRED_PATH = "mouse/results/yuan_predictions.csv"
OUTPUT_PATH = "results/mouse_comparison.csv"

M18_COLS = [
    "ALT_M18", "LIP_M18", "ALP_M18", "Ca_M18", "Cl_M18", "Fe_M18",
    "Mg_M18", "Phos_M18", "K_M18", "Na_M18", "ALB_M18", "TP_M18",
    "CO2_M18", "TBIL_M18", "HDL_M18", "T4_M18", "BUN_M18",
]
LABEL_MAP = {
    "ALT_M18": "ALT", "LIP_M18": "Lipase", "ALP_M18": "ALP",
    "Ca_M18": "Calcium", "Cl_M18": "Chloride", "Fe_M18": "Iron",
    "Mg_M18": "Magnesium", "Phos_M18": "Phosphorus", "K_M18": "Potassium",
    "Na_M18": "Sodium", "ALB_M18": "Albumin", "TP_M18": "Total Protein",
    "CO2_M18": "CO2", "TBIL_M18": "Total Bilirubin", "HDL_M18": "HDL",
    "T4_M18": "T4", "BUN_M18": "BUN",
}

client = OpenAI(
    base_url="https://swchnq0ekc3scmqw.us-east-2.aws.endpoints.huggingface.cloud/v1",
    api_key="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",
)


def build_prompt(row):
    sex_word = "female" if row["sex"] == "f" else "male"
    biomarkers = []
    for col in M18_COLS:
        val = row.get(col)
        if pd.notna(val):
            biomarkers.append(f"{LABEL_MAP[col]}={val:.1f}")
    bio_str = ", ".join(biomarkers)
    return (
        f"A {sex_word} mouse of strain {row['strain']}. "
        f"At 18 months of age, blood chemistry measurements were: {bio_str}. "
        f"Based on these biomarkers, predict lifespan in days. "
        f'Return JSON only: {{"prediction_days": <number>, "reasoning": "<string>"}}'
    )


def strip_think(text):
    if "</think>" in text:
        text = text.split("</think>")[-1].strip()
    return text


def extract_prediction(text):
    text = strip_think(text)
    text = re.sub(r"^```(?:json)?\s*", "", text.strip())
    text = re.sub(r"\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        try:
            obj = json.loads(text[start:end + 1])
            days = obj.get("prediction_days") or obj.get("predicted_days")
            reasoning = obj.get("reasoning", "")
            if days is not None:
                return int(float(days)), reasoning
        except (json.JSONDecodeError, ValueError):
            pass
    m = re.search(r"(\d{3,4})\s*(?:days)?", text)
    if m:
        return int(m.group(1)), text[:200]
    return None, text[:200]


def main():
    df = pd.read_csv(DATA_PATH)
    preds_df = pd.read_csv(PRED_PATH)

    test_ids = preds_df["animal_id"].tolist()[:5]
    test_rows = df[df["animal_id"].isin(test_ids)]

    xgb_lookup = dict(zip(preds_df["animal_id"], preds_df["predicted_days"]))

    results = []
    for _, row in test_rows.iterrows():
        aid = row["animal_id"]
        actual = int(row["lifespandays"])
        xgb_pred = int(xgb_lookup[aid])
        xgb_err = abs(xgb_pred - actual)

        prompt = build_prompt(row)
        print(f"\n--- {aid} ({row['strain']}, {row['sex']}) ---")
        print(f"  Actual: {actual}d | XGBoost: {xgb_pred}d (err={xgb_err}d)")

        try:
            resp = client.chat.completions.create(
                model="longevity-llm",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1200,
            )
            raw = resp.choices[0].message.content or ""
            llm_days, reasoning = extract_prediction(raw)
            llm_err = abs(llm_days - actual) if llm_days else None
            print(f"  LLM:    {llm_days}d (err={llm_err}d)")
            print(f"  Reason: {reasoning[:100]}")
        except Exception as e:
            llm_days, llm_err, reasoning = None, None, f"ERROR: {e}"
            print(f"  LLM error: {e}")

        results.append({
            "animal_id": aid,
            "strain": row["strain"],
            "sex": row["sex"],
            "actual_days": actual,
            "xgboost_prediction": xgb_pred,
            "xgboost_error": xgb_err,
            "llm_prediction": llm_days,
            "llm_error": llm_err,
            "llm_reasoning": reasoning,
        })
        time.sleep(1)

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    print(f"\n{'='*75}")
    print(f"{'Mouse':<14} {'Actual':>7} {'XGB':>7} {'XGB Err':>8} {'LLM':>7} {'LLM Err':>8}")
    print("-" * 75)
    xgb_errs, llm_errs = [], []
    for _, r in out.iterrows():
        llm_s = str(r["llm_prediction"]) if pd.notna(r["llm_prediction"]) else "N/A"
        llm_e = str(int(r["llm_error"])) if pd.notna(r["llm_error"]) else "N/A"
        print(f"{r['animal_id']:<14} {r['actual_days']:>7} {r['xgboost_prediction']:>7} "
              f"{r['xgboost_error']:>8} {llm_s:>7} {llm_e:>8}")
        xgb_errs.append(r["xgboost_error"])
        if pd.notna(r["llm_error"]):
            llm_errs.append(r["llm_error"])
    print("-" * 75)
    xgb_mae = sum(xgb_errs) / len(xgb_errs)
    print(f"XGBoost MAE: {xgb_mae:.0f} days")
    if llm_errs:
        llm_mae = sum(llm_errs) / len(llm_errs)
        print(f"LLM MAE:     {llm_mae:.0f} days")
        winner = "XGBoost" if xgb_mae < llm_mae else "Longevity-LLM"
        print(f"\nWinner: {winner}")


if __name__ == "__main__":
    main()
