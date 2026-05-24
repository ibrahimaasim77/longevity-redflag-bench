"""Full comparison: XGBoost (5-fold CV) vs Longevity-LLM on all 146 Yuan3 mice."""

import json
import pickle
import re
import time
import warnings

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

DATA_PATH = "data/mouse_ml_dataset.csv"
OUTPUT_PATH = "results/mouse_comparison_full.csv"

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
    df = pd.read_csv(DATA_PATH).dropna(subset=["lifespandays"])
    print(f"Dataset: {len(df)} mice")

    # Impute
    overall_medians = df[M18_COLS].median()
    for col in M18_COLS:
        group_med = df.groupby(["strain", "sex"])[col].transform("median")
        df[col] = df[col].fillna(group_med).fillna(overall_medians[col])

    # Encode
    le_strain = LabelEncoder()
    le_sex = LabelEncoder()
    df["strain_enc"] = le_strain.fit_transform(df["strain"])
    df["sex_enc"] = le_sex.fit_transform(df["sex"])

    feature_cols = M18_COLS + ["strain_enc", "sex_enc"]
    X = df[feature_cols].values
    y = df["lifespandays"].values

    # 5-fold CV for XGBoost predictions on all mice
    xgb_preds = np.zeros(len(df))
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        model = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            min_child_weight=5, random_state=42, verbosity=0,
        )
        model.fit(X[train_idx], y[train_idx])
        xgb_preds[test_idx] = model.predict(X[test_idx])

    xgb_mae = mean_absolute_error(y, xgb_preds)
    xgb_r2 = r2_score(y, xgb_preds)
    print(f"XGBoost 5-fold CV: MAE={xgb_mae:.1f}d, R²={xgb_r2:.3f}")

    # LLM predictions
    results = []
    errors = 0
    for i, (idx, row) in enumerate(df.iterrows()):
        xgb_pred = int(round(xgb_preds[i]))
        xgb_err = abs(xgb_pred - int(row["lifespandays"]))

        prompt = build_prompt(row)
        try:
            resp = client.chat.completions.create(
                model="longevity-llm",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=1200,
            )
            raw = resp.choices[0].message.content or ""
            llm_days, reasoning = extract_prediction(raw)
            llm_err = abs(llm_days - int(row["lifespandays"])) if llm_days else None
        except Exception as e:
            llm_days, llm_err, reasoning = None, None, f"ERROR: {e}"
            errors += 1

        results.append({
            "animal_id": row["animal_id"],
            "strain": row["strain"],
            "sex": row["sex"],
            "actual_days": int(row["lifespandays"]),
            "xgboost_prediction": xgb_pred,
            "xgboost_error": xgb_err,
            "llm_prediction": llm_days,
            "llm_error": llm_err,
            "llm_reasoning": reasoning,
        })

        if (i + 1) % 10 == 0:
            done_llm = sum(1 for r in results if r["llm_prediction"] is not None)
            print(f"  {i+1}/146 done ({done_llm} LLM successes, {errors} errors)")

        time.sleep(0.5)

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(out)} rows to {OUTPUT_PATH}")

    # Summary
    valid = out.dropna(subset=["llm_prediction"])
    llm_mae = valid["llm_error"].mean()
    llm_r2 = r2_score(valid["actual_days"], valid["llm_prediction"])
    xgb_valid_mae = valid["xgboost_error"].mean()

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS ({len(valid)} mice with both predictions)")
    print(f"{'='*60}")
    print(f"{'':>20} {'MAE (days)':>12} {'R²':>10}")
    print(f"-"*45)
    print(f"{'XGBoost (5-fold CV)':<20} {xgb_valid_mae:>12.1f} {xgb_r2:>10.3f}")
    print(f"{'Longevity-LLM':<20} {llm_mae:>12.1f} {llm_r2:>10.3f}")
    print(f"-"*45)

    xgb_wins = (valid["xgboost_error"] < valid["llm_error"]).sum()
    llm_wins = (valid["llm_error"] < valid["xgboost_error"]).sum()
    ties = (valid["xgboost_error"] == valid["llm_error"]).sum()
    print(f"XGBoost wins: {xgb_wins} | LLM wins: {llm_wins} | Ties: {ties}")
    print(f"LLM errors: {errors}")


if __name__ == "__main__":
    main()
