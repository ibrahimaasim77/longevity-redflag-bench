"""Compare Longevity-LLM vs RF model predictions on 5 MPD mice."""

import json
import pickle
import re
import time

import pandas as pd
import requests

DATA_PATH = "mouse/data/mouse_lifespan_long.csv"
MODEL_PATH = "mouse/models/best_model.pkl"
OUTPUT_PATH = "results/mouse_comparison.csv"

LLM_URL = "https://sqrq2pj09htgequ0.us-east-2.aws.endpoints.huggingface.cloud/v1/chat/completions"

PROMPT_TEMPLATE = """You are a mouse lifespan prediction model. Given the following biomarker data for a laboratory mouse from the Mouse Phenome Database (Nelson1 study), predict the mouse's lifespan in days and explain your reasoning.

Mouse biomarkers:
- Strain: {strain}
- Sex: {sex}
- Diet: {diet} (AL = ad libitum, DR = dietary restriction)
- Body weight: {bw:.1f} g
- Lean mass: {lean:.1f} g
- Fat mass: {fat:.2f} g
- Body fat percentage: {pct_fat:.1f}%

Respond with:
1. Your predicted lifespan in days (as a single number on the first line, e.g. "Predicted lifespan: 850 days")
2. A brief explanation of your reasoning (2-3 sentences)"""


def select_mice(df, n=5):
    df = df.dropna(subset=["lifespan_days"]).sort_values("lifespan_days")
    indices = [0, len(df) // 4, len(df) // 2, 3 * len(df) // 4, len(df) - 1]
    return df.iloc[indices].reset_index(drop=True)


def query_llm(mouse_row):
    prompt = PROMPT_TEMPLATE.format(**mouse_row)
    payload = {
        "model": "tgi",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 300,
        "temperature": 0.1,
    }
    resp = requests.post(LLM_URL, json=payload, timeout=60)
    resp.raise_for_status()
    content = resp.json()["choices"][0]["message"]["content"]
    return content


def extract_days(text):
    patterns = [
        r"[Pp]redicted\s+lifespan[:\s]+(\d+)",
        r"(\d{3,4})\s*days",
        r"(\d{3,4})",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return int(m.group(1))
    return None


def main():
    df = pd.read_csv(DATA_PATH)
    model = pickle.load(open(MODEL_PATH, "rb"))
    sample = select_mice(df)

    features = ["strain", "sex", "diet", "bw", "lean", "fat", "pct_fat"]
    ml_preds = model.predict(sample[features])

    results = []
    for i, (_, row) in enumerate(sample.iterrows()):
        print(f"\n--- Mouse {i+1}/5: {row['mouse_id']} ({row['strain']}) ---")
        print(f"  Actual: {row['lifespan_days']:.0f}d | ML pred: {ml_preds[i]:.0f}d")

        try:
            llm_response = query_llm(row)
            llm_days = extract_days(llm_response)
            reasoning = llm_response.strip().replace("\n", " | ")
            print(f"  LLM pred: {llm_days}d")
            print(f"  LLM reasoning: {reasoning[:120]}...")
        except Exception as e:
            llm_days = None
            reasoning = f"ERROR: {e}"
            print(f"  LLM error: {e}")

        results.append(
            {
                "mouse_id": row["mouse_id"],
                "strain": row["strain"],
                "actual_days": int(row["lifespan_days"]),
                "ml_prediction": int(round(ml_preds[i])),
                "llm_prediction": llm_days,
                "ml_error": int(round(abs(ml_preds[i] - row["lifespan_days"]))),
                "llm_error": (
                    abs(llm_days - row["lifespan_days"]) if llm_days else None
                ),
                "llm_reasoning": reasoning,
            }
        )
        time.sleep(1)

    out = pd.DataFrame(results)
    out.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")

    print(f"\n{'='*70}")
    print(f"{'Mouse ID':<15} {'Actual':>7} {'ML':>7} {'LLM':>7} {'ML Err':>7} {'LLM Err':>8}")
    print("-" * 70)
    ml_errors = []
    llm_errors = []
    for _, r in out.iterrows():
        llm_str = str(r["llm_prediction"]) if pd.notna(r["llm_prediction"]) else "N/A"
        llm_err = str(int(r["llm_error"])) if pd.notna(r["llm_error"]) else "N/A"
        print(
            f"{r['mouse_id']:<15} {r['actual_days']:>7} {r['ml_prediction']:>7} "
            f"{llm_str:>7} {r['ml_error']:>7} {llm_err:>8}"
        )
        ml_errors.append(r["ml_error"])
        if pd.notna(r["llm_error"]):
            llm_errors.append(r["llm_error"])

    print("-" * 70)
    ml_mae = sum(ml_errors) / len(ml_errors)
    print(f"{'ML MAE':<15} {ml_mae:>39.0f}")
    if llm_errors:
        llm_mae = sum(llm_errors) / len(llm_errors)
        print(f"{'LLM MAE':<15} {llm_mae:>48.0f}")
        winner = "ML (Random Forest)" if ml_mae < llm_mae else "Longevity-LLM"
        print(f"\nWinner: {winner} (lower MAE)")


if __name__ == "__main__":
    main()
