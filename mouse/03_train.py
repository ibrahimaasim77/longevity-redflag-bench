"""STEP 3 — Train Linear / RandomForest / XGBoost, evaluate MAE + R².

Per-mouse 80/20 split (random_state=42). Outputs:
  - results/predictions.csv      (test-set predictions from the best model)
  - results/model_comparison.csv (MAE, R² for each model)

Run with:   python3 mouse/03_train.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
except Exception as e:
    print(f"XGBoost unavailable: {e}")
    XGBRegressor = None

ROOT = Path(__file__).parent
IN_PATH = ROOT / "data" / "mouse_lifespan_long.csv"
RESULTS_DIR = ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_FRACTION = 0.20
CAT_FEATURES = ["strain", "sex", "diet"]
NUM_FEATURES = ["bw", "lean", "fat", "pct_fat"]
TARGET = "lifespan_days"
ID = "mouse_id"


def main():
    # 1. LOAD
    df = pd.read_csv(IN_PATH)
    assert df[ID].is_unique, "mouse_id must be unique"
    print(f"loaded {IN_PATH.name}: shape={df.shape}")

    # 2. TRAIN/TEST SPLIT BY MOUSE (each row is one unique mouse → row split = mouse split)
    X = df[CAT_FEATURES + NUM_FEATURES]
    y = df[TARGET]
    ids = df[ID]
    X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
        X, y, ids, test_size=TEST_FRACTION, random_state=SEED
    )
    print(f"train: n={len(X_tr)}   test: n={len(X_te)}")

    # 3. PREPROCESSOR (one-hot for cats, passthrough for nums)
    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])

    # 4. THREE MODELS
    models = {
        "Linear": Pipeline([
            ("pre", preproc),
            ("model", LinearRegression()),
        ]),
        "RandomForest": Pipeline([
            ("pre", preproc),
            ("model", RandomForestRegressor(
                n_estimators=400, min_samples_leaf=3,
                random_state=SEED, n_jobs=-1,
            )),
        ]),
    }
    if XGBRegressor is not None:
        models["XGBoost"] = Pipeline([
            ("pre", preproc),
            ("model", XGBRegressor(
                n_estimators=500, learning_rate=0.05, max_depth=4,
                random_state=SEED, n_jobs=-1, verbosity=0,
            )),
        ])

    # 5. FIT EACH, COLLECT METRICS
    print(f"\n--- fitting {len(models)} models ---")
    results = []
    all_preds = {}
    for name, pipe in models.items():
        pipe.fit(X_tr, y_tr)
        p = pipe.predict(X_te)
        mae = mean_absolute_error(y_te, p)
        r2 = r2_score(y_te, p)
        results.append({"model": name, "mae_days": mae, "r2": r2})
        all_preds[name] = p
        print(f"  {name:14s}  MAE={mae:7.1f} days   R²={r2:+.3f}")

    results_df = pd.DataFrame(results).sort_values("mae_days").reset_index(drop=True)

    print(f"\n--- MODEL COMPARISON (sorted by MAE, lower is better) ---")
    print(results_df.to_string(index=False, formatters={
        "mae_days": "{:.1f}".format, "r2": "{:+.3f}".format,
    }))

    # 6. SAVE PREDICTIONS FROM BEST MODEL
    best_name = results_df.iloc[0]["model"]
    preds_df = pd.DataFrame({
        "mouse_id": id_te.values,
        "strain": X_te["strain"].values,
        "sex": X_te["sex"].values,
        "diet": X_te["diet"].values,
        "predicted_days": all_preds[best_name],
        "actual_days": y_te.values,
        "error": all_preds[best_name] - y_te.values,
    })
    preds_df.to_csv(RESULTS_DIR / "predictions.csv", index=False)
    results_df.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)
    print(f"\nwrote {RESULTS_DIR/'predictions.csv'}   (best model: {best_name})")
    print(f"wrote {RESULTS_DIR/'model_comparison.csv'}")

    # quick sanity stats on residuals
    err = preds_df["error"]
    print(f"\nResiduals on test set (best model = {best_name}):")
    print(f"  median error: {err.median():+.1f} d")
    print(f"  std error   : {err.std():.1f} d")
    print(f"  worst over  : {err.max():+.0f} d  (predicted too long)")
    print(f"  worst under : {err.min():+.0f} d  (predicted too short)")


if __name__ == "__main__":
    main()
