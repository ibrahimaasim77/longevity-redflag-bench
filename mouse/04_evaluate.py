"""STEP 4 — Plot predictions, feature importances, save best model.

Refits the best model (from model_comparison.csv) on the same train split
(same seed=42), then saves plots + pickle.

Run with:   python3 mouse/04_evaluate.py
"""

from __future__ import annotations

import pickle
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

ROOT = Path(__file__).parent
IN_PATH = ROOT / "data" / "mouse_lifespan_long.csv"
RESULTS_DIR = ROOT / "results"
MODELS_DIR = ROOT / "models"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
TEST_FRACTION = 0.20
CAT_FEATURES = ["strain", "sex", "diet"]
NUM_FEATURES = ["bw", "lean", "fat", "pct_fat"]
TARGET = "lifespan_days"
ID = "mouse_id"


def main():
    df = pd.read_csv(IN_PATH)
    X = df[CAT_FEATURES + NUM_FEATURES]
    y = df[TARGET]
    ids = df[ID]
    X_tr, X_te, y_tr, y_te, id_tr, id_te = train_test_split(
        X, y, ids, test_size=TEST_FRACTION, random_state=SEED
    )

    cmp = pd.read_csv(RESULTS_DIR / "model_comparison.csv")
    best_name = cmp.sort_values("mae_days").iloc[0]["model"]
    print(f"best model: {best_name}")

    preproc = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CAT_FEATURES),
        ("num", "passthrough", NUM_FEATURES),
    ])

    pipes = {
        "Linear": Pipeline([("pre", preproc), ("model", LinearRegression())]),
        "RandomForest": Pipeline([("pre", preproc), ("model", RandomForestRegressor(
            n_estimators=400, min_samples_leaf=3, random_state=SEED, n_jobs=-1))]),
    }
    if XGBRegressor is not None:
        pipes["XGBoost"] = Pipeline([("pre", preproc), ("model", XGBRegressor(
            n_estimators=500, learning_rate=0.05, max_depth=4,
            random_state=SEED, n_jobs=-1, verbosity=0))]),

    pipe = pipes[best_name]
    pipe.fit(X_tr, y_tr)
    preds = pipe.predict(X_te)
    mae = mean_absolute_error(y_te, preds)
    r2 = r2_score(y_te, preds)
    print(f"refit MAE={mae:.1f} d   R²={r2:.3f}")

    sns.set_theme(style="whitegrid", context="talk")

    # --- predicted vs actual ---
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_te, preds, alpha=0.5, s=20, edgecolors="none")
    lo, hi = min(y_te.min(), preds.min()), max(y_te.max(), preds.max())
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=1)
    ax.set_xlabel("actual lifespan (days)")
    ax.set_ylabel("predicted lifespan (days)")
    ax.set_title(f"{best_name}: predicted vs actual\nMAE={mae:.0f}d  R²={r2:.2f}")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "predicted_vs_actual.png", dpi=140)
    plt.close(fig)
    print(f"saved {RESULTS_DIR/'predicted_vs_actual.png'}")

    # --- residuals ---
    resid = preds - y_te.values
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(preds, resid, alpha=0.5, s=20, edgecolors="none")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlabel("predicted lifespan (days)")
    ax.set_ylabel("residual (predicted - actual)")
    ax.set_title(f"{best_name}: residuals")
    fig.tight_layout()
    fig.savefig(RESULTS_DIR / "residuals.png", dpi=140)
    plt.close(fig)
    print(f"saved {RESULTS_DIR/'residuals.png'}")

    # --- feature importances ---
    est = pipe.named_steps["model"]
    pre = pipe.named_steps["pre"]
    feature_names = pre.get_feature_names_out()
    if hasattr(est, "feature_importances_"):
        imp = pd.DataFrame({
            "feature": feature_names,
            "importance": est.feature_importances_,
        }).sort_values("importance", ascending=False)
        imp.to_csv(RESULTS_DIR / "feature_importance.csv", index=False)

        print(f"\n--- TOP 10 FEATURES ---")
        print(imp.head(10).to_string(index=False, formatters={"importance": "{:.4f}".format}))

        top = imp.head(20).iloc[::-1]
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.barh(top["feature"], top["importance"])
        ax.set_xlabel("importance")
        ax.set_title(f"{best_name}: top 20 features")
        fig.tight_layout()
        fig.savefig(RESULTS_DIR / "feature_importance.png", dpi=140)
        plt.close(fig)
        print(f"saved {RESULTS_DIR/'feature_importance.png'}")
    else:
        print("linear model — no feature_importances_; skipping importance plot")

    # --- save model ---
    with open(MODELS_DIR / "best_model.pkl", "wb") as f:
        pickle.dump(pipe, f)
    print(f"\nsaved {MODELS_DIR/'best_model.pkl'}")

    print(f"\n--- SUMMARY ---")
    print(f"best model      : {best_name}")
    print(f"MAE             : {mae:.1f} days")
    print(f"R²              : {r2:.3f}")
    print(f"predictions.csv : {RESULTS_DIR/'predictions.csv'}")
    print(f"model pickle    : {MODELS_DIR/'best_model.pkl'}")
    print(f"plots           : predicted_vs_actual.png, residuals.png, feature_importance.png")


if __name__ == "__main__":
    main()
