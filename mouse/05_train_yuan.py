"""Train ML models on Yuan3 blood chemistry (M18) + Yuan2 lifespan."""

import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

DATA_PATH = "data/mouse_ml_dataset.csv"
MODEL_OUT = "mouse/models/yuan_best_model.pkl"
PRED_OUT = "mouse/results/yuan_predictions.csv"

df = pd.read_csv(DATA_PATH)

# Step 1: Keep only M18 columns + identifiers + target
m18_cols = [c for c in df.columns if c.endswith("_M18")]
keep = ["animal_id", "strain", "sex"] + m18_cols + ["lifespandays"]
df = df[keep].copy()
print(f"Step 1: Kept {len(m18_cols)} M18 biomarker columns")
print(f"  Features: {m18_cols}")

# Step 2: Drop rows with no lifespan
before = len(df)
df = df.dropna(subset=["lifespandays"])
print(f"Step 2: {before} -> {len(df)} rows (dropped {before - len(df)} without lifespan)")

# Step 3: Impute missing M18 values — group median, fallback to overall median
overall_medians = df[m18_cols].median()
for col in m18_cols:
    group_med = df.groupby(["strain", "sex"])[col].transform("median")
    df[col] = df[col].fillna(group_med).fillna(overall_medians[col])

remaining_null = df[m18_cols].isna().sum().sum()
print(f"Step 3: Imputed missing values. Remaining nulls: {remaining_null}")

# Step 4: Encode strain and sex
le_strain = LabelEncoder()
le_sex = LabelEncoder()
df["strain_enc"] = le_strain.fit_transform(df["strain"])
df["sex_enc"] = le_sex.fit_transform(df["sex"])
print(f"Step 4: Encoded {df['strain_enc'].nunique()} strains, {df['sex_enc'].nunique()} sexes")

# Step 5: Train/test split by animal_id (80/20)
feature_cols = m18_cols + ["strain_enc", "sex_enc"]
X = df[feature_cols]
y = df["lifespandays"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
train_ids = df.iloc[X_train.index]["animal_id"]
test_ids = df.iloc[X_test.index]["animal_id"]
assert len(set(train_ids) & set(test_ids)) == 0, "Leak detected!"
print(f"Step 5: Train={len(X_train)}, Test={len(X_test)}, no animal overlap")

# Step 6: Train models
try:
    from xgboost import XGBRegressor
    has_xgb = True
except ImportError:
    has_xgb = False
    print("  (xgboost not installed, skipping)")

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42
    ),
}
if has_xgb:
    models["XGBoost"] = XGBRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        min_child_weight=5, random_state=42, verbosity=0,
    )

# Step 7: Evaluate
print(f"\n{'='*55}")
print(f"{'Model':<22} {'MAE (days)':>12} {'R²':>10}")
print("-" * 55)

best_name = None
best_mae = float("inf")
best_model = None

for name, model in models.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    r2 = r2_score(y_test, preds)
    print(f"{name:<22} {mae:>12.1f} {r2:>10.3f}")
    if mae < best_mae:
        best_mae = mae
        best_name = name
        best_model = model

print("-" * 55)
print(f"Best: {best_name} (MAE={best_mae:.1f}d)")

# Step 8: Save best model and predictions
pickle.dump(best_model, open(MODEL_OUT, "wb"))
print(f"\nSaved model to {MODEL_OUT}")

test_preds = best_model.predict(X_test)
test_df = df.iloc[X_test.index][["animal_id", "strain", "sex"]].copy()
test_df["actual_days"] = y_test.values
test_df["predicted_days"] = test_preds.round(0).astype(int)
test_df["error"] = (test_preds - y_test.values).round(0).astype(int)
test_df["model_used"] = best_name
test_df.to_csv(PRED_OUT, index=False)
print(f"Saved predictions to {PRED_OUT}")
