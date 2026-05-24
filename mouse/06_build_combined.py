"""Build maximum training dataset from Yuan3+Yuan2 and Jaxpheno4+Yuan2."""

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

# --- Part A: Yuan3 + Yuan2 on animal_id (146 mice, M18 blood chemistry) ---
yuan2 = pd.read_csv("data/Yuan2.csv")
yuan3 = pd.read_csv("data/Yuan3.csv")

m18_cols = [c for c in yuan3.columns if c.endswith("_M18")]
yuan3_sub = yuan3[["animal_id", "strain", "sex"] + m18_cols].copy()
merged_a = yuan3_sub.merge(yuan2[["animal_id", "lifespandays"]], on="animal_id", how="inner")
merged_a["source"] = "yuan3_animal"
print(f"Part A (Yuan3+Yuan2 animal_id): {len(merged_a)} mice, {len(m18_cols)} biomarkers")

# --- Part B: Jaxpheno4 + Yuan2 on strain+sex (CBC medians) ---
jax = pd.read_csv("data/Jaxpheno4.csv")

# Use 16-week CBC columns (closer to adulthood)
cbc16_cols = [c for c in jax.columns if c.endswith("16")]
cbc16_rename = {c: c.replace("16", "_cbc") for c in cbc16_cols}

# Compute strain+sex median CBC from Jaxpheno4
jax_medians = jax.groupby(["strain", "sex"])[cbc16_cols].median().reset_index()
jax_medians = jax_medians.rename(columns=cbc16_rename)
cbc_feature_cols = list(cbc16_rename.values())

# Join to Yuan2 mice by strain+sex
merged_b = yuan2.merge(jax_medians, on=["strain", "sex"], how="inner")
merged_b = merged_b.rename(columns={"lifespandays": "lifespandays"})
merged_b["source"] = "jaxpheno4_strainsex"
print(f"Part B (Jaxpheno4+Yuan2 strain+sex): {len(merged_b)} mice, {len(cbc_feature_cols)} CBC features")

# --- Combine: align columns, deduplicate ---
# Shared columns: strain, sex, lifespandays, source
# Feature columns differ (M18 chemistry vs CBC) — union with NaN fill

all_feature_cols = sorted(set(m18_cols) | set(cbc_feature_cols))

for col in all_feature_cols:
    if col not in merged_a.columns:
        merged_a[col] = np.nan
    if col not in merged_b.columns:
        merged_b[col] = np.nan

keep_cols = ["strain", "sex", "lifespandays", "source"] + all_feature_cols
# Add animal_id where available
merged_a["mouse_key"] = merged_a["animal_id"]
merged_b["mouse_key"] = merged_b["animal_id"]

combined = pd.concat([
    merged_a[["mouse_key", "strain", "sex", "lifespandays", "source"] + all_feature_cols],
    merged_b[["mouse_key", "strain", "sex", "lifespandays", "source"] + all_feature_cols],
], ignore_index=True)

# Deduplicate: if same animal_id appears in both, keep yuan3 (richer features)
before = len(combined)
combined = combined.drop_duplicates(subset=["mouse_key"], keep="first")
combined = combined.dropna(subset=["lifespandays"])
print(f"\nCombined: {before} -> {len(combined)} after dedup")
print(f"  From Yuan3 animal join: {(combined['source']=='yuan3_animal').sum()}")
print(f"  From Jaxpheno4 strain+sex join: {(combined['source']=='jaxpheno4_strainsex').sum()}")
print(f"  Total feature columns: {len(all_feature_cols)}")

# --- Fill missing: per-source median imputation ---
for col in all_feature_cols:
    group_med = combined.groupby(["strain", "sex"])[col].transform("median")
    overall_med = combined[col].median()
    combined[col] = combined[col].fillna(group_med).fillna(overall_med if pd.notna(overall_med) else 0)

remaining_null = combined[all_feature_cols].isna().sum().sum()
print(f"  Remaining nulls after imputation: {remaining_null}")

# --- Encode categoricals ---
le_strain = LabelEncoder()
le_sex = LabelEncoder()
combined["strain_enc"] = le_strain.fit_transform(combined["strain"])
combined["sex_enc"] = le_sex.fit_transform(combined["sex"])

# --- Train/test split ---
feature_cols = all_feature_cols + ["strain_enc", "sex_enc"]
X = combined[feature_cols]
y = combined["lifespandays"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTrain: {len(X_train)}, Test: {len(X_test)}")

# --- Train XGBoost ---
try:
    from xgboost import XGBRegressor
    has_xgb = True
except ImportError:
    has_xgb = False

models = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(
        n_estimators=200, max_depth=6, min_samples_leaf=5, random_state=42
    ),
}
if has_xgb:
    models["XGBoost"] = XGBRegressor(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        min_child_weight=5, random_state=42, verbosity=0,
    )

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

# --- Save ---
pickle.dump(best_model, open("mouse/models/combined_best_model.pkl", "wb"))
print(f"\nSaved model to mouse/models/combined_best_model.pkl")

# Feature importance
if hasattr(best_model, "feature_importances_"):
    imp = sorted(zip(feature_cols, best_model.feature_importances_), key=lambda x: -x[1])
    print(f"\nTop 10 features:")
    for i, (f, v) in enumerate(imp[:10], 1):
        print(f"  {i}. {f:<20} {v:.3f}")

# Save predictions
test_df = combined.loc[X_test.index, ["mouse_key", "strain", "sex", "source"]].copy()
test_df["actual_days"] = y_test.values
test_df["predicted_days"] = best_model.predict(X_test).round(0).astype(int)
test_df["error"] = abs(test_df["predicted_days"] - test_df["actual_days"])
test_df["model_used"] = best_name
test_df.to_csv("mouse/results/combined_predictions.csv", index=False)
print(f"Saved predictions to mouse/results/combined_predictions.csv")
