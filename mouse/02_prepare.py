"""STEP 2 — Prepare Nelson1 for ML.

Reshape Nelson1 from wide (per-mouse AL/DR split columns) to long
(one row per mouse with a diet column), clean nulls, save the result for
03_train.py.

Run with:   python3 mouse/02_prepare.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent
IN_PATH = ROOT / "data" / "mouse_lifespan.csv"
OUT_PATH = ROOT / "data" / "mouse_lifespan_long.csv"


def main():
    # 1. LOAD
    df = pd.read_csv(IN_PATH)
    print(f"loaded {IN_PATH.name}: shape={df.shape}")
    print(f"columns: {df.columns.tolist()}")

    # 2. RESHAPE WIDE → LONG
    def slice_diet(diet: str) -> pd.DataFrame:
        sub = df[[
            "strain", "sex", "animal_id",
            f"bw_{diet}", f"lean_{diet}", f"fat_{diet}",
            f"pct_fat_{diet}", f"lifespan_{diet}",
        ]].copy()
        sub.columns = ["strain", "sex", "mouse_id",
                       "bw", "lean", "fat", "pct_fat", "lifespan_days"]
        sub["diet"] = diet
        return sub.dropna(subset=["lifespan_days"])

    long = pd.concat([slice_diet("AL"), slice_diet("DR")], ignore_index=True)
    print(f"\nafter wide→long reshape: shape={long.shape}")
    print(f"diet counts: {dict(long['diet'].value_counts())}")

    # 3. DROP COLUMNS WITH > 40% MISSING (post-reshape — should be none)
    miss = long.isna().mean()
    drop_cols = miss[miss > 0.40].index.tolist()
    if drop_cols:
        long = long.drop(columns=drop_cols)
    print(f"\ndropped columns (>40% missing): {drop_cols}")
    print(f"remaining per-column missing %:")
    print((long.isna().mean() * 100).round(1).to_string())

    # 4. IMPUTE BIOMARKER NULLS BY MEDIAN GROUPED BY strain × sex × diet
    biomarker_cols = [c for c in ["bw", "lean", "fat", "pct_fat"] if c in long.columns]
    for c in biomarker_cols:
        long[c] = (
            long.groupby(["strain", "sex", "diet"])[c]
                .transform(lambda s: s.fillna(s.median()))
        )
    # Fallback: any group that had no values at all → overall column median
    for c in biomarker_cols:
        long[c] = long[c].fillna(long[c].median())

    # 5. VALIDATE
    assert long["mouse_id"].is_unique, "mouse_id must be unique"
    assert long["lifespan_days"].notna().all(), "lifespan_days must be fully populated"
    print(f"\n--- VALIDATION ---")
    print(f"shape: {long.shape}")
    print(f"dtypes:\n{long.dtypes.to_string()}")
    print(f"\nnulls per column:\n{long.isna().sum().to_string()}")
    print(f"\nsex counts: {dict(long['sex'].value_counts())}")
    print(f"diet counts: {dict(long['diet'].value_counts())}")
    print(f"strains: {long['strain'].nunique()}  (range of medians: "
          f"{long.groupby('strain')['lifespan_days'].median().min():.0f}–"
          f"{long.groupby('strain')['lifespan_days'].median().max():.0f} days)")

    print(f"\nfirst 5 rows:")
    print(long.head().to_string(index=False))

    # 6. SAVE
    # column order: mouse_id, strain, sex, diet, biomarkers, target
    cols_out = ["mouse_id", "strain", "sex", "diet"] + biomarker_cols + ["lifespan_days"]
    long = long[cols_out]
    long.to_csv(OUT_PATH, index=False)
    print(f"\nwrote {OUT_PATH}  ({OUT_PATH.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
