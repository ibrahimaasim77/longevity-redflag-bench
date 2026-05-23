"""STEP 1 — fetch a mouse lifespan dataset from MPD and show what we got.

Strategy:
  - Yuan2 is the canonical per-animal lifespan dataset (31 inbred strains, both
    sexes, individual death dates). It's the cleanest survival table on MPD.
  - We also pull `measurements.csv` so we can later identify biomarker measures
    that join to these same mice / strains for ML features.

This script ONLY does step 1 of the user's spec: download, load, print columns,
shape, and first 5 rows. It does not clean, train, or save models.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path

import requests

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Per-animal lifespan rows for Yuan2
YUAN2_URL = "https://phenome.jax.org/projects/Yuan2/csvanimaldata"
# Backup: a static dump of the Yuan2 project dataset hosted by JAX
YUAN2_BACKUP = "https://www.jax.org/phenomedoc?name=MPD_projdatasets/Yuan2.csv"
# Global measure catalog — lets us find biomarker measures to join later
MEASUREMENTS_URL = "https://www.jax.org/phenomedoc?name=MPD_downloads/measurements.csv"

YUAN2_PATH = DATA_DIR / "yuan2_animaldata.csv"
MEASUREMENTS_PATH = DATA_DIR / "measurements.csv"

HEADERS = {"User-Agent": "longevity-redflag-bench/0.1"}


def fetch(url: str, dest: Path, label: str) -> Path:
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest} ({dest.stat().st_size:,} bytes)")
        return dest
    print(f"  fetching {label}: {url}")
    r = requests.get(url, headers=HEADERS, timeout=120, allow_redirects=True)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"  saved:  {dest} ({len(r.content):,} bytes)")
    return dest


def main():
    print("=" * 70)
    print("STEP 1 — DOWNLOAD")
    print("=" * 70)
    try:
        fetch(YUAN2_URL, YUAN2_PATH, "Yuan2 per-animal lifespan")
    except Exception as e:
        print(f"  primary URL failed ({e!s}); trying backup")
        fetch(YUAN2_BACKUP, YUAN2_PATH, "Yuan2 (backup)")
    # measurements.csv is optional — used later to find joinable biomarkers
    try:
        fetch(MEASUREMENTS_URL, MEASUREMENTS_PATH, "MPD measurements catalog")
    except Exception as e:
        print(f"  measurements.csv unavailable ({e!s}); continuing without it")

    print("\n" + "=" * 70)
    print("STEP 1 — LOAD & INSPECT")
    print("=" * 70)

    import pandas as pd

    df = pd.read_csv(YUAN2_PATH)
    print(f"\n>>> yuan2_animaldata.csv")
    print(f"shape: {df.shape}  (rows × cols)")
    print(f"columns ({len(df.columns)}):")
    for c in df.columns:
        print(f"  - {c}  dtype={df[c].dtype}  n_unique={df[c].nunique()}  n_null={df[c].isna().sum()}")
    print(f"\nfirst 5 rows:")
    print(df.head().to_string(max_cols=20, max_colwidth=40))

    # Quick descriptive on lifespan-looking columns
    print(f"\ndtypes summary:")
    print(df.dtypes.value_counts().to_string())

    if MEASUREMENTS_PATH.exists():
        print(f"\n--- measurements.csv (catalog) ---")
        meas = pd.read_csv(MEASUREMENTS_PATH, low_memory=False)
        print(f"shape: {meas.shape}")
        print(f"columns: {list(meas.columns)}")
        if "varname" in meas.columns:
            cands = meas[meas["varname"].astype(str).str.contains("lifespan|survival|death", case=False, na=False)]
            print(f"\nlifespan/survival/death-tagged measures in catalog: {len(cands)}")
            if len(cands):
                cols_to_show = [c for c in ("measnum", "projsym", "varname", "description", "units") if c in cands.columns]
                print(cands[cols_to_show].head(10).to_string(index=False, max_colwidth=40))
    else:
        print(f"\n(no measurements.csv available — skipped catalog scan)")

    print("\n" + "=" * 70)
    print("STOPPED at step 1 per user instructions.")
    print("Confirm columns look right before proceeding to clean/train.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
