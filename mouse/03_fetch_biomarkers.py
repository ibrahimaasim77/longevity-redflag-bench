"""Pull Yuan1 (IGF-1, body weight) and Yuan3 (blood chemistry) per-animal data.
Stop and show columns + strain overlap with Yuan2 before any joining.

Yuan1 is the same JAX cohort series as Yuan2; biomarkers measured at 6/12/18 mo.
Yuan3 adds 16 blood-chem markers on the same strains.

Together with Yuan2's lifespan they give us ~18-20 candidate features per
strain × sex group to predict when a mouse dies.
"""
from __future__ import annotations

from pathlib import Path
import sys
import requests
import pandas as pd

DATA = Path(__file__).parent / "data"
DATA.mkdir(parents=True, exist_ok=True)

SOURCES = {
    "yuan1_animaldata.csv": "https://phenome.jax.org/projects/Yuan1/csvanimaldata",
    "yuan3_animaldata.csv": "https://phenome.jax.org/projects/Yuan3/csvanimaldata",
}

HEADERS = {"User-Agent": "longevity-redflag-bench/0.1"}


def fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        print(f"  cached: {dest.name} ({dest.stat().st_size:,} bytes)")
        return
    print(f"  fetch:  {url}")
    r = requests.get(url, headers=HEADERS, timeout=120, allow_redirects=True)
    r.raise_for_status()
    dest.write_bytes(r.content)
    print(f"  saved:  {dest.name} ({len(r.content):,} bytes)")


def inspect(path, label):
    print(f"\n>>> {label}  ({path.name})")
    df = pd.read_csv(path, low_memory=False)
    print(f"shape: {df.shape}")
    print(f"columns ({len(df.columns)}):")
    for c in df.columns:
        print(
            f"  - {c}  dtype={df[c].dtype}  n_unique={df[c].nunique()}  n_null={df[c].isna().sum()}"
        )
    print(f"first 3 rows:")
    print(df.head(3).to_string(max_cols=20, max_colwidth=30))
    return df


def main():
    print("=" * 70)
    print("DOWNLOAD")
    print("=" * 70)
    for name, url in SOURCES.items():
        fetch(url, DATA / name)

    print("\n" + "=" * 70)
    print("INSPECT")
    print("=" * 70)
    y1 = inspect(DATA / "yuan1_animaldata.csv", "Yuan1 (IGF-1, body weight)")
    y3 = inspect(DATA / "yuan3_animaldata.csv", "Yuan3 (blood chemistry)")

    print("\n" + "=" * 70)
    print("STRAIN OVERLAP WITH YUAN2 LIFESPAN DATA")
    print("=" * 70)
    y2 = pd.read_csv(DATA / "yuan2_animaldata.csv")
    yuan2_strains = set(y2["strain"].unique())

    for name, df in [("Yuan1", y1), ("Yuan3", y3)]:
        if "strain" in df.columns:
            s = set(df["strain"].unique())
            overlap = yuan2_strains & s
            only_y2 = yuan2_strains - s
            only_other = s - yuan2_strains
            print(f"\n{name}:")
            print(f"  strains in {name}: {len(s)}  |  in Yuan2: {len(yuan2_strains)}  "
                  f"|  shared: {len(overlap)}")
            if only_y2:
                print(f"  in Yuan2 only ({len(only_y2)}): {sorted(only_y2)[:6]}...")
            if only_other:
                print(f"  in {name} only ({len(only_other)}): {sorted(only_other)[:6]}...")
        else:
            print(f"\n{name}: no 'strain' column? cols = {list(df.columns)[:10]}")

    print("\n" + "=" * 70)
    print("STOPPED. Confirm columns + overlap look right before proceeding.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
