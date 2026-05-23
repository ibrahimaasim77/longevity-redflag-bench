"""Discover mouse lifespan data via the free public MPD API.

No auth, no key. Base URL: https://phenome.jax.org

Steps:
  1. GET /api/projects?csv=yes   → flat list of every project
     Filter by name for: lifespan, survival, aging, longevity, mortality
  2. For each candidate, GET /api/projects/{projsym}/dataset?csv=yes
     Save the first viable one as data/mouse_lifespan.csv
  3. Print columns + first 5 rows, then stop.
"""
from __future__ import annotations

import io
import re
import sys
from pathlib import Path

import pandas as pd
import requests

BASE = "https://phenome.jax.org"
DATA = Path(__file__).parent / "data"
DATA.mkdir(parents=True, exist_ok=True)
OUT = DATA / "mouse_lifespan.csv"
HEADERS = {"User-Agent": "longevity-redflag-bench/0.1"}

PATTERN = re.compile(r"lifespan|survival|longevity|aging|mortality|death", re.IGNORECASE)


def get(url: str) -> requests.Response:
    print(f"  GET {url}")
    r = requests.get(url, headers=HEADERS, timeout=60, allow_redirects=True)
    r.raise_for_status()
    return r


def main():
    # --- STEP 1 — list projects ---
    print("=" * 70)
    print("STEP 1 — discover lifespan/aging projects")
    print("=" * 70)

    r = get(f"{BASE}/api/projects?csv=yes")
    projects = pd.read_csv(io.StringIO(r.text))
    print(f"  loaded {len(projects)} projects")
    print(f"  columns available: {list(projects.columns)[:12]}")

    # Search every text-y column for the lifespan/aging keywords
    text_cols = [c for c in projects.columns if projects[c].dtype == object]
    mask = pd.Series(False, index=projects.index)
    for c in text_cols:
        mask = mask | projects[c].astype(str).str.contains(PATTERN, na=False)

    candidates = projects[mask].copy()
    if "projsym" not in candidates.columns:
        # Some MPD endpoints call it 'project' or 'symbol'; show what's there
        print(f"  WARN: no 'projsym' column. Columns: {list(candidates.columns)}")
        sys.exit(1)

    print(f"\n  matched {len(candidates)} candidate projects (showing top 15):")
    show_cols = [c for c in ("projsym", "name", "description", "largecollab") if c in candidates.columns]
    print(candidates[show_cols].head(15).to_string(index=False, max_colwidth=70))

    # --- STEP 2 — pick the cleanest lifespan candidate and download it ---
    print("\n" + "=" * 70)
    print("STEP 2 — download the per-animal dataset for each candidate")
    print("=" * 70)

    # Prefer projects whose name explicitly contains 'lifespan' or 'survival'
    def rank(row):
        name = " ".join(str(row.get(c) or "") for c in show_cols).lower()
        score = 0
        if "lifespan" in name: score += 4
        if "survival" in name: score += 3
        if "longevity" in name: score += 2
        if "death" in name or "mortality" in name: score += 1
        if "aging" in name: score += 1
        return score

    candidates["rank"] = candidates.apply(rank, axis=1)
    candidates = candidates.sort_values("rank", ascending=False)
    print(f"\n  top-ranked candidates:")
    print(candidates[show_cols + ["rank"]].head(10).to_string(index=False, max_colwidth=70))

    saved = None
    for _, row in candidates.head(10).iterrows():
        projsym = row["projsym"]
        url = f"{BASE}/api/projects/{projsym}/dataset?csv=yes"
        try:
            r = get(url)
        except Exception as e:
            print(f"  {projsym}: failed ({e})")
            continue

        if not r.text.strip() or r.text.strip().startswith("<"):
            print(f"  {projsym}: empty/HTML response, skipping")
            continue

        try:
            df = pd.read_csv(io.StringIO(r.text), low_memory=False)
        except Exception as e:
            print(f"  {projsym}: not parseable as CSV ({e})")
            continue

        # Sniff for a lifespan-like column
        lifespan_col = None
        for c in df.columns:
            cn = c.lower()
            if any(k in cn for k in ("lifespan", "days_survived", "age_at_death",
                                     "lifespandays", "survival", "death_age")):
                lifespan_col = c
                break

        print(f"  {projsym}: shape={df.shape}  lifespan_col={lifespan_col}")
        if lifespan_col is not None and df[lifespan_col].notna().sum() > 50:
            df.to_csv(OUT, index=False)
            print(f"  → saved {OUT}  ({len(df)} rows, lifespan column = {lifespan_col!r})")
            saved = (projsym, lifespan_col, df)
            break

    if saved is None:
        print("\nNo dataset matched the 'has a lifespan column' filter via the dataset endpoint.")
        print("Falling back to the per-project csvanimaldata endpoint for the highest-ranked candidate.")
        top = candidates.iloc[0]["projsym"]
        url = f"{BASE}/projects/{top}/csvanimaldata"
        r = get(url)
        df = pd.read_csv(io.StringIO(r.text), low_memory=False)
        df.to_csv(OUT, index=False)
        saved = (top, None, df)
        print(f"  → saved {OUT}  ({len(df)} rows from {top})")

    # --- STEP 3 — show columns + first 5 rows, stop ---
    print("\n" + "=" * 70)
    print("STEP 3 — inspect the saved dataset")
    print("=" * 70)
    projsym, lifespan_col, df = saved
    print(f"  source project: {projsym}")
    print(f"  saved to:       {OUT}")
    print(f"  shape:          {df.shape}")
    print(f"  lifespan col:   {lifespan_col!r}")
    print(f"\n  columns ({len(df.columns)}):")
    for c in df.columns:
        print(f"    - {c}  dtype={df[c].dtype}  n_unique={df[c].nunique()}  n_null={df[c].isna().sum()}")
    print(f"\n  first 5 rows:")
    print(df.head().to_string(max_cols=20, max_colwidth=40))

    print("\n" + "=" * 70)
    print("STOPPED at step 3 per user instructions.")
    print("=" * 70)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
