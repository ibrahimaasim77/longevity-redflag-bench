"""STEP 2 — Explore characteristics vs. lifespan in the Yuan2 dataset.

We have 1,778 mice with three characteristics:
  - strain (31 inbred lines, a heritable genetic background)
  - sex (m/f)
  - animal_id (unique, used to confirm independence)

This script answers: which of these characteristics actually predict when a
mouse dies, and by how much?

Outputs (saved to mouse/results/):
  - explore_summary.txt        — text report of all key numbers
  - per_strain_summary.csv     — n, median, IQR, mean±SD per strain×sex
  - hist_overall.png           — overall lifespan distribution
  - box_by_sex.png             — sex effect, overall
  - box_by_strain.png          — strain ranking (sorted)
  - heatmap_strain_sex.png     — strain × sex median lifespan
  - sex_diff_by_strain.png     — within-strain male-vs-female delta
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # no display needed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "yuan2_animaldata.csv"
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)

if not DATA.exists():
    sys.exit(f"missing {DATA}; run 01_fetch.py first")

df = pd.read_csv(DATA)
# Drop the empty strain-median column
df = df.drop(columns=["lifespan_median"], errors="ignore")
df["lifespan_years"] = df["lifespandays"] / 365.25

report_lines = []


def log(msg=""):
    print(msg)
    report_lines.append(str(msg))


log("=" * 70)
log(f"DATA: {DATA.name}  shape={df.shape}")
log("=" * 70)
log(f"strains    : {df['strain'].nunique()}")
log(f"sexes      : {sorted(df['sex'].unique())}")
log(f"mice total : {df['animal_id'].nunique()} unique  ({len(df)} rows)")
log(f"lifespan   : min={df['lifespandays'].min()}d  "
    f"median={df['lifespandays'].median():.0f}d  "
    f"max={df['lifespandays'].max()}d  "
    f"({df['lifespandays'].max()/365.25:.1f} years)")
log("")

# ---------------------------------------------------------------------------
# 1. Per-group summary
# ---------------------------------------------------------------------------
log("=" * 70)
log("PER-STRAIN × SEX SUMMARY (sorted by median lifespan, descending)")
log("=" * 70)

g = df.groupby(["strain", "sex"], observed=True)["lifespandays"]
summary = (
    g.agg(n="count", median="median", mean="mean", std="std",
          q25=lambda s: s.quantile(0.25), q75=lambda s: s.quantile(0.75))
    .reset_index()
    .sort_values("median", ascending=False)
)
summary["iqr"] = summary["q75"] - summary["q25"]
summary.to_csv(OUT / "per_strain_summary.csv", index=False)

log("Top 10 longest-lived strain×sex groups:")
log(summary.head(10).to_string(index=False, formatters={
    "median": "{:.0f}".format, "mean": "{:.0f}".format,
    "std": "{:.0f}".format, "q25": "{:.0f}".format, "q75": "{:.0f}".format,
    "iqr": "{:.0f}".format,
}))
log("")
log("Bottom 10 shortest-lived strain×sex groups:")
log(summary.tail(10).to_string(index=False, formatters={
    "median": "{:.0f}".format, "mean": "{:.0f}".format,
    "std": "{:.0f}".format, "q25": "{:.0f}".format, "q75": "{:.0f}".format,
    "iqr": "{:.0f}".format,
}))
log("")

# ---------------------------------------------------------------------------
# 2. Sex effect (overall)
# ---------------------------------------------------------------------------
log("=" * 70)
log("SEX EFFECT (pooled across all strains)")
log("=" * 70)

male = df.loc[df["sex"] == "m", "lifespandays"]
female = df.loc[df["sex"] == "f", "lifespandays"]
log(f"  n_male   = {len(male):4d}  median = {male.median():.0f}d  mean = {male.mean():.1f}d")
log(f"  n_female = {len(female):4d}  median = {female.median():.0f}d  mean = {female.mean():.1f}d")

t_stat, t_p = stats.ttest_ind(male, female, equal_var=False)
u_stat, u_p = stats.mannwhitneyu(male, female, alternative="two-sided")
log(f"  Welch t-test (means)  : t={t_stat:.3f}  p={t_p:.2e}")
log(f"  Mann-Whitney U (ranks): U={u_stat:.0f}  p={u_p:.2e}")
log(f"  Median delta (m-f)    : {male.median() - female.median():+.0f} days")
log("")

# ---------------------------------------------------------------------------
# 3. Strain effect (overall)
# ---------------------------------------------------------------------------
log("=" * 70)
log("STRAIN EFFECT (across all 31 strains, sex-pooled)")
log("=" * 70)

groups = [grp["lifespandays"].values for _, grp in df.groupby("strain", observed=True)]
kw_stat, kw_p = stats.kruskal(*groups)
log(f"  Kruskal-Wallis: H={kw_stat:.1f}  p={kw_p:.2e}  df={len(groups)-1}")

strain_medians = df.groupby("strain", observed=True)["lifespandays"].median().sort_values()
log(f"  longest-lived strain   : {strain_medians.index[-1]}  ({strain_medians.iloc[-1]:.0f}d)")
log(f"  shortest-lived strain  : {strain_medians.index[0]}  ({strain_medians.iloc[0]:.0f}d)")
log(f"  spread (max - min med.): {strain_medians.iloc[-1] - strain_medians.iloc[0]:.0f}d "
    f"({(strain_medians.iloc[-1] - strain_medians.iloc[0]) / 365.25:.1f} years)")
log("")

# ---------------------------------------------------------------------------
# 4. Variance partition: how much of lifespan variance is explained by
#    strain alone, sex alone, and strain×sex interaction?
# ---------------------------------------------------------------------------
log("=" * 70)
log("VARIANCE PARTITION (η² from one-way and two-way ANOVA-style decomposition)")
log("=" * 70)


def eta_sq(by_cols: list[str]) -> float:
    """Simple proportion of variance explained by group means."""
    grand_mean = df["lifespandays"].mean()
    ss_total = ((df["lifespandays"] - grand_mean) ** 2).sum()
    group_means = df.groupby(by_cols, observed=True)["lifespandays"].transform("mean")
    ss_between = ((group_means - grand_mean) ** 2).sum()
    return ss_between / ss_total


eta_strain = eta_sq(["strain"])
eta_sex = eta_sq(["sex"])
eta_both = eta_sq(["strain", "sex"])
log(f"  strain only        : η² = {eta_strain:.3f}  ({eta_strain*100:.1f}% of variance)")
log(f"  sex only           : η² = {eta_sex:.3f}  ({eta_sex*100:.1f}% of variance)")
log(f"  strain + sex combo : η² = {eta_both:.3f}  ({eta_both*100:.1f}% of variance)")
log(f"  unexplained        : {(1-eta_both)*100:.1f}% — within-strain×sex biological/environmental noise")
log("")

# ---------------------------------------------------------------------------
# 5. Per-strain sex effect: where is the sex gap largest?
# ---------------------------------------------------------------------------
log("=" * 70)
log("WITHIN-STRAIN SEX EFFECT (median male − median female), top 5 each direction")
log("=" * 70)

sex_pivot = df.pivot_table(
    index="strain", columns="sex", values="lifespandays", aggfunc="median"
)
sex_pivot["delta_m_minus_f"] = sex_pivot.get("m") - sex_pivot.get("f")
delta_sorted = sex_pivot.dropna(subset=["delta_m_minus_f"]).sort_values("delta_m_minus_f")
log("Strains where MALES outlive females the most:")
log(delta_sorted.tail(5).to_string(formatters={c: "{:.0f}".format for c in delta_sorted.columns}))
log("")
log("Strains where FEMALES outlive males the most:")
log(delta_sorted.head(5).to_string(formatters={c: "{:.0f}".format for c in delta_sorted.columns}))
log("")

# ---------------------------------------------------------------------------
# Save the text report
# ---------------------------------------------------------------------------
(OUT / "explore_summary.txt").write_text("\n".join(report_lines))
print(f"\nSaved text report: {OUT/'explore_summary.txt'}")

# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
sns.set_theme(style="whitegrid", context="talk")

# Histogram + KDE of overall lifespan, split by sex
fig, ax = plt.subplots(figsize=(10, 5))
sns.histplot(data=df, x="lifespandays", hue="sex", bins=40, kde=True, ax=ax)
ax.set_xlabel("lifespan (days)")
ax.set_title(f"Yuan2 lifespan distribution (n={len(df)} mice, 31 strains)")
fig.tight_layout()
fig.savefig(OUT / "hist_overall.png", dpi=140)
plt.close(fig)

# Box plot by sex
fig, ax = plt.subplots(figsize=(6, 5))
sns.boxplot(data=df, x="sex", y="lifespandays", hue="sex", legend=False, ax=ax, palette="Set2")
sns.stripplot(data=df, x="sex", y="lifespandays", color="black", alpha=0.15, size=2, ax=ax)
ax.set_title(f"Lifespan by sex  (p_t={t_p:.1e})")
ax.set_ylabel("lifespan (days)")
fig.tight_layout()
fig.savefig(OUT / "box_by_sex.png", dpi=140)
plt.close(fig)

# Box plot by strain (sorted by median)
order = df.groupby("strain", observed=True)["lifespandays"].median().sort_values().index
fig, ax = plt.subplots(figsize=(13, 7))
sns.boxplot(data=df, x="strain", y="lifespandays", order=order, ax=ax,
            color="lightsteelblue", fliersize=3)
ax.tick_params(axis="x", rotation=75, labelsize=8)
ax.set_title(f"Lifespan by strain (Kruskal-Wallis p={kw_p:.1e})")
ax.set_ylabel("lifespan (days)")
fig.tight_layout()
fig.savefig(OUT / "box_by_strain.png", dpi=140)
plt.close(fig)

# Heatmap: strain × sex median
pivot = df.pivot_table(index="strain", columns="sex", values="lifespandays", aggfunc="median")
pivot = pivot.loc[order]  # same order as box plot
fig, ax = plt.subplots(figsize=(5, 9))
sns.heatmap(pivot, annot=True, fmt=".0f", cmap="viridis", ax=ax,
            cbar_kws={"label": "median lifespan (days)"})
ax.set_title("Median lifespan by strain × sex")
ax.tick_params(axis="y", labelsize=8)
fig.tight_layout()
fig.savefig(OUT / "heatmap_strain_sex.png", dpi=140)
plt.close(fig)

# Sex-difference by strain — sorted bar
delta = sex_pivot.dropna(subset=["delta_m_minus_f"]).sort_values("delta_m_minus_f")
fig, ax = plt.subplots(figsize=(13, 6))
colors = ["#d62728" if v > 0 else "#1f77b4" for v in delta["delta_m_minus_f"]]
ax.bar(delta.index, delta["delta_m_minus_f"], color=colors)
ax.axhline(0, color="black", linewidth=0.8)
ax.tick_params(axis="x", rotation=75, labelsize=8)
ax.set_ylabel("median(male) − median(female), days")
ax.set_title("Within-strain sex effect (red = males live longer; blue = females live longer)")
fig.tight_layout()
fig.savefig(OUT / "sex_diff_by_strain.png", dpi=140)
plt.close(fig)

print(f"Saved 5 plots to {OUT}/")
print("\nDone. Open mouse/results/explore_summary.txt for the full numeric report.")
