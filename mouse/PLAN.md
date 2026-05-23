# Mouse Lifespan ML — Resume Plan

> Read this first if you're picking up after a plan/session reset.

## Where we are right now

✅ **STEP 1 done** — fetched a real lifespan dataset from the Mouse Phenome
Database via the public API. Saved at:

```
mouse/data/mouse_lifespan.csv     (source: Nelson1 project, 790 mice, 44 strains)
mouse/data/yuan2_animaldata.csv   (source: Yuan2 project, 1778 mice, 31 strains — lifespan only)
```

Nelson1 is the better ML dataset because it has real biomarkers, not just
strain identity.

✅ **EDA done** on Yuan2 — see `mouse/results/explore_summary.txt` and the PNGs
in `mouse/results/`. Key finding: strain explains 27% of lifespan variance, sex
explains 0.1%, 71% is unexplained noise that biomarkers could fill.

## What's left

| step | file | status |
|---|---|---|
| 2 — reshape Nelson1 wide→long, impute, save clean ML-ready CSV | `02_prepare.py` | ✅ done |
| 3 — train Linear, RF, XGBoost, evaluate MAE + R² | `03_train.py` | ✅ done |
| 4 — plot pred-vs-actual, residuals, feature importances; save best model | `04_evaluate.py` | ✅ done |

All pipeline steps complete. See Results below.

## Results

- **Best model**: Random Forest — MAE = 173 days, R² = 0.17
- Linear baseline: MAE = 182d, R² = 0.10; XGBoost: MAE = 178d, R² = 0.14
- **Top features**: fat mass (0.33), pct_fat (0.28), body weight (0.14), lean mass (0.10)
- Body composition dominates; strain, sex, and diet contribute < 0.05 each
- Outputs: `models/best_model.pkl`, `results/predictions.csv`, `results/feature_importance.csv`
- Plots: `results/predicted_vs_actual.png`, `results/residuals.png`, `results/feature_importance.png`
- R² = 0.17 reflects high biological noise (71% unexplained variance per EDA)

## Nelson1 dataset structure (critical to understand before step 2)

The CSV is in **wide format**. Each mouse was assigned to **either** ad libitum
(AL) **or** dietary restriction (DR) — never both. That's why:

- ~414 mice have `lifespan_AL` populated (AL group)
- ~376 mice have `lifespan_DR` populated (DR group)
- `bw_AL` / `bw_DR`, `lean_AL` / `lean_DR`, etc. follow the same pattern

Step 2's job is to collapse this into one row per mouse with columns:
`mouse_id, strain, sex, diet, bw, lean, fat, pct_fat, lifespan_days`.

## Constraints from the original spec

- **Train/test split must be by `animal_id`** so the same mouse never appears
  in both sets. Each mouse_id is unique → a simple random row split works.
- **Drop columns with >40% missing values** — but check first, because the
  Nelson1 wide-format nulls are structural (half-AL / half-DR), not missing
  data. After reshape, the long-format columns should have far fewer nulls.
- **Impute remaining nulls with median grouped by strain × sex** (per the
  user's spec). After diet is its own column, group by `strain × sex × diet`.
- **Three models**: Linear (baseline sanity check), Random Forest, XGBoost.
- **Metrics**: MAE in days, R². Print a comparison table.
- **Plots**: predicted vs actual for the best model → `results/predicted_vs_actual.png`.
- **Save**: `models/best_model.pkl`, `results/predictions.csv` with columns
  `mouse_id, strain, sex, predicted_days, actual_days, error`.

## Environment

Already installed: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scipy`,
`scikit-learn` (via requirements.txt), `requests`, `openai`, `datasets`.

Need to install when picking up: `xgboost` (one command: `pip install xgboost`).

## File order to implement / run

```bash
# already done:
python3 mouse/01_discover_via_api.py   # downloads mouse_lifespan.csv

# next, in this order:
python3 mouse/02_prepare.py            # makes mouse_lifespan_long.csv (clean, ML-ready)
python3 mouse/03_train.py              # trains 3 models, prints metrics, saves predictions
python3 mouse/04_evaluate.py           # plots + importance + saves best model
```

## Gotchas

- The `lifespan_AL` and `lifespan_DR` columns are the targets. After reshape
  they become a single `lifespan_days` column.
- The user wants **per-mouse** train/test, NOT per-strain. Don't accidentally
  group-split by strain — that overestimates generalization.
- `pct_fat` is derived from `fat / (lean + fat)` — it's redundant with the
  raw lean/fat columns. Tree models will figure that out, but it can confuse
  the linear baseline. Mention this in `03_train.py` if asked.
- The Yuan2 dataset (1,778 mice, no biomarkers) is still available if anyone
  wants to expand n by joining at strain×sex level — but it's optional; stick
  with Nelson1 for the main ML pipeline.
