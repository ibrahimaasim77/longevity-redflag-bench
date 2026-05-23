"""STUB — owner: CS teammate. Locked interface; fill the body.

Build the analysis cohort with a CENSORING-CORRECT 10-year mortality outcome.
This is the highest-risk correctness item in the whole project (build-plan.md §4).

Algorithm:
  1. Load NHANES 1999-2008 continuous cycles (demographics + exam + labs + questionnaire)
     and the Linked Mortality File (LMF). Join on SEQN.
  2. Outcome at 10-year horizon (120 months), using MORTSTAT + PERMTH_INT:
        - MORTSTAT==1 (deceased) AND PERMTH_INT <= 120  -> died_10yr = 1
        - PERMTH_INT >= 120 (alive through horizon)      -> died_10yr = 0
        - alive (MORTSTAT==0) AND PERMTH_INT < 120        -> CENSORED -> EXCLUDE (drop row)
     Never code a censored-before-horizon person as "survived".
  3. Keep `cycle` (the covariate-split key) and demographic covariates.

Return a pandas DataFrame with at least:
    seqn, cycle, age, sex, race_eth, <features...>, died_10yr (0/1), permth_int, mortstat
Rows where the outcome is censored/unknown must be dropped, not imputed.
"""

from __future__ import annotations


def build_cohort(cycles=("1999-2000", "2001-2002", "2003-2004", "2005-2006", "2007-2008"),
                 horizon_months: int = 120):
    """-> pandas.DataFrame. See module docstring for the exact contract."""
    raise NotImplementedError("CS teammate: implement censoring-correct 10-yr cohort. See docstring.")
