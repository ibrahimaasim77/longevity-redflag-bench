"""STUB — owner: CS teammate. Baselines are MANDATORY for Statistical Rigor
(build-plan.md §4) — a benchmark with no reported baseline loses points outright.

Fit on the train split, evaluate on test, report via src.score.metrics. These numbers
go in the methodology README next to the Longevity-LLM numbers.
"""

from __future__ import annotations


def majority_class(cohort_train, cohort_test) -> dict:
    raise NotImplementedError("CS: predict the majority class; report balanced metrics.")


def age_only(cohort_train, cohort_test) -> dict:
    raise NotImplementedError("CS: logistic regression on age alone.")


def logistic_full(cohort_train, cohort_test) -> dict:
    raise NotImplementedError("CS: logistic regression on all features.")


def cox_model(cohort_train, cohort_test) -> dict:
    raise NotImplementedError("CS: Cox PH (lifelines) for the regression/survival task; report C-index.")
