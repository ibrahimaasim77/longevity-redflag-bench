"""STUB — owner: Anderson. The 5 task generators. Each returns a list of
BenchmarkRecord (the contract). >= 50 per task. See mock/make_mock.py for worked
examples of every record shape.

Layer A (absolute GT, real outcome): binary_survival, ordinal_risk, regression
Layer B (relative GT, red-flag effect): pairwise_counterfactual, set_generation

Wire phrasing variants (phrasings.py) for Diversity and tag split by `covariates.cycle`
(build-plan.md §4 covariate split). Validate output with validate/validate_jsonl.py.
"""

from __future__ import annotations

from typing import List

from schema.records import BenchmarkRecord


def gen_binary_survival(cohort, n: int = 60) -> List[BenchmarkRecord]:
    raise NotImplementedError("Anderson: real profile -> survive >=10yr? GT=died_10yr.")


def gen_ordinal_risk(cohort, n: int = 60) -> List[BenchmarkRecord]:
    raise NotImplementedError("Anderson: low/med/high vs empirical tertiles.")


def gen_pairwise_counterfactual(cohort, redflags, effects, n: int = 60) -> List[BenchmarkRecord]:
    raise NotImplementedError("Anderson: A vs A+redflag; GT=relative effect.")


def gen_set_generation(cohort, redflags, n: int = 60) -> List[BenchmarkRecord]:
    raise NotImplementedError("Anderson: which factors raise THIS person's risk? include keyword traps.")


def gen_regression(cohort, n: int = 60) -> List[BenchmarkRecord]:
    raise NotImplementedError("Anderson: risk score / years; handle censoring in GT.")


ALL_GENERATORS = (
    gen_binary_survival, gen_ordinal_risk,
    gen_pairwise_counterfactual, gen_set_generation, gen_regression,
)
