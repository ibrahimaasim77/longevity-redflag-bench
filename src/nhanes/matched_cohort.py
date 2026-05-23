"""STUB — owner: CS teammate (covariates specified by Bio 1). The rigorous Layer-B
ground truth: empirical effect of a red flag computed from NHANES itself.

For a red flag, find real individuals matching on (age-band, sex[, more]) who DO vs
DON'T have it, and compute the empirical 10-year mortality-rate difference + CI. This
is data-derived and retrieval-resistant (scores on Statistical Rigor). Where this
diverges from published HRs (esp. in-remission / on-medication), that's the signal.

Returns the `expected_direction` + `magnitude_band` that populate a relative GroundTruth.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class MatchedEffect:
    red_flag: str
    expected_direction: str   # increase_risk | decrease_risk | no_change
    rate_with: float
    rate_without: float
    risk_diff: float
    ci: tuple
    n_with: int
    n_without: int

    @property
    def magnitude_band(self) -> str:
        return f"RD {self.risk_diff:+.2%} (95% CI {self.ci[0]:+.2%},{self.ci[1]:+.2%})"


def empirical_effect(cohort, red_flag_key: str, flag_predicate, match_on=("age_band", "sex")) -> MatchedEffect:
    """cohort: DataFrame from build_cohort. flag_predicate: row->bool. TODO(cs)."""
    raise NotImplementedError("CS teammate: matched-cohort empirical effect. Bio 1 specifies match_on.")
