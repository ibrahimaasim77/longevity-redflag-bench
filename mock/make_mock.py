"""Generate mock/mock_records.jsonl — fake but schema-valid records spanning all 5
task types + both ground-truth flavors. Lets the validator, scorer, and any viz be
built and tested NOW, before the real NHANES pipeline lands.

This file doubles as worked examples of how to construct a BenchmarkRecord. Run:
    python mock/make_mock.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schema.records import (AnswerFormat, BenchmarkRecord, ChatMessage, Covariates,
                            GroundTruth, GTKind, OutcomeSource, Role, Split, TaskType)

SYS = "You are a clinical longevity model. Answer concisely in the requested format."


def _profile(age, sex, extra=""):
    return (f"Patient: age {age}, sex {sex}. Labs: SBP 128 mmHg, BMI 27.4, "
            f"total cholesterol 198, HbA1c 5.6%, never-smoker.{extra}")


def msgs(user):
    return [ChatMessage(role=Role.system, content=SYS), ChatMessage(role=Role.user, content=user)]


def build():
    recs = []

    # --- binary_survival (Layer A, absolute) ---
    for i, (age, sex, ans, cyc) in enumerate(
        [(67, "male", "no", "1999-2000"), (54, "female", "yes", "1999-2000"),
         (72, "male", "no", "2003-2004"), (61, "female", "yes", "2003-2004")]):
        recs.append(BenchmarkRecord(
            item_id=f"bin-{i:03d}",
            task=TaskType.binary_survival, format=AnswerFormat.binary,
            split=Split.train if cyc == "1999-2000" else Split.test,
            messages=msgs(_profile(age, sex) + " Will this person survive at least 10 years? Answer yes or no."),
            ground_truth=GroundTruth(gt_kind=GTKind.absolute, outcome_source=OutcomeSource.nhanes_lmf_10yr, answer=ans),
            base_profile_id=f"nhanes-{i:04d}",
            covariates=Covariates(age=age, sex=sex, cycle=cyc)))

    # --- ordinal_risk (Layer A) ---
    for i, (age, sex, ans, cyc) in enumerate(
        [(80, "male", "high", "1999-2000"), (45, "female", "low", "2003-2004"),
         (66, "male", "medium", "2003-2004")]):
        recs.append(BenchmarkRecord(
            item_id=f"ord-{i:03d}",
            task=TaskType.ordinal_risk, format=AnswerFormat.ternary,
            split=Split.train if cyc == "1999-2000" else Split.test,
            messages=msgs(_profile(age, sex) + " Classify 10-year mortality risk as low, medium, or high."),
            ground_truth=GroundTruth(gt_kind=GTKind.absolute, outcome_source=OutcomeSource.nhanes_lmf_10yr, answer=ans),
            base_profile_id=f"nhanes-{10+i:04d}", covariates=Covariates(age=age, sex=sex, cycle=cyc)))

    # --- pairwise_counterfactual (Layer B, relative) ---
    for i, (rf, direction, band, mod, cyc) in enumerate(
        [("current_smoker", "increase_risk", "HR 2.0-3.0", ["pack-years", "age"], "1999-2000"),
         ("past_cancer_in_remission", "increase_risk", "HR 1.0-1.3", ["remission status", "current biomarkers"], "2003-2004"),
         ("excellent_self_rated_health", "decrease_risk", "HR <1.0", ["protective signal"], "2003-2004")]):
        recs.append(BenchmarkRecord(
            item_id=f"pair-{i:03d}",
            task=TaskType.pairwise_counterfactual, format=AnswerFormat.pairwise,
            split=Split.train if cyc == "1999-2000" else Split.test,
            messages=msgs(f"Profile A: {_profile(63, 'male')}\nProfile B: same patient but with '{rf}'.\n"
                          "Which profile has HIGHER 10-year mortality risk? Answer A or B."),
            ground_truth=GroundTruth(gt_kind=GTKind.relative, outcome_source=OutcomeSource.matched_cohort,
                                     expected_direction=direction, magnitude_band=band, should_moderate=mod,
                                     evidence_ids=[f"ev-{rf}"]),
            base_profile_id=f"nhanes-{20+i:04d}", red_flag=rf, covariates=Covariates(age=63, sex="male", cycle=cyc)))

    # --- set_generation (Layer B) — includes a keyword-reactive trap ---
    recs.append(BenchmarkRecord(
        item_id="set-000",
        task=TaskType.set_generation, format=AnswerFormat.set_generation, split=Split.train,
        messages=msgs(_profile(70, "male", " On effective antihypertensive therapy, BP well controlled.")
                      + " Which of these RAISE this person's 10-year mortality risk? "
                        "Options: current smoking, controlled hypertension on medication, age 70, daily exercise. "
                        "Return a JSON list."),
        ground_truth=GroundTruth(gt_kind=GTKind.relative, outcome_source=OutcomeSource.epi_directional,
                                 expected_direction="increase_risk", magnitude_band="mixed",
                                 should_moderate=["controlled hypertension is managed, not a raw red flag"],
                                 answer=["current smoking", "age 70"], evidence_ids=["ev-trap-htn"]),
        base_profile_id="nhanes-0030", red_flag="set_mixed", covariates=Covariates(age=70, sex="male", cycle="1999-2000")))

    # --- regression (Layer A) ---
    for i, (age, sex, yrs, cyc) in enumerate([(58, "female", 9.2, "1999-2000"), (75, "male", 4.1, "2003-2004")]):
        recs.append(BenchmarkRecord(
            item_id=f"reg-{i:03d}",
            task=TaskType.regression, format=AnswerFormat.regression,
            split=Split.train if cyc == "1999-2000" else Split.test,
            messages=msgs(_profile(age, sex) + " Estimate this person's risk score for 10-year mortality (0-100)."),
            ground_truth=GroundTruth(gt_kind=GTKind.absolute, outcome_source=OutcomeSource.nhanes_lmf_10yr, answer=yrs),
            base_profile_id=f"nhanes-{40+i:04d}", covariates=Covariates(age=age, sex=sex, cycle=cyc)))

    return recs


def main():
    recs = build()
    out = os.path.join(os.path.dirname(__file__), "mock_records.jsonl")
    with open(out, "w", encoding="utf-8") as f:
        for r in recs:
            f.write(r.model_dump_json() + "\n")
    print(f"wrote {len(recs)} mock records -> {out}")


if __name__ == "__main__":
    main()
