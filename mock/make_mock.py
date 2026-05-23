"""Generate mock/mock_records.jsonl in LongevityBench format, for our THREE NOVEL
red-flag tasks (the plain NHANES mortality tasks already exist as LB-0042/46/50/54 —
we don't rebuild those). Lets the validator/parser/scorer be built and tested NOW.

Doubles as worked examples of the reconciled contract. Run:  python mock/make_mock.py

Task ids (lb_id is per-TASK, shared across its rows; continue past LB-0054):
  LB-0142  nhanes_redflag_pairwise   (pairwise / accuracy)
  LB-0146  nhanes_redflag_relevance  (binary  / accuracy)  -- includes keyword-reactive traps
  LB-0150  nhanes_redflag_setgen     (generation / jaccard)
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from schema.records import (BenchmarkRecord, ChatMessage, Format, Metric, Role,
                            SYSTEM_PROMPT)


def _profile(age, sex, eth="non-Hispanic white", extra_meds="None", extra_qx=""):
    """Mirror the LB-0042 rendering: sectioned, no identifiers."""
    return (
        f"Demographics: {age}-year-old {sex} of {eth} ethnicity.\n"
        f"Body measurements: BMI 27.4 kg/m2, waist circumference 98.0 cm, "
        f"diastolic BP 78.0 mm Hg, systolic BP 128.0 mm Hg, pulse 70.0 bpm.\n"
        f"Blood test results: Albumin 44.0 g/L; Total cholesterol 198.0 mg/dL; "
        f"HDL 52.0 mg/dL; Glycohemoglobin 5.6%; C-reactive protein 0.20 mg/dL; "
        f"Creatinine 78.0 umol/L; White blood cell count 6.2 1000 cells/uL.\n"
        f"Current medications: {extra_meds}.\n"
        f"Questionnaire: Never smoker. Engages in moderate weekly activity.{extra_qx}"
    )


def _msgs(user, gold):
    return [ChatMessage(role=Role.system, content=SYSTEM_PROMPT),
            ChatMessage(role=Role.user, content=user),
            ChatMessage(role=Role.assistant, content=gold)]


def build():
    recs = []

    # --- LB-0142  pairwise: A (base) vs B (base + ONE red flag) ---------------- #
    pairwise = [
        ("current_smoker", "current smoker (≈1 pack/day)", "increase_risk", "HR 2.0-3.0",
         ["pack-years", "age"], "B"),
        ("past_cancer_in_remission", "history of cancer, in sustained remission >5y, normal current biomarkers",
         "increase_risk", "HR 1.0-1.3", ["remission status", "current biomarkers"], "B"),
        ("excellent_self_rated_health", "reports excellent self-rated health",
         "decrease_risk", "HR <1.0", ["protective signal"], "A"),  # B is lower-risk -> higher is A
        ("high_systolic_bp", "systolic BP 168 mm Hg (uncontrolled)", "increase_risk",
         "HR per +20mmHg", ["control status"], "B"),
    ]
    for i, (rf, phrase, direction, band, mod, gold) in enumerate(pairwise):
        cyc = "1999-2000" if i % 2 == 0 else "2003-2004"
        user = ("Compare these two patient profiles for 10-year mortality risk.\n\n"
                "Options: A. Profile A  B. Profile B\n\n"
                f"Profile A:\n{_profile(63, 'male')}\n\n"
                f"Profile B: the same patient, except: {phrase}.\n\n"
                "Which profile has the HIGHER 10-year mortality risk? Answer with the option letter.")
        recs.append(BenchmarkRecord(
            lb_id="LB-0142", pool="nhanes_redflag_pairwise",
            display_name="NHANES Red-Flag / Pairwise", display_group="NHANES Red-Flag Robustness",
            format=Format.pairwise, metric=Metric.accuracy, units=None,
            messages=_msgs(user, gold), task="nhanes_redflag_pairwise",
            metadata={"base_profile_id": f"rf-{i:04d}", "red_flag": rf,
                      "expected_direction": direction, "magnitude_band": band,
                      "should_moderate": mod, "evidence_ids": [f"ev-{rf}"],
                      "split": "train" if cyc == "1999-2000" else "test", "cycle": cyc}))

    # --- LB-0146  relevance (binary): is THIS flag a real driver for THIS patient? -- #
    relevance = [
        ("current_smoker", "current smoker", "A", ["genuine driver"], "increase_risk"),       # yes
        ("controlled_hypertension", "hypertension, well controlled on medication", "B",         # NO -- trap
         ["controlled/managed, not a raw red flag"], "no_change"),
        ("family_history_diabetes", "family history of diabetes (patient HbA1c normal)", "B",   # NO -- trap
         ["hereditary cue, not current disease"], "no_change"),
    ]
    for i, (rf, phrase, gold, mod, direction) in enumerate(relevance):
        cyc = "1999-2000" if i % 2 == 0 else "2003-2004"
        meds = "Lisinopril (taken for 5 years)" if rf == "controlled_hypertension" else "None"
        user = (f"{_profile(70, 'female', extra_meds=meds)}\n\n"
                f"Is '{phrase}' a clinically significant driver of THIS patient's 10-year "
                "mortality risk, given their full profile?\n\n"
                "Options: A. Yes  B. No\n\nAnswer with the option letter.")
        recs.append(BenchmarkRecord(
            lb_id="LB-0146", pool="nhanes_redflag_relevance",
            display_name="NHANES Red-Flag / Relevance", display_group="NHANES Red-Flag Robustness",
            format=Format.binary, metric=Metric.accuracy, units=None,
            messages=_msgs(user, gold), task="nhanes_redflag_relevance",
            metadata={"base_profile_id": f"rel-{i:04d}", "red_flag": rf,
                      "expected_direction": direction, "magnitude_band": "context-dependent",
                      "should_moderate": mod, "evidence_ids": [f"ev-{rf}"], "is_trap": gold == "B",
                      "split": "train" if cyc == "1999-2000" else "test", "cycle": cyc}))

    # --- LB-0150  set generation (jaccard): which listed factors RAISE this risk? ---- #
    setgen = [
        (["current smoking", "age 72"],
         "current smoking; controlled hypertension on medication; age 72; daily exercise",
         "2003-2004"),
        (["current smoking", "uncontrolled diabetes"],
         "regular exercise; current smoking; family history only of cancer; uncontrolled diabetes",
         "1999-2000"),
    ]
    for i, (gold_set, options, cyc) in enumerate(setgen):
        user = (f"{_profile(72, 'male')}\n\n"
                "From the list below, return ONLY the factors that RAISE this patient's "
                "10-year mortality risk, as a comma-separated list.\n"
                f"Factors: {options}.")
        recs.append(BenchmarkRecord(
            lb_id="LB-0150", pool="nhanes_redflag_setgen",
            display_name="NHANES Red-Flag / Set", display_group="NHANES Red-Flag Robustness",
            format=Format.generation, metric=Metric.jaccard, units=None,
            messages=_msgs(user, ", ".join(gold_set)), task="nhanes_redflag_setgen",
            metadata={"base_profile_id": f"set-{i:04d}", "red_flag": "set_mixed",
                      "expected_direction": "increase_risk", "magnitude_band": "mixed",
                      "should_moderate": ["distinguish raw drivers from managed/hereditary cues"],
                      "gold_set": gold_set, "evidence_ids": ["ev-set-trap"],
                      "split": "train" if cyc == "1999-2000" else "test", "cycle": cyc}))

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
