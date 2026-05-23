"""THE CONTRACT.

This is the single coordination mechanism for the whole team. Every workstream
(NHANES pipeline, task generators, scorer, validator) reads and writes
`BenchmarkRecord`. Do NOT change a field without telling the team in the channel —
a silent schema change breaks everyone downstream.

A submission line in `benchmark.jsonl` is exactly `BenchmarkRecord.model_dump_json()`.

Ground truth comes in two flavors (see build-plan.md §2):
  - Layer A (absolute): a real linked outcome. `gt_kind="absolute"`, `answer` set,
    `outcome_source="nhanes_lmf_10yr"`.
  - Layer B (relative): a counterfactual effect. `gt_kind="relative"`,
    `expected_direction` + `magnitude_band` + `should_moderate` set,
    `outcome_source` in {"epi_directional","matched_cohort"}.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field, model_validator


# --------------------------------------------------------------------------- #
# Enumerations — the closed vocabularies. Add a value here, not as a free string.
# --------------------------------------------------------------------------- #
class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class TaskType(str, Enum):
    binary_survival = "binary_survival"          # Layer A
    ordinal_risk = "ordinal_risk"                # Layer A
    pairwise_counterfactual = "pairwise_counterfactual"  # Layer B
    set_generation = "set_generation"            # Layer B
    regression = "regression"                    # Layer A


class AnswerFormat(str, Enum):
    binary = "binary"
    multiple_choice = "multiple_choice"
    ternary = "ternary"
    pairwise = "pairwise"
    regression = "regression"
    set_generation = "set_generation"


class Split(str, Enum):
    train = "train"
    test = "test"


class GTKind(str, Enum):
    absolute = "absolute"   # real linked outcome (Layer A)
    relative = "relative"   # counterfactual effect (Layer B)


class OutcomeSource(str, Enum):
    nhanes_lmf_10yr = "nhanes_lmf_10yr"     # NHANES Linked Mortality File, 10-yr horizon
    epi_directional = "epi_directional"     # published epidemiology, direction only
    matched_cohort = "matched_cohort"       # empirical effect from matched NHANES cohort


# --------------------------------------------------------------------------- #
# Records
# --------------------------------------------------------------------------- #
class ChatMessage(BaseModel):
    role: Role
    content: str


class GroundTruth(BaseModel):
    """Verifiable ground truth. Exactly one flavor must be fully populated."""
    gt_kind: GTKind
    outcome_source: OutcomeSource

    # Layer A (absolute)
    answer: Optional[Union[str, float, int, List[str]]] = None

    # Layer B (relative)
    expected_direction: Optional[str] = None      # e.g. "increase_risk" | "decrease_risk" | "no_change"
    magnitude_band: Optional[str] = None           # e.g. "HR 1.0-1.3"
    should_moderate: List[str] = Field(default_factory=list)
    evidence_ids: List[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _check_flavor(self) -> "GroundTruth":
        if self.gt_kind == GTKind.absolute:
            if self.answer is None:
                raise ValueError("absolute ground truth requires `answer`")
        else:  # relative
            if not self.expected_direction:
                raise ValueError("relative ground truth requires `expected_direction`")
        return self


class Covariates(BaseModel):
    """Used for the covariate train-test split (split by `cycle`) and rigor reporting."""
    age: Optional[float] = None
    sex: Optional[str] = None
    race_eth: Optional[str] = None
    cycle: Optional[str] = None          # NHANES cycle, e.g. "1999-2000" — the split key

    model_config = {"extra": "allow"}    # extra clinical covariates welcome


class BenchmarkRecord(BaseModel):
    item_id: str
    task: TaskType
    format: AnswerFormat
    split: Split
    messages: List[ChatMessage]
    ground_truth: GroundTruth

    # provenance / metadata
    base_profile_id: Optional[str] = None    # NHANES-derived profile this came from
    red_flag: Optional[str] = None           # the perturbation applied (Layer B), if any
    covariates: Covariates = Field(default_factory=Covariates)

    # filled by the validator (validate/validate_jsonl.py); leave None on generation
    token_count_cl100k: Optional[int] = None

    @model_validator(mode="after")
    def _check_task_gt_consistency(self) -> "BenchmarkRecord":
        layer_b = {TaskType.pairwise_counterfactual, TaskType.set_generation}
        if self.task in layer_b and self.ground_truth.gt_kind != GTKind.relative:
            raise ValueError(f"{self.task.value} requires relative ground truth")
        if self.task not in layer_b and self.ground_truth.gt_kind != GTKind.absolute:
            raise ValueError(f"{self.task.value} requires absolute ground truth")
        if not self.messages:
            raise ValueError("messages must be non-empty")
        return self


def prompt_text(record: BenchmarkRecord) -> str:
    """The text whose tokens count against the 30K cl100k budget: all message content."""
    return "\n".join(m.content for m in record.messages)
