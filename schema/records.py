"""THE CONTRACT — matches the LongevityBench row schema verbatim so our tasks
*extend the framework* (the track's explicit ask) and are judged in their format.

Confirmed against a real `LB-0042` row (NHANES Mortality / Binary):
  fields: lb_id, pool, display_name, display_group, domain, format, metric, units,
          messages, task, has_reasoning, metadata
  - gold answer = the FINAL assistant message content (e.g. "A" for an MC/binary item)
  - prompt sent to the model = messages[:-1] (system + user)
  - metadata is a JSON STRING of per-row provenance (their LB-0042 used {"follow_up": ...})

We carry OUR verifiable ground truth (red-flag direction, matched-cohort band,
should_moderate, split, covariates) INSIDE `metadata` — same schema, our rigor.

Do NOT change a field without telling the team; this is what everyone builds against.
A submission line is exactly `record.model_dump_json()`.
"""

from __future__ import annotations

import json
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator

# Reuse LongevityBench's system prompt verbatim (from the real LB-0042 row).
SYSTEM_PROMPT = ("You are a biomedical AI specialized in aging biology, trained on "
                 "genomic, proteomic, and clinical data.")


class Role(str, Enum):
    system = "system"
    user = "user"
    assistant = "assistant"


class Domain(str, Enum):
    epigenomics = "epigenomics"
    transcriptomics = "transcriptomics"
    proteomics = "proteomics"
    clinical = "clinical"
    genetics = "genetics"


class Format(str, Enum):
    binary = "binary"
    multiclass = "multiclass"
    ternary = "ternary"
    pairwise = "pairwise"
    regression = "regression"
    generation = "generation"


class Metric(str, Enum):
    accuracy = "accuracy"
    mae = "mae"
    jaccard = "jaccard"


# format -> required metric (from the dataset card)
_METRIC_FOR_FORMAT = {
    Format.binary: Metric.accuracy,
    Format.multiclass: Metric.accuracy,
    Format.ternary: Metric.accuracy,
    Format.pairwise: Metric.accuracy,
    Format.regression: Metric.mae,
    Format.generation: Metric.jaccard,
}


class ChatMessage(BaseModel):
    role: Role
    content: str


class BenchmarkRecord(BaseModel):
    lb_id: str                       # e.g. "LB-0142" (continue past their range; avoid collisions)
    pool: str                        # source slug, e.g. "nhanes_redflag_pairwise"
    display_name: str                # e.g. "NHANES Red-Flag / Pairwise"
    display_group: str               # e.g. "NHANES Red-Flag Robustness"
    domain: Domain = Domain.clinical
    format: Format
    metric: Metric
    units: Optional[str] = None      # "years"|"months"|"days"|None
    messages: List[ChatMessage]      # system + user [+ assistant gold]
    task: str                        # free-text/slug task description
    has_reasoning: bool = False
    metadata: str                    # JSON-encoded string (see meta()/build helpers)

    # filled by the validator; leave None on generation
    token_count_cl100k: Optional[int] = None

    @field_validator("metadata", mode="before")
    @classmethod
    def _dump_metadata(cls, v: Any) -> str:
        """Accept a dict at construction for convenience; store as a JSON string."""
        if isinstance(v, dict):
            return json.dumps(v)
        return v

    @model_validator(mode="after")
    def _check(self) -> "BenchmarkRecord":
        if not self.messages:
            raise ValueError("messages must be non-empty")
        if _METRIC_FOR_FORMAT[self.format] != self.metric:
            raise ValueError(
                f"format={self.format.value} requires metric={_METRIC_FOR_FORMAT[self.format].value}, "
                f"got {self.metric.value}")
        # gold must be the trailing assistant turn
        if self.messages[-1].role != Role.assistant or not self.messages[-1].content.strip():
            raise ValueError("last message must be a non-empty assistant turn (the gold answer)")
        # metadata must be valid JSON
        try:
            json.loads(self.metadata)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"metadata must be a JSON string: {e}")
        return self

    # ---- helpers ---------------------------------------------------------- #
    def gold(self) -> str:
        return self.messages[-1].content.strip()

    def meta(self) -> Dict[str, Any]:
        return json.loads(self.metadata)


def prompt_messages(record: BenchmarkRecord) -> List[ChatMessage]:
    """What is actually sent to the model: everything except the trailing gold."""
    if record.messages and record.messages[-1].role == Role.assistant:
        return record.messages[:-1]
    return list(record.messages)


def prompt_text(record: BenchmarkRecord) -> str:
    """Text whose tokens count against the 30K cl100k budget (prompt only, not gold)."""
    return "\n".join(m.content for m in prompt_messages(record))
