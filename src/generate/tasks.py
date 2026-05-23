"""STUB — owner: Anderson. The mouse-longevity task generators. Each returns a list of
BenchmarkRecord (LongevityBench format). >= 50 rows per task (rows of a task share lb_id).
See mock/make_mock.py for the record SHAPE (note: that mock still holds the LEGACY NHANES
red-flag example rows — regenerate it as a mouse example when these generators land).

EXPERIMENT (build-plan.md §1-2; LRN-mouse-longevity-experiment-design-2026-05-23):
give the model a mouse GENOTYPE (alleles + zygosity) + the strain's PHENOTYPE profile
(its non-mortality MP terms) and predict the survival/lethality label, scored vs the
recorded MGI/IMPC ground truth. Render EVERY item in BOTH ablation conditions, tagged
in metadata.condition:
  - "geno_pheno"  : genotype + phenotype  (reasoning + recall)
  - "pheno_only"  : phenotype only, alleles removed  (reasoning, recall ablated)
Δ_recall = acc(geno_pheno) − acc(pheno_only) per model is the headline finding.

Canonical task IDs (match README.md + build-plan.md — do NOT reuse the legacy red-flag
LB-0142/0146/0150 ids, which mean different tasks here):
  LB-0138 gen_mgi_survival_binary    binary/accuracy      — impairs survival Y/N? (balanced)  [PRIMARY]
  LB-0142 gen_impc_viability         multiclass/accuracy  — viable / subviable / lethal (IMPC)
  LB-0146 gen_mgi_genotype_pairwise  pairwise/accuracy    — which genotype is more deleterious

Gold goes in the trailing assistant message. Verifiable GT + provenance go in metadata:
the recorded label, the ablation `condition`, the `gene` (the split key — split BY GENE so
a gene never spans train/test), zygosity, and source PMIDs / IMPC ids. Contamination control:
drop genotypes whose gene is in data/famous_gene_blocklist.csv. Validate with
validate/validate_jsonl.py.

PROMPT PATTERN (verified live 2026-05-23): the endpoint is vLLM-served `longevity-llm`,
28K ctx, and IGNORES JSON formatting — it reasons in verbose prose. So end EVERY prompt
with: "Reason briefly, then on the FINAL line output exactly: Answer: <letter>"
(call with max_tokens >= ~400 so it finishes). src/model/parse.py extracts the trailing
letter (the region after </think>); the reasoning prose feeds the bonus scorer -> set
has_reasoning=True. ~8s/call, so parallelize the eval pass.
"""

from __future__ import annotations

import random
from typing import List

from schema.records import (
    SYSTEM_PROMPT, BenchmarkRecord, ChatMessage, Domain, Format, Metric, Role,
)
from src import config
from src.data.mgi import GenotypeRow
from src.generate.profiles import CONDITIONS, profile_id_for, render_user_message

__all__ = ["CONDITIONS", "gen_mgi_survival_binary", "gen_impc_viability",
           "gen_mgi_genotype_pairwise", "ALL_GENERATORS"]


def _select(rows: List[GenotypeRow], n: int, seed: int, test_frac: float = 0.2) -> List[GenotypeRow]:
    """Pick n label-balanced, answerable items spanning BOTH splits. Deterministic in `seed`
    and INDEPENDENT of condition, so geno_pheno and pheno_only select the IDENTICAL genotypes
    -> the ablation is matched item-for-item (Δ_recall is computed on the same strains)."""
    rng = random.Random(seed)
    eligible = [r for r in rows if r.phenotype_terms]  # pheno_only must be answerable from phenotype
    by_split = {"train": [r for r in eligible if r.split == "train"],
                "test": [r for r in eligible if r.split == "test"]}
    n_test = max(2, round(n * test_frac))
    picked: List[GenotypeRow] = []
    for split, k in (("train", n - n_test), ("test", n_test)):
        pool = by_split[split]
        pos = [r for r in pool if r.label == 1]
        neg = [r for r in pool if r.label == 0]
        kp = min(k // 2, len(pos))
        kn = min(k - kp, len(neg))
        picked += rng.sample(pos, kp) + rng.sample(neg, kn)
    rng.shuffle(picked)
    return picked


def gen_mgi_survival_binary(rows: List[GenotypeRow], condition: str,
                            n: int = 60, seed: int = config.SEED) -> List[BenchmarkRecord]:
    """LB-0138 [PRIMARY]. Genotype + phenotype profile -> does this genotype impair survival?
    (binary/accuracy). `rows` come from src.data.mgi.load_mgi (already balanced + gene-split +
    famous-tagged). Renders the selected items under one ablation `condition`; call once per
    condition with the same seed for a matched pair. gold A=Yes (impairs), B=No."""
    out: List[BenchmarkRecord] = []
    for row in _select(rows, n, seed):
        gold = "A" if row.label == 1 else "B"
        meta = {
            "genotype_id": row.genotype_id,
            "genes": row.genes,
            "zygosity": row.zygosity,
            "expression_direction": row.expression_direction,
            "condition": condition,            # the ablation tag (Lever A)
            "label_impairs_survival": row.label,
            "is_famous": row.is_famous,        # Lever B slice key
            "split": row.split,                # gene-grouped covariate split
            "base_profile_id": profile_id_for(row),
            "n_phenotype_terms": len(row.phenotype_terms),
            "pmids": row.pmids,                # verifiable ground truth
            "source": "MGI",
        }
        out.append(BenchmarkRecord(
            lb_id="LB-0138",
            pool="mgi_survival_binary",
            display_name="MGI Genotype Survival / Binary",
            display_group="Mouse Longevity (MGI)",
            domain=Domain.genetics,
            format=Format.binary,
            metric=Metric.accuracy,
            units=None,
            messages=[
                ChatMessage(role=Role.system, content=SYSTEM_PROMPT),
                ChatMessage(role=Role.user, content=render_user_message(row, condition)),
                ChatMessage(role=Role.assistant, content=gold),
            ],
            task="mgi_survival_binary",
            has_reasoning=True,
            metadata=meta,
        ))
    return out


def gen_impc_viability(impc_rows, condition: str, n: int = 60) -> List[BenchmarkRecord]:
    """LB-0142. Homozygous knockout -> IMPC viability: viable / subviable / lethal
    (multiclass/accuracy). BLOCKED: the current impc_viability.csv extract is lethal-only —
    needs a viable/subviable Solr pull before this is buildable (build-plan IMPC note)."""
    raise NotImplementedError("Blocked on data: re-pull IMPC viable/subviable before building.")


def gen_mgi_genotype_pairwise(rows, condition: str, n: int = 60) -> List[BenchmarkRecord]:
    """LB-0146. Two genotypes -> which is MORE deleterious to survival (pairwise/accuracy).
    Pair a survival-impairing genotype against a no-mortality one; gold = the deleterious option."""
    raise NotImplementedError("Anderson: build pairwise; gold = more-deleterious genotype.")


ALL_GENERATORS = (gen_mgi_survival_binary, gen_impc_viability, gen_mgi_genotype_pairwise)
