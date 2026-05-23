"""Step-1 loader: MGI genotype -> survival rows for the mouse-longevity benchmark.

Reads data/mgi_genotype_phenotype.csv (built by scripts/build_mgi_dataset.py) and returns
GenotypeRow objects ready for the LB-0138 binary generator (and LB-0146 pairwise). Each row
carries the phenotype profile, the survival label, a famous-gene tag, and a train/test split.

Three rigor pieces live HERE (build-plan.md §6; LRN-mouse-longevity-experiment-design-2026-05-23):

  1. Famous-gene tagging (Lever B). is_famous = any constituent gene is on
     data/famous_gene_blocklist.csv. We TAG every row rather than dropping, so the
     famous-vs-obscure comparison can be sliced at eval time without rebuilding.
     `include_famous=False` (default) filters them out for the clean retrieval-resistant set.

  2. Balance the binary task. The raw set is ~26/74 (19,816 impair survival vs 54,757 no
     mortality) -> an always-"no-effect" guesser scores 74%. We keep all positives and
     downsample negatives to match, WITHIN each split, so the majority baseline is 50%.

  3. Split BY GENE (the leakage key). The same gene must never span train/test, or a model
     memorizes gene->label from train and aces test by lookup. Multi-gene genotypes
     (e.g. Alx4|Bmp4) are handled with union-find: genes that co-occur form a COMPONENT,
     and whole components go to one side, so no constituent gene leaks across the split.

NOT here: the geno/pheno ablation (Lever A) is applied by the generator, not the loader.

Run `python -m src.data.mgi` (or `python src/data/mgi.py`) to print a summary + leakage check.
"""

from __future__ import annotations

import csv
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from src import config

MGI_CSV = config.DATA_DIR / "mgi_genotype_phenotype.csv"
BLOCKLIST_CSV = config.DATA_DIR / "famous_gene_blocklist.csv"


@dataclass
class GenotypeRow:
    genotype_id: str            # MGI:xxxxxxx
    genes: List[str]            # constituent gene symbols (from gene_symbols, '|'-split)
    component: str              # gene-component id (the split group; stable per connected gene set)
    alleles: str                # allelic_composition (the genotype string shown in geno_pheno)
    zygosity: str               # homozygote | heterozygote | multi-locus | ...
    expression_direction: str   # decreased | increased | none | unknown | mixed | altered
    genetic_background: str
    phenotype_terms: List[str]  # non-mortality MP terms = the phenotype profile (mortality already excluded)
    label: int                  # 1 = impairs survival, 0 = no mortality phenotype
    pmids: List[str]            # provenance for verifiable GT
    is_famous: bool             # any gene in the GenAge famous-gene blocklist
    split: str = "train"        # train | test (assigned by gene component)


# --------------------------------------------------------------------------- #
# union-find over co-occurring genes -> components (so multi-gene rows can't leak)
# --------------------------------------------------------------------------- #
class _UnionFind:
    def __init__(self) -> None:
        self._parent: Dict[str, str] = {}

    def find(self, x: str) -> str:
        self._parent.setdefault(x, x)
        root = x
        while self._parent[root] != root:
            root = self._parent[root]
        while self._parent[x] != root:  # path compression
            self._parent[x], x = root, self._parent[x]
        return root

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[ra] = rb


def _split_field(value: str, sep: str = "|") -> List[str]:
    return [p.strip() for p in (value or "").split(sep) if p.strip()]


def _load_blocklist() -> Set[str]:
    if not BLOCKLIST_CSV.exists():
        return set()
    genes: Set[str] = set()
    with BLOCKLIST_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            g = (row.get("gene_symbol") or "").strip()
            if g:
                genes.add(g)
    return genes


def _read_raw() -> List[dict]:
    if not MGI_CSV.exists():
        raise FileNotFoundError(f"{MGI_CSV} not found — run scripts/build_mgi_dataset.py first.")
    with MGI_CSV.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _assign_components(parsed: List[dict]) -> None:
    """Mutate each parsed row, setting row['component'] to its gene-component root."""
    uf = _UnionFind()
    for row in parsed:
        genes = row["genes"]
        if not genes:
            continue
        first = genes[0]
        uf.find(first)
        for g in genes[1:]:
            uf.union(first, g)
    for row in parsed:
        # component id = root of the first gene; genotypes with no gene get their own group
        row["component"] = uf.find(row["genes"][0]) if row["genes"] else f"_nogene:{row['genotype_id']}"


def _gene_group_split(parsed: List[dict], test_frac: float, seed: int) -> None:
    """Assign split='train'|'test' so a whole gene-component lands on one side only."""
    components = sorted({row["component"] for row in parsed})
    rng = random.Random(seed)
    rng.shuffle(components)
    n_test = max(1, round(len(components) * test_frac)) if components else 0
    test_components = set(components[:n_test])
    for row in parsed:
        row["split"] = "test" if row["component"] in test_components else "train"


def _balance_within_splits(rows: List[GenotypeRow], seed: int) -> List[GenotypeRow]:
    """Keep all positives; downsample negatives to the positive count, per split."""
    out: List[GenotypeRow] = []
    rng = random.Random(seed)
    for split in ("train", "test"):
        pos = [r for r in rows if r.split == split and r.label == 1]
        neg = [r for r in rows if r.split == split and r.label == 0]
        if neg and len(neg) > len(pos):
            neg = rng.sample(neg, len(pos))
        out.extend(pos + neg)
    return out


def load_mgi(
    *,
    include_famous: bool = False,
    balance: bool = True,
    test_frac: float = 0.2,
    seed: int = config.SEED,
) -> List[GenotypeRow]:
    """Load MGI genotype rows.

    include_famous : keep genes on the famous-gene blocklist (default False = obscure-only).
                     is_famous is tagged regardless, so you can also slice at eval time.
    balance        : downsample negatives to match positives, within each split.
    test_frac      : fraction of gene-components held out for test.
    seed           : reproducible shuffle/sample (config.SEED).
    """
    blocklist = _load_blocklist()
    parsed: List[dict] = []
    for raw in _read_raw():
        genes = _split_field(raw.get("gene_symbols", ""))
        parsed.append({
            "genotype_id": raw.get("genotype_id", ""),
            "genes": genes,
            "alleles": raw.get("allelic_composition", ""),
            "zygosity": raw.get("zygosity", ""),
            "expression_direction": raw.get("expression_direction", ""),
            "genetic_background": raw.get("genetic_background", ""),
            "phenotype_terms": _split_field(raw.get("phenotype_terms", "")),
            "label": 1 if str(raw.get("label_impairs_survival", "")).strip() in ("1", "True", "true") else 0,
            "pmids": _split_field(raw.get("pmids", "")),
            "is_famous": any(g in blocklist for g in genes),
        })

    if not include_famous:
        parsed = [r for r in parsed if not r["is_famous"]]

    _assign_components(parsed)
    _gene_group_split(parsed, test_frac=test_frac, seed=seed)

    rows = [GenotypeRow(**{k: r[k] for k in (
        "genotype_id", "genes", "component", "alleles", "zygosity", "expression_direction",
        "genetic_background", "phenotype_terms", "label", "pmids", "is_famous", "split",
    )}) for r in parsed]

    if balance:
        rows = _balance_within_splits(rows, seed=seed)
    return rows


def summarize(rows: List[GenotypeRow]) -> Dict[str, object]:
    def counts(split: Optional[str]) -> Dict[str, int]:
        sub = [r for r in rows if split is None or r.split == split]
        return {"n": len(sub), "pos": sum(r.label for r in sub), "neg": sum(1 for r in sub if r.label == 0)}

    train_genes = {g for r in rows if r.split == "train" for g in r.genes}
    test_genes = {g for r in rows if r.split == "test" for g in r.genes}
    return {
        "total": counts(None),
        "train": counts("train"),
        "test": counts("test"),
        "famous_rows": sum(1 for r in rows if r.is_famous),
        "gene_overlap_train_test": len(train_genes & test_genes),  # MUST be 0 (leakage check)
    }


if __name__ == "__main__":
    import json

    for label, kw in (("obscure-only, balanced", {}),
                      ("famous included, balanced", {"include_famous": True}),
                      ("obscure-only, UNbalanced", {"balance": False})):
        s = summarize(load_mgi(**kw))
        print(f"\n[{label}]")
        print(json.dumps(s, indent=2))
        assert s["gene_overlap_train_test"] == 0, "LEAKAGE: a gene spans train/test!"
    print("\nLeakage check passed (no gene spans train/test in any config).")
