"""Render a GenotypeRow (src.data.mgi.GenotypeRow) into the user-message text for a task,
under a given ablation condition. The condition IS the reasoning-vs-recall lever (Lever A):

  geno_pheno : show gene(s) + allelic composition + zygosity + the phenotype profile.
  pheno_only : show ONLY the phenotype profile (+ a generic zygosity/mechanism framing);
               the gene name and allele identifiers are removed, so the model cannot
               answer by recalling facts about the specific gene — it must reason from
               the phenotype. acc(geno_pheno) - acc(pheno_only) = the model's recall reliance.

The phenotype profile is the strain's recorded NON-mortality MP terms (the mortality/aging
terms were withheld at extract time — they are the label). Every prompt ends with the
verified forced-answer line so src/model/parse.py can read the trailing letter (the 9B
endpoint ignores JSON and reasons in prose).
"""

from __future__ import annotations

from src.data.mgi import GenotypeRow

CONDITIONS = ("geno_pheno", "pheno_only")

_ANSWER_LINE = "Reason briefly, then on the FINAL line output exactly: Answer: <letter>"

_QUESTION = ("Question: Does this genotype impair survival "
             "(cause premature death or a shortened lifespan)?\n\n"
             "Options: A. Yes  B. No")

# generic, identity-free phrasing of zygosity for the pheno_only arm (keeps the
# recessive/dominant mechanism cue, which is reasoning — not gene recall)
_ZYGOSITY_PHRASE = {
    "homozygote": "a homozygous mutation",
    "heterozygote": "a heterozygous mutation",
    "hemizygote": "a hemizygous mutation",
    "multi-locus": "mutations at multiple loci",
    "compound/other": "a compound genotype",
    "other": "a mutation",
}


def profile_id_for(row: GenotypeRow) -> str:
    """Stable id for provenance + the validator's split-leakage check. The MGI genotype_id
    is already stable and opaque (not a leaky free-text key), and is fixed to one split by
    the gene-component grouping in the loader."""
    return row.genotype_id


def _phenotype_block(row: GenotypeRow) -> str:
    if not row.phenotype_terms:
        return "Recorded phenotype profile: none reported (no non-lethal phenotype findings)."
    bullets = "\n".join(f"- {t}" for t in row.phenotype_terms)
    return "Recorded phenotype profile (excluding any lifespan/mortality findings):\n" + bullets


def _genotype_block(row: GenotypeRow) -> str:
    genes = ", ".join(row.genes) if row.genes else "(unnamed)"
    return (f"Gene(s): {genes}\n"
            f"Allelic composition: {row.alleles}\n"
            f"Zygosity: {row.zygosity}")


def render_user_message(row: GenotypeRow, condition: str) -> str:
    """Build the user-turn text for `row` under `condition`. Identical phenotype block in
    both conditions; only the genotype header differs (shown vs withheld)."""
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}; expected one of {CONDITIONS}")
    if condition == "geno_pheno":
        head = "A laboratory mouse strain carries the following mutation:\n\n" + _genotype_block(row)
    else:  # pheno_only
        zyg = _ZYGOSITY_PHRASE.get(row.zygosity, "a mutation")
        head = f"A laboratory mouse strain carries {zyg} in an undisclosed gene."
    return f"{head}\n\n{_phenotype_block(row)}\n\n{_QUESTION}\n\n{_ANSWER_LINE}"
