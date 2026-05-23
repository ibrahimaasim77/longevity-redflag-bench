"""Build the MGI genotype+phenotype -> survival SOURCE CSV for the mouse-longevity benchmark.

Groups MGI_PhenoGenoMP.rpt by genotype; splits each genotype's MP terms into the
mortality/aging branch (MP:0010768 subtree = the LABEL) vs the rest (the phenotype
profile we show the model); derives zygosity from the allelic composition. Each
genotype's engineered alleles are joined to MGI_PhenotypicAllele.rpt to derive an
`expression_direction` (decreased / increased / altered / none / mixed / unknown)
from the curated allele attributes -- MGI has no direct "expression level" field, so
this is the engineered loss-/gain-of-function direction inferred from the allele.

Inputs (default /tmp; pass paths as argv):
  - mp.obo                    (MP ontology; descendants of MP:0010768)
  - MGI_PhenoGenoMP.rpt       (genotype -> MP annotations)
  - MGI_PhenotypicAllele.rpt  (allele -> curated attributes; OPTIONAL but recommended)
Output: data/mgi_genotype_phenotype.csv  (one row per genotype)

Usage: python scripts/build_mgi_dataset.py [mgi.rpt] [mp.obo] [allele.rpt]
"""

import csv
import os
import re
import sys
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MGI = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mgi.rpt"
OBO = sys.argv[2] if len(sys.argv) > 2 else "/tmp/mp.obo"
ALLELE = sys.argv[3] if len(sys.argv) > 3 else os.path.join(REPO, "data", "MGI_PhenotypicAllele.rpt")
OUT = os.path.join(REPO, "data", "mgi_genotype_phenotype.csv")
MORTALITY_ROOT = "MP:0010768"  # mortality/aging

# --- Allele-attribute -> expression-direction mapping ----------------------------
# MGI attribute vocab (MGI_PhenotypicAllele.rpt col 4, '|'-delimited). Direction is
# the *engineered functional* direction, the closest proxy MGI offers to expression
# up/down -- there is no literal mRNA/protein-level field.
ATTR_DECREASED = {"Null/knockout", "Hypomorph", "Knockdown", "Inducible degradation"}
ATTR_INCREASED = {"Constitutively active"}  # the one unambiguous "more" attribute
# functional change that isn't a clean over/under-expression call:
ATTR_ALTERED = {"Dominant negative", "Modified regulatory region",
                "Modified isoform(s)", "Altered localization"}
# tool / labeling / no-effect attributes -> "none" (reporter, cre, tags, floxed-but-uncut):
ATTR_NONE = {"No functional change", "Reporter", "Recombinase", "Transactivator",
             "Epitope tag", "Conditional ready", "Inducible", "RMCE-ready",
             "Transposon concatemer", "Endonuclease", "Transposase",
             "Humanized sequence", "Lineage barcode", "Inserted expressed sequence"}
ATTR_UNKNOWN = {"", "Not Specified", "Not Applicable"}


def load_mp(obo):
    """Return (id->name, set(descendants of MORTALITY_ROOT))."""
    name, isa, cur = {}, defaultdict(list), None
    for line in open(obo, encoding="utf-8", errors="ignore"):
        line = line.rstrip()
        if line == "[Term]":
            cur = {"id": None}
        elif cur is not None and line.startswith("id: MP:"):
            cur["id"] = line[4:]
        elif cur is not None and line.startswith("name:"):
            name[cur["id"]] = line[6:]
        elif cur is not None and line.startswith("is_a: MP:"):
            isa[cur["id"]].append(line[6:].split("!")[0].strip())
    children = defaultdict(list)
    for c, parents in isa.items():
        for p in parents:
            children[p].append(c)
    seen, stack = set(), [MORTALITY_ROOT]
    while stack:
        x = stack.pop()
        for c in children.get(x, []):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    return name, seen


def zygosity(allelic_comp):
    parts = [p.strip() for p in allelic_comp.split("/")]
    if "," in allelic_comp:
        return "multi-locus"
    if len(parts) == 2:
        if parts[0] == parts[1]:
            return "homozygote"
        if "<+>" in parts[1] or parts[1] in ("+", ""):
            return "heterozygote"
        if parts[1] in ("Y", "0", "-"):
            return "hemizygote"
        return "compound/other"
    return "other"


def genes_from(allelic_comp):
    # gene symbol is the text before each "<allele>"
    return sorted(set(re.findall(r"([A-Za-z0-9_.\-]+)<", allelic_comp)))


def load_allele_attrs(path):
    """allele_symbol -> (frozenset(attributes), allele_type). Empty dict if missing."""
    amap = {}
    if not os.path.exists(path):
        return amap
    with open(path, encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.startswith("#"):
                continue
            c = line.rstrip("\n").split("\t")
            if len(c) <= 4:
                continue
            sym, atype, attr = c[1], c[3], c[4]
            attrs = frozenset(a.strip() for a in attr.split("|") if a.strip())
            amap[sym] = (attrs, atype)
    return amap


def parse_alleles(allelic_comp):
    """Split a composition into allele symbols, splitting on '/' and ',' ONLY at
    paren/angle-bracket depth 0 (allele symbols themselves contain '/' and ',',
    e.g. <tm1(cre/ERT2)Crm> or <C3H/HeJ>). Drops wild-type (<+>, +) and Y/0/- slots."""
    out, buf, par, ang = [], "", 0, 0
    for ch in allelic_comp:
        if ch == "(":
            par += 1; buf += ch
        elif ch == ")":
            par -= 1; buf += ch
        elif ch == "<":
            ang += 1; buf += ch
        elif ch == ">":
            ang -= 1; buf += ch
        elif ch in "/," and par == 0 and ang == 0:
            out.append(buf); buf = ""
        else:
            buf += ch
    out.append(buf)
    res = []
    for a in (x.strip() for x in out):
        if not a or a in ("+", "Y", "0", "-", "?") or a.endswith("<+>"):
            continue
        res.append(a)
    return res


def _allele_direction(attrs, atype):
    """One allele's expression direction from its attributes. Loss dominates a
    knockout-reporter; transgene that inserts a non-tool expressed sequence -> increased."""
    if attrs & ATTR_DECREASED:
        return "decreased"
    if attrs & ATTR_INCREASED:
        return "increased"
    # transgene-overexpression heuristic: a transgenic allele carrying an inserted
    # expressed sequence whose purpose isn't purely labeling/driving (no Reporter/
    # Recombinase/Transactivator-only). Approximate, by design -- see module docstring.
    if (atype == "Transgenic" and "Inserted expressed sequence" in attrs
            and not (attrs & {"Reporter", "Recombinase", "Transactivator"})):
        return "increased"
    if attrs & ATTR_ALTERED:
        return "altered"
    if attrs - ATTR_UNKNOWN:
        return "none"
    return "unknown"


def expression_direction(allelic_comp, amap):
    """Aggregate per-allele directions across a genotype.
    Returns (direction, per_allele_detail_string)."""
    dirs, detail = set(), []
    for sym in parse_alleles(allelic_comp):
        entry = amap.get(sym)
        if entry is None:
            dirs.add("unknown")
            continue
        attrs, atype = entry
        d = _allele_direction(attrs, atype)
        dirs.add(d)
        detail.append(f"{sym}={'+'.join(sorted(attrs)) or '?'}")
    signal = dirs - {"none", "unknown"}
    if "decreased" in signal and "increased" in signal:
        direction = "mixed"
    elif "decreased" in signal:
        direction = "decreased"
    elif "increased" in signal:
        direction = "increased"
    elif "altered" in signal:
        direction = "altered"
    elif "none" in dirs:
        direction = "none"
    else:
        direction = "unknown"
    return direction, " ; ".join(detail)


def main():
    if not os.path.exists(MGI) or not os.path.exists(OBO):
        sys.exit(f"missing input(s): {MGI} / {OBO}")
    id2name, mortality = load_mp(OBO)
    amap = load_allele_attrs(ALLELE)
    if not amap:
        print(f"WARNING: allele file not found ({ALLELE}); "
              f"expression_direction will be 'unknown' for all rows.")

    geno = {}  # genotype_acc -> dict
    with open(MGI, encoding="utf-8", errors="ignore") as f:
        for line in f:
            cols = line.rstrip("\n").split("\t")
            if len(cols) < 5:
                continue  # skip header/partial last line
            allelic, _allele, background, mp, pmid = cols[0], cols[1], cols[2], cols[3], cols[4]
            gacc = cols[6] if len(cols) > 6 and cols[6] else allelic + "|" + background
            if not mp.startswith("MP:"):
                continue
            g = geno.setdefault(gacc, {"allelic": allelic, "background": background,
                                       "mp": set(), "pmids": set()})
            g["mp"].add(mp)
            if pmid:
                g["pmids"].add(pmid)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    n_pos = n_neg = n_usable = 0
    dir_counts = defaultdict(int)
    with open(OUT, "w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(["genotype_id", "allelic_composition", "gene_symbols", "zygosity",
                    "expression_direction", "allele_attributes",
                    "genetic_background", "label_impairs_survival", "mortality_terms",
                    "n_phenotype_terms", "phenotype_terms", "pmids"])
        for gacc, g in geno.items():
            mort = sorted(g["mp"] & mortality)
            pheno = sorted(g["mp"] - mortality)
            label = 1 if mort else 0
            n_pos += label
            n_neg += (1 - label)
            if pheno:
                n_usable += 1
            direction, attr_detail = expression_direction(g["allelic"], amap)
            dir_counts[direction] += 1
            w.writerow([
                gacc, g["allelic"], "|".join(genes_from(g["allelic"])), zygosity(g["allelic"]),
                direction, attr_detail,
                g["background"], label,
                "|".join(id2name.get(m, m) for m in mort),
                len(pheno),
                "|".join(id2name.get(p, p) for p in pheno),
                "|".join(sorted(g["pmids"])),
            ])

    print(f"genotypes: {len(geno)}")
    print(f"  impairs-survival (label=1): {n_pos}")
    print(f"  no-mortality (label=0):     {n_neg}")
    print(f"  with >=1 phenotype term (usable for genotype+phenotype task): {n_usable}")
    print(f"  mortality/aging MP terms in ontology subtree: {len(mortality)}")
    print(f"  allele attributes loaded: {len(amap)}")
    print("  expression_direction distribution:")
    for d in ("decreased", "increased", "altered", "mixed", "none", "unknown"):
        print(f"    {d:10} {dir_counts.get(d, 0)}")
    print(f"wrote -> {OUT}")


if __name__ == "__main__":
    main()
