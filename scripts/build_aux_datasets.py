"""Build the auxiliary database CSVs that sit alongside data/mgi_genotype_phenotype.csv:
  - data/impc_viability.csv        IMPC homozygous lethal/subviable calls (CORE source #2)
  - data/mp_mortality_terms.csv    MP mortality/aging label vocabulary (128-term subtree)
  - data/famous_gene_blocklist.csv contamination-control blocklist (GenAge best-effort + starter)

Each section is independent (try/except) so one failure doesn't sink the others.
Usage: python scripts/build_aux_datasets.py [mp.obo path]
"""

import csv
import io
import json
import os
import sys
import urllib.request
import zipfile
from collections import defaultdict

REPO = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA = os.path.join(REPO, "data")
OBO = sys.argv[1] if len(sys.argv) > 1 else "/tmp/mp.obo"
os.makedirs(DATA, exist_ok=True)


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={"User-Agent": "longevity-bench/0.1"})
    return urllib.request.urlopen(req, timeout=timeout).read()


# --- 1. IMPC viability (lethal + subviable) via Solr -------------------------- #
def build_impc():
    url = ("https://www.ebi.ac.uk/mi/impc/solr/genotype-phenotype/select"
           "?q=(mp_term_name:*lethal*%20OR%20mp_term_name:*subviable*)"
           "&rows=4000&wt=json&fl=marker_symbol,allele_symbol,zygosity,mp_term_name")
    d = json.loads(get(url))
    docs = d["response"]["docs"]
    out = os.path.join(DATA, "impc_viability.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["marker_symbol", "allele_symbol", "zygosity", "viability_outcome", "impairs_survival"])
        for x in docs:
            w.writerow([x.get("marker_symbol", ""), x.get("allele_symbol", ""),
                        x.get("zygosity", ""), x.get("mp_term_name", ""), 1])
    print(f"impc_viability.csv: {len(docs)} lethal/subviable calls (of {d['response']['numFound']} total)")


# --- 2. MP mortality/aging label vocabulary ----------------------------------- #
def build_mp_terms():
    name, isa, cur = {}, defaultdict(list), None
    for line in open(OBO, encoding="utf-8", errors="ignore"):
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
    for c, ps in isa.items():
        for p in ps:
            children[p].append(c)
    seen, stack = set(), ["MP:0010768"]
    while stack:
        x = stack.pop()
        for c in children.get(x, []):
            if c not in seen:
                seen.add(c)
                stack.append(c)
    out = os.path.join(DATA, "mp_mortality_terms.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["mp_id", "name"])
        for mp in sorted(seen):
            w.writerow([mp, name.get(mp, "")])
    print(f"mp_mortality_terms.csv: {len(seen)} terms under mortality/aging (MP:0010768)")


# --- 3. Famous-gene blocklist (GenAge best-effort + curated starter) ---------- #
STARTER = ["Sirt1", "Sirt3", "Sirt6", "Sirt7", "Mtor", "Igf1", "Igf1r", "Ghr", "Gh",
           "Pou1f1", "Prop1", "Kl", "Foxo1", "Foxo3", "Trp53", "Tert", "Terc", "Lmna",
           "Sod1", "Sod2", "Cat", "Insr", "Irs1", "Irs2", "Akt1", "Pten", "Tsc1", "Tsc2",
           "Rps6kb1", "Atg5", "Atg7", "Nampt", "Nfe2l2", "Prkaa1", "Prkaa2", "Ucp2",
           "Coq7", "Shc1", "Ercc1", "Ercc2", "Wrn", "Bub1b", "Cdkn2a", "Cdkn1a", "Myc"]


def build_blocklist():
    genes, source = set(STARTER), "curated-starter"
    try:
        raw = get("https://genomics.senescence.info/genes/models_genes.zip", timeout=40)
        z = zipfile.ZipFile(io.BytesIO(raw))
        fn = next(n for n in z.namelist() if n.endswith(".csv"))
        rows = z.read(fn).decode("utf-8", "ignore").splitlines()
        rdr = csv.DictReader(rows)
        col = next((c for c in (rdr.fieldnames or []) if "symbol" in c.lower() or "gene" in c.lower()), None)
        if col:
            for r in rdr:
                if r.get(col):
                    genes.add(r[col].strip())
            source = "genage_models + starter"
    except Exception as e:  # noqa: BLE001
        print(f"  (GenAge download failed: {e}; using curated starter only)")
    out = os.path.join(DATA, "famous_gene_blocklist.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["gene_symbol", "source"])
        for g in sorted(genes):
            w.writerow([g, source])
    print(f"famous_gene_blocklist.csv: {len(genes)} genes ({source})")


if __name__ == "__main__":
    for fn in (build_impc, build_mp_terms, build_blocklist):
        try:
            fn()
        except Exception as e:  # noqa: BLE001
            print(f"  ! {fn.__name__} failed: {e}")
