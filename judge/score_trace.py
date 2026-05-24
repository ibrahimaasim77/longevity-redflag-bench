"""Reasoning-trace scorer for Longevity-LLM benchmark (extra credit).

Scores a model's <think> trace on biological correctness — not just final answer.
Designed to be: automatable, hard to hack, grounded in real databases (MGI).

Three programmatic checks (no API needed, instant, free):
  1. Gene hallucination:  does every gene mentioned actually exist in MGI?
  2. Think/answer consistency: does the reasoning contradict the final answer?
  3. System grounding: are claimed pathway/system memberships real?

One Claude-verified check (cheap, ~$0.001/call):
  4. Pathway claim verification: ask Haiku to verify specific biological claims
     against the gene's known MGI annotations.

Output: trace_score 0.0–1.0 per trace, with itemized sub-scores.

Usage:
    python judge/score_trace.py traces.jsonl --out judge/trace_scores.jsonl
    python judge/score_trace.py traces.jsonl --no-api   # skip Claude, programmatic only

Each line of traces.jsonl must have: {"trace": "...", "answer": "A/B", "genes_in_prompt": [...]}
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

JUDGE_DIR = Path(__file__).parent
REPO_ROOT = JUDGE_DIR.parent

# ── Load MGI ground truth (extracted from Anderson's mgi_labeled.csv) ─────────

def _load_gene_set(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    return {line.strip().lower() for line in path.read_text().splitlines() if line.strip()}


def _load_gene_systems(path: Path) -> Dict[str, Set[str]]:
    if not path.exists():
        return {}
    mapping = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            gene = row["gene"].strip().lower()
            systems = {s.strip().lower() for s in row["systems"].split("|") if s.strip()}
            mapping[gene] = systems
    return mapping


def _load_famous_genes(path: Path) -> Set[str]:
    if not path.exists():
        return set()
    genes = set()
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            genes.add(row["gene_symbol"].strip().lower())
    return genes


MGI_GENES = _load_gene_set(JUDGE_DIR / "mgi_genes.txt")
GENE_SYSTEMS = _load_gene_systems(JUDGE_DIR / "gene_systems.csv")
FAMOUS_GENES = _load_famous_genes(JUDGE_DIR / "famous_genes.csv")

# Common gene symbol pattern: 1-5 letters/digits, often with numbers
# Handles: Rb1, Tp53, Braf, Sirt6, Apoe, Clint1, etc.
# Excludes single letters (A, B) and common English words
_GENE_PATTERN = re.compile(
    r"\b([A-Z][a-z0-9]{1,6}(?:[A-Z][a-z0-9]*)*)\b"  # CamelCase gene symbols (mouse)
    r"|\b([A-Z][A-Z0-9]{1,8})\b"                      # ALL-CAPS gene symbols (human-style)
)
_COMMON_WORDS = {
    "the", "and", "for", "are", "but", "not", "you", "all", "can", "had", "her",
    "was", "one", "our", "out", "has", "his", "how", "its", "may", "new", "now",
    "old", "see", "two", "way", "who", "did", "get", "let", "say", "she", "too",
    "use", "also", "been", "each", "from", "have", "here", "high", "into", "just",
    "like", "long", "make", "many", "more", "most", "much", "must", "name", "only",
    "over", "such", "take", "than", "them", "then", "this", "very", "when", "will",
    "with", "would", "could", "should", "these", "those", "their", "there", "where",
    "which", "while", "about", "after", "being", "below", "between", "both", "does",
    "down", "during", "first", "given", "known", "large", "later", "less", "likely",
    "lower", "major", "might", "never", "often", "other", "overall", "part", "poor",
    "role", "same", "show", "since", "small", "some", "still", "study", "that",
    "thus", "well", "what", "year", "based", "case", "data", "death", "early",
    "effect", "gene", "group", "human", "level", "life", "loss", "male", "model",
    "mouse", "normal", "note", "null", "percent", "rate", "risk", "result", "size",
    "stress", "system", "term", "test", "time", "type", "used", "using", "value",
    "work", "yes", "profile", "strain", "answer", "option", "predict", "survival",
    "impact", "evidence", "finding", "suggest", "indicate", "associated", "related",
    "observed", "reported", "analysis", "function", "process", "pathway", "protein",
    "cell", "tissue", "organ", "blood", "bone", "brain", "heart", "liver", "lung",
    "skin", "muscle", "kidney", "immune", "neural", "vascular", "golgi",
    "nad", "nadh", "nadph", "atp", "gtp", "dna", "rna", "mrna", "trna", "rrna", "cdna",
    "ros", "vldl", "ldl", "hdl", "glut4", "impc", "mgi", "pmid", "omim",
    "akt", "erk", "mapk", "jak", "stat", "nfkb", "tnf", "tgf", "bmp",
    "however", "therefore", "moreover", "furthermore", "although", "because",
    "niemann", "pick", "marfan", "alzheimer", "parkinson",
    "age", "aging", "aged", "young", "adult", "embryo", "born", "birth",
    "increased", "decreased", "reduced", "elevated", "impaired", "abnormal",
    "phenotype", "genotype", "allele", "mutation", "knockout", "heterozygous",
    "homozygous", "wild", "dominant", "recessive",
    "mortality", "morbidity", "lifespan", "longevity", "lethal", "viable",
}

SYSTEM_KEYWORDS = {
    "cardiovascular": "cardiovascular system",
    "cardiac": "cardiovascular system",
    "heart": "cardiovascular system",
    "vascular": "cardiovascular system",
    "immune": "immune system",
    "immunological": "immune system",
    "nervous": "nervous system",
    "neurological": "nervous system",
    "neural": "nervous system",
    "brain": "nervous system",
    "hematopoietic": "hematopoietic system",
    "blood": "hematopoietic system",
    "skeletal": "skeleton",
    "bone": "skeleton",
    "metabolic": "homeostasis/metabolism",
    "metabolism": "homeostasis/metabolism",
    "liver": "liver/biliary system",
    "hepatic": "liver/biliary system",
    "renal": "renal/urinary system",
    "kidney": "renal/urinary system",
    "respiratory": "respiratory system",
    "lung": "respiratory system",
    "pulmonary": "respiratory system",
    "endocrine": "endocrine/exocrine gland",
    "thyroid": "endocrine/exocrine gland",
    "digestive": "digestive/alimentary",
    "intestinal": "digestive/alimentary",
}


# ── Check 1: Gene hallucination ──────────────────────────────────────────────

def extract_gene_symbols(text: str) -> List[str]:
    """Extract candidate gene symbols from a reasoning trace."""
    candidates = []
    for m in _GENE_PATTERN.finditer(text):
        symbol = m.group(1) or m.group(2)
        if symbol and symbol.lower() not in _COMMON_WORDS and len(symbol) >= 2:
            candidates.append(symbol)
    return list(dict.fromkeys(candidates))


def check_gene_hallucination(trace: str, genes_in_prompt: List[str]) -> Dict:
    """Verify every gene mentioned in the trace exists in MGI."""
    mentioned = extract_gene_symbols(trace)
    prompt_genes = {g.lower() for g in (genes_in_prompt or [])}

    real, hallucinated, famous = [], [], []
    for g in mentioned:
        gl = g.lower()
        if gl in prompt_genes:
            continue
        if gl in MGI_GENES:
            real.append(g)
            if gl in FAMOUS_GENES:
                famous.append(g)
        else:
            hallucinated.append(g)

    n_checked = len(real) + len(hallucinated)
    score = 1.0 if n_checked == 0 else len(real) / n_checked

    return {
        "score": round(score, 3),
        "mentioned_genes": mentioned,
        "real_genes": real,
        "hallucinated_genes": hallucinated,
        "famous_genes_cited": famous,
        "n_checked": n_checked,
    }


# ── Check 2: Think/answer consistency ────────────────────────────────────────

_NEGATION = re.compile(r"\b(not|no|does not|doesn.t|do not|without|never)\b", re.I)

_IMPAIRS_PATTERN = re.compile(
    r"\b(impairs?|shortens?|reduces?|decreases?)\b.{0,30}\b(survival|lifespan|longevity)\b"
    r"|\b(lethal|embryonic.lethality|perinatal.lethality|premature.death)\b"
    r"|\byes\b.{0,20}\b(impairs?|affect)\b",
    re.I,
)
_NO_EFFECT_PATTERN = re.compile(
    r"\b(does not|doesn.t|do not|not)\b.{0,20}\b(impair|affect|shorten|reduce)\b"
    r"|\bnormal\s+(lifespan|survival|longevity)\b"
    r"|\bno\s+(mortality|survival|lethal).{0,15}(effect|impact|phenotype)\b"
    r"|\bno\s+(evidence).{0,20}(reduced survival|impair|shorten|lethal)"
    r"|\bnot\s+shorten\s+lifespan\b",
    re.I,
)


def check_think_answer_consistency(trace: str, final_answer: str) -> Dict:
    """Check if the <think> trace contradicts the final answer letter."""
    think_match = re.search(r"<think>(.*?)</think>", trace, re.S)
    thinking = think_match.group(1) if think_match else trace

    answer_upper = (final_answer or "").strip().upper()
    trace_says_no_effect = bool(_NO_EFFECT_PATTERN.search(thinking))

    # Check for AFFIRMATIVE impairs claims (not preceded by negation)
    trace_says_impairs = False
    for m in _IMPAIRS_PATTERN.finditer(thinking):
        start = max(0, m.start() - 25)
        prefix = thinking[start:m.start()]
        if not _NEGATION.search(prefix):
            trace_says_impairs = True
            break

    contradicts = False
    reason = "consistent"

    if answer_upper == "A" and trace_says_no_effect and not trace_says_impairs:
        contradicts = True
        reason = "answer=impairs but trace argues no effect"
    elif answer_upper == "B" and trace_says_impairs and not trace_says_no_effect:
        contradicts = True
        reason = "answer=no_effect but trace argues impairs survival"

    return {
        "score": 0.0 if contradicts else 1.0,
        "contradicts": contradicts,
        "reason": reason,
        "trace_says_impairs": trace_says_impairs,
        "trace_says_no_effect": trace_says_no_effect,
        "final_answer": answer_upper,
    }


# ── Check 3: System/pathway grounding ────────────────────────────────────────

def check_system_grounding(trace: str, genes_in_prompt: List[str]) -> Dict:
    """Verify that system/pathway claims match MGI annotations for the gene."""
    claimed_systems = set()
    for keyword, system in SYSTEM_KEYWORDS.items():
        pattern = re.compile(
            rf"\b{re.escape(keyword)}\b.{{0,60}}\b(pathway|system|function|role|involvement|effect)\b"
            rf"|\b(pathway|system|function|role|involvement|effect)\b.{{0,60}}\b{re.escape(keyword)}\b",
            re.I,
        )
        if pattern.search(trace):
            claimed_systems.add(system)

    if not claimed_systems or not genes_in_prompt:
        return {"score": 1.0, "claimed_systems": list(claimed_systems),
                "verified": [], "unverified": [], "n_checked": 0}

    known_systems = set()
    for g in genes_in_prompt:
        gl = g.lower()
        if gl in GENE_SYSTEMS:
            known_systems |= GENE_SYSTEMS[gl]

    if not known_systems:
        return {"score": 1.0, "claimed_systems": list(claimed_systems),
                "verified": [], "unverified": [], "n_checked": 0}

    verified = [s for s in claimed_systems if s in known_systems]
    unverified = [s for s in claimed_systems if s not in known_systems]
    n = len(claimed_systems)
    score = len(verified) / n if n > 0 else 1.0

    return {
        "score": round(score, 3),
        "claimed_systems": list(claimed_systems),
        "verified": verified,
        "unverified": unverified,
        "n_checked": n,
    }


# ── Check 4: Claude-verified pathway claims (optional, ~$0.001/call) ─────────

def verify_with_claude(trace: str, genes_in_prompt: List[str],
                       gene_systems: Dict[str, Set[str]]) -> Dict:
    """Use Claude Haiku to verify specific biological claims in the trace."""
    try:
        from anthropic import Anthropic
    except ImportError:
        return {"score": None, "error": "anthropic not installed", "skipped": True}

    sys_path = REPO_ROOT / "src"
    if str(sys_path) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from src import config

    api_key = config.ANTHROPIC_API_KEY
    if not api_key:
        return {"score": None, "error": "ANTHROPIC_API_KEY not set", "skipped": True}

    gene_context = ""
    for g in (genes_in_prompt or []):
        gl = g.lower()
        if gl in GENE_SYSTEMS:
            gene_context += f"\n  {g}: known systems = {', '.join(sorted(GENE_SYSTEMS[gl]))}"

    prompt = (
        "You are a mouse genetics expert. A language model produced the reasoning trace below "
        "about a mouse genotype's effect on survival. Score its biological accuracy.\n\n"
        f"TRACE:\n{trace[:2000]}\n\n"
        f"GENES IN THE PROMPT: {', '.join(genes_in_prompt or ['none'])}\n"
        f"KNOWN MGI ANNOTATIONS:{gene_context or ' none available'}\n\n"
        "Score 0.0-1.0 on biological accuracy. Deduct for:\n"
        "- Mentioning genes not in the prompt without justification (-0.2 each)\n"
        "- Wrong pathway/system attribution (-0.2 each)\n"
        "- Fabricated mechanisms or citations (-0.3 each)\n"
        "- Contradicting known MGI annotations (-0.3)\n"
        "Return ONLY JSON: {\"score\": 0.0-1.0, \"deductions\": [\"reason1\", ...], \"notes\": \"<20 words\"}"
    )

    try:
        client = Anthropic(api_key=api_key)
        msg = client.messages.create(
            model=config.JUDGE_MODEL,
            max_tokens=300,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = msg.content[0].text
        m = re.search(r"\{.*\}", text, re.S)
        data = json.loads(m.group(0)) if m else {"score": 0.5}
        return {
            "score": round(float(data.get("score", 0.5)), 3),
            "deductions": data.get("deductions", []),
            "notes": data.get("notes", ""),
            "skipped": False,
        }
    except Exception as e:
        return {"score": None, "error": str(e), "skipped": True}


# ── Composite scorer ─────────────────────────────────────────────────────────

WEIGHTS = {
    "gene_hallucination": 0.30,
    "think_answer_consistency": 0.25,
    "system_grounding": 0.20,
    "claude_verification": 0.25,
}

WEIGHTS_NO_API = {
    "gene_hallucination": 0.40,
    "think_answer_consistency": 0.35,
    "system_grounding": 0.25,
}


def score_trace(trace: str, final_answer: str, genes_in_prompt: List[str],
                use_api: bool = True) -> Dict:
    """Score a single reasoning trace. Returns 0.0–1.0 composite + sub-scores."""

    gene_result = check_gene_hallucination(trace, genes_in_prompt)
    consistency_result = check_think_answer_consistency(trace, final_answer)
    system_result = check_system_grounding(trace, genes_in_prompt)

    sub_scores = {
        "gene_hallucination": gene_result,
        "think_answer_consistency": consistency_result,
        "system_grounding": system_result,
    }

    if use_api:
        claude_result = verify_with_claude(trace, genes_in_prompt, GENE_SYSTEMS)
        sub_scores["claude_verification"] = claude_result

        if claude_result.get("skipped"):
            weights = WEIGHTS_NO_API
        else:
            weights = WEIGHTS
    else:
        weights = WEIGHTS_NO_API

    composite = 0.0
    for key, w in weights.items():
        s = sub_scores.get(key, {}).get("score")
        if s is not None:
            composite += w * s

    return {
        "trace_score": round(composite, 3),
        "sub_scores": sub_scores,
        "weights_used": weights,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Score LLM reasoning traces for biological correctness")
    parser.add_argument("input", help="JSONL file with traces (fields: trace, answer, genes_in_prompt)")
    parser.add_argument("--out", "-o", default=None, help="Output JSONL path (default: stdout)")
    parser.add_argument("--no-api", action="store_true", help="Skip Claude API calls (programmatic checks only)")
    args = parser.parse_args()

    results = []
    with open(args.input) as f:
        for i, line in enumerate(f):
            row = json.loads(line)
            trace = row.get("trace", row.get("reasoning", ""))
            answer = row.get("answer", row.get("final_answer", ""))
            genes = row.get("genes_in_prompt", row.get("genes", []))

            result = score_trace(trace, answer, genes, use_api=not args.no_api)
            result["item_index"] = i
            result["item_id"] = row.get("item_id", f"item-{i}")
            results.append(result)

            # Progress
            print(f"  [{i}] {result['item_id']}: trace_score={result['trace_score']:.3f}", file=sys.stderr)

    # Summary
    scores = [r["trace_score"] for r in results]
    mean_score = sum(scores) / len(scores) if scores else 0
    print(f"\n  MEAN TRACE SCORE: {mean_score:.3f} ({len(scores)} traces)", file=sys.stderr)

    # Output
    out = open(args.out, "w") if args.out else sys.stdout
    for r in results:
        out.write(json.dumps(r) + "\n")
    if args.out:
        out.close()
        print(f"  Saved to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
