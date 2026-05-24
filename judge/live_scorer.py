#!/usr/bin/env python3
"""Live end-to-end demo: Query Longevity-LLM → Score trace → Claude analyzes it.

Flow:
  1. Send a mouse genotype prompt to Longevity-LLM (the model under test)
  2. Get back the reasoning trace + answer
  3. Run programmatic checks (gene hallucination, consistency, pathway)
  4. Send trace to Claude Haiku for biological verification
  5. Display everything in real time with colors

Usage:
    python judge/live_scorer.py                  # uses 5 pre-selected genotypes
    python judge/live_scorer.py --use-cached     # skip LLM call, use existing traces

Requires:
    - MODEL_ACCESS_TOKEN in .env (for Longevity-LLM endpoint)
    - ANTHROPIC_API_KEY in .env (for Claude Haiku analysis)
"""

import json, sys, time, os
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from src import config
from score_trace import score_trace, MGI_GENES, GENE_SYSTEMS

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"
MAGENTA = "\033[95m"

# ── 5 demo genotypes (from the controlled eval set) ──────────────────────────

DEMO_CASES = [
    {
        "genotype_id": "MGI:2669271",
        "gene": "Proc",
        "description": "Procollagen C-protein peptidase knockout, homozygous",
        "phenotypes": "liver fibrosis, collagen defects, hemorrhage, abnormal blood coagulation, thrombosis",
        "gold": "A",
        "condition": "geno_pheno",
    },
    {
        "genotype_id": "MGI:2680724",
        "gene": "Osmr",
        "description": "Oncostatin M receptor knockout, homozygous",
        "phenotypes": "insulin resistance, hepatic steatosis, chronic inflammation, elevated triglycerides",
        "gold": "B",
        "condition": "geno_pheno",
    },
    {
        "genotype_id": "MGI:6408315",
        "gene": "Cry1",
        "description": "Cryptochrome 1 knockout, homozygous",
        "phenotypes": "increased leukocyte count, decreased serum albumin",
        "gold": "B",
        "condition": "geno_pheno",
    },
    {
        "genotype_id": "MGI:5451030",
        "gene": "Ndufs4",
        "description": "NADH dehydrogenase subunit knockout, homozygous",
        "phenotypes": "weight loss, impaired coordination, respiratory distress, microgliosis",
        "gold": "A",
        "condition": "geno_pheno",
    },
    {
        "genotype_id": "MGI:3583329",
        "gene": "Ret",
        "description": "Ret proto-oncogene knockout, homozygous",
        "phenotypes": "absent kidney, abnormal intestinal peristalsis",
        "gold": "A",
        "condition": "geno_pheno",
    },
]


def build_prompt(case):
    """Build the benchmark prompt for Longevity-LLM."""
    if case["condition"] == "geno_pheno":
        genotype_block = (
            f"Genotype: {case['gene']} knockout ({case['description']})\n"
        )
    else:
        genotype_block = "Genotype: [HIDDEN]\n"

    return (
        "You are a biomedical AI specialized in aging biology.\n\n"
        f"{genotype_block}"
        f"Phenotype profile: {case['phenotypes']}\n\n"
        "Does this genotype impair survival (shorten lifespan or cause lethality)?\n"
        "Options: A. Yes, impairs survival  B. No, does not impair survival\n\n"
        "Think step by step, then answer with the option letter.\n"
        "Answer: "
    )


def call_longevity_llm(prompt):
    """Call the Longevity-LLM endpoint."""
    try:
        from openai import OpenAI
    except ImportError:
        return None, "openai package not installed"

    token = config.MODEL_ACCESS_TOKEN
    if not token:
        return None, "MODEL_ACCESS_TOKEN not set in .env"

    try:
        client = OpenAI(
            base_url=config.LONGEVITY_BASE_URL,
            api_key=token,
        )
        start = time.time()
        response = client.chat.completions.create(
            model=config.LONGEVITY_MODEL,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=400,
            temperature=0,
        )
        latency = time.time() - start
        text = response.choices[0].message.content
        return {"text": text, "latency": latency}, None
    except Exception as e:
        return None, str(e)


def call_claude_analysis(trace, gene, phenotypes, pred, gold):
    """Ask Claude Haiku to analyze the biological accuracy of the trace."""
    try:
        from anthropic import Anthropic
    except ImportError:
        return None, "anthropic package not installed"

    api_key = config.ANTHROPIC_API_KEY
    if not api_key:
        return None, "ANTHROPIC_API_KEY not set in .env"

    gene_systems = GENE_SYSTEMS.get(gene.lower(), set())
    known_systems = ", ".join(sorted(gene_systems)) if gene_systems else "unknown"

    prompt = (
        f"You are a mouse genetics expert reviewing an LLM's reasoning about whether "
        f"a gene knockout impairs survival.\n\n"
        f"GENE: {gene}\n"
        f"KNOWN MGI SYSTEMS: {known_systems}\n"
        f"PHENOTYPES IN PROMPT: {phenotypes}\n"
        f"GOLD ANSWER: {'A (impairs survival)' if gold == 'A' else 'B (does not impair)'}\n"
        f"MODEL PREDICTED: {'A (impairs survival)' if pred == 'A' else 'B (does not impair)'}\n\n"
        f"MODEL'S REASONING:\n{trace[:1500]}\n\n"
        f"Score the reasoning 0.0-1.0 on biological accuracy. Check:\n"
        f"1. Are the biological mechanisms described real and correct?\n"
        f"2. Are gene-pathway attributions accurate?\n"
        f"3. Is the logic connecting phenotype → survival sound?\n"
        f"4. Any fabricated facts or hallucinated interactions?\n\n"
        f"Return ONLY JSON:\n"
        f'{{"score": 0.0-1.0, "biological_accuracy": "1-sentence verdict", '
        f'"fabrications": ["list any invented facts"], '
        f'"correct_claims": ["list verified claims"]}}'
    )

    try:
        client = Anthropic(api_key=api_key)
        start = time.time()
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=300,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.time() - start
        text = msg.content[0].text
        import re
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            data = json.loads(m.group(0))
            data["latency"] = latency
            return data, None
        return {"score": 0.5, "biological_accuracy": text[:100], "latency": latency}, None
    except Exception as e:
        return None, str(e)


def extract_answer(text):
    """Extract A or B from model output."""
    import re
    region = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    m = re.search(r"\b(?:answer|option)\s*[:=]?\s*\(?([AB])\)?", region, re.I)
    if m:
        return m.group(1).upper()
    m = re.search(r"\b([AB])\b", region[-50:])
    return m.group(1).upper() if m else "?"


def print_bar(score, width=25):
    filled = int(score * width)
    color = GREEN if score >= 0.8 else YELLOW if score >= 0.5 else RED
    return f"{color}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET} {color}{score:.3f}{RESET}"


def load_cached_traces():
    """Load existing traces from eval data."""
    path = ROOT / "data" / "eval_controlled_240.jsonl"
    if not path.exists():
        return {}
    traces = {}
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            key = f"{r['genotype_id']}_{r['condition']}"
            traces[key] = r.get("raw", "")
    return traces


def main():
    use_cached = "--use-cached" in sys.argv

    print(f"\n{BOLD}{CYAN}{'═' * 65}{RESET}")
    print(f"{BOLD}{CYAN}  LIVE REASONING SCORER{RESET}")
    print(f"{BOLD}{CYAN}  Longevity-LLM → Programmatic Checks → Claude Analysis{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 65}{RESET}")
    print(f"{DIM}  Step 1: Query Longevity-LLM for reasoning trace{RESET}")
    print(f"{DIM}  Step 2: Run 3 programmatic checks (gene, consistency, pathway){RESET}")
    print(f"{DIM}  Step 3: Claude Haiku verifies biological accuracy{RESET}\n")

    cached = load_cached_traces()

    for i, case in enumerate(DEMO_CASES, 1):
        print(f"{BOLD}{CYAN}{'─' * 65}{RESET}")
        print(f"{BOLD}{CYAN}  [{i}/5] {case['gene']} — {case['description']}{RESET}")
        print(f"{DIM}  Gold: {'A (impairs survival)' if case['gold'] == 'A' else 'B (no effect)'} | Condition: {case['condition']}{RESET}")
        print()

        # ── Step 1: Get trace from Longevity-LLM ──
        trace = None
        pred = None

        if use_cached:
            key = f"{case['genotype_id']}_{case['condition']}"
            trace = cached.get(key, "")
            if trace:
                pred = extract_answer(trace)
                print(f"  {DIM}[cached trace, {len(trace)} chars]{RESET}")
        else:
            print(f"  {YELLOW}Querying Longevity-LLM...{RESET}", end=" ", flush=True)
            prompt = build_prompt(case)
            result, err = call_longevity_llm(prompt)
            if result:
                trace = result["text"]
                pred = extract_answer(trace)
                print(f"{GREEN}✓{RESET} ({result['latency']:.1f}s)")
            else:
                print(f"{RED}✗ {err}{RESET}")
                key = f"{case['genotype_id']}_{case['condition']}"
                trace = cached.get(key, "")
                if trace:
                    pred = extract_answer(trace)
                    print(f"  {DIM}[falling back to cached trace]{RESET}")
                else:
                    print(f"  {RED}No cached trace available, skipping{RESET}\n")
                    continue

        if not trace:
            print(f"  {RED}No trace available{RESET}\n")
            continue

        # Show prediction
        correct = pred == case["gold"]
        status = f"{GREEN}✓ CORRECT{RESET}" if correct else f"{RED}✗ WRONG{RESET}"
        print(f"  Predicted: {BOLD}{pred}{RESET} {status}")
        print(f"  {DIM}Trace: \"{trace[:100]}...\"{RESET}")
        print()

        # ── Step 2: Programmatic checks ──
        print(f"  {YELLOW}Running programmatic checks...{RESET}")
        prog_result = score_trace(trace, pred, [case["gene"]], use_api=False)

        print(f"  {BOLD}Programmatic Score:{RESET} {print_bar(prog_result['trace_score'])}")
        for key, sub in prog_result["sub_scores"].items():
            label = key.replace("_", " ").title()
            s = sub.get("score")
            if s is None:
                continue
            print(f"    {label:28s} {print_bar(s)}")
            if sub.get("hallucinated_genes"):
                print(f"      {RED}⚠ Hallucinated: {', '.join(sub['hallucinated_genes'][:3])}{RESET}")
            if sub.get("contradicts"):
                print(f"      {RED}✗ {sub['reason']}{RESET}")
            if sub.get("unverified"):
                print(f"      {RED}✗ Ungrounded: {', '.join(sub['unverified'])}{RESET}")
        print()

        # ── Step 3: Claude analysis ──
        print(f"  {MAGENTA}Asking Claude Haiku to verify biology...{RESET}", end=" ", flush=True)
        claude_result, err = call_claude_analysis(
            trace, case["gene"], case["phenotypes"], pred, case["gold"]
        )

        if claude_result:
            print(f"{GREEN}✓{RESET} ({claude_result.get('latency', 0):.1f}s)")
            cs = claude_result.get("score", 0.5)
            print(f"  {BOLD}Claude Score:{RESET}       {print_bar(cs)}")
            print(f"  {BOLD}Verdict:{RESET}           {claude_result.get('biological_accuracy', 'N/A')}")
            if claude_result.get("fabrications"):
                print(f"  {RED}Fabrications:{RESET}      {', '.join(claude_result['fabrications'][:2])}")
            if claude_result.get("correct_claims"):
                print(f"  {GREEN}Verified claims:{RESET}  {', '.join(claude_result['correct_claims'][:2])}")
        else:
            print(f"{RED}✗ {err}{RESET}")
            print(f"  {DIM}(Add ANTHROPIC_API_KEY to .env to enable Claude verification){RESET}")

        # ── Combined score ──
        if claude_result and claude_result.get("score") is not None:
            combined = prog_result["trace_score"] * 0.6 + claude_result["score"] * 0.4
            print(f"\n  {BOLD}Combined Score:{RESET}    {print_bar(combined)}")
            print(f"  {DIM}(60% programmatic + 40% Claude){RESET}")

        print()

    print(f"{BOLD}{CYAN}{'═' * 65}{RESET}")
    print(f"  {BOLD}Demo complete.{RESET}")
    print(f"  {DIM}Programmatic checks: $0.00 | Claude calls: ~$0.005 total{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 65}{RESET}\n")


if __name__ == "__main__":
    main()
