#!/usr/bin/env python3
"""Live demo: run the trace scorer on 5 curated examples in front of judges.

Usage:  python judge/demo_trace_scorer.py

No API key needed. Runs in ~2 seconds. Shows colored terminal output.
Each example highlights a different failure mode the scorer catches.
"""

import sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from score_trace import score_trace

GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
BOLD = "\033[1m"
DIM = "\033[2m"
RESET = "\033[0m"

EXAMPLES = [
    {
        "title": "GOOD TRACE — Rb1 knockout (famous gene, correct reasoning)",
        "trace": (
            "<think>\n"
            "Rb1 is a well-characterized tumor suppressor. Homozygous knockout of Rb1 "
            "causes embryonic lethality during organogenesis, with defects in erythropoiesis "
            "and neurogenesis. The phenotype profile shows liver hypoplasia, increased neuron "
            "apoptosis, and abnormal placental transport — all consistent with a lethal "
            "developmental phenotype. This genotype impairs survival.\n"
            "</think>\n"
            "Answer: A"
        ),
        "answer": "A",
        "genes": ["Rb1"],
        "expect": "HIGH score — correct biology, consistent reasoning",
    },
    {
        "title": "HALLUCINATED GENE — model invents Zfpx99",
        "trace": (
            "<think>\n"
            "The Clint1 knockout shows metabolic phenotypes. Interestingly, Zfpx99 is known "
            "to interact with Clint1 in the Golgi trafficking pathway, and loss of this "
            "interaction leads to impaired lipid metabolism. The Brkfake2 protein complex "
            "is also disrupted. Based on these interactions, survival is likely impaired.\n"
            "</think>\n"
            "Answer: A"
        ),
        "answer": "A",
        "genes": ["Clint1"],
        "expect": "LOW gene score — Zfpx99 and Brkfake2 don't exist in MGI",
    },
    {
        "title": "THINK/ANSWER CONTRADICTION",
        "trace": (
            "<think>\n"
            "Examining this Apoe knockout, the phenotype profile shows normal body weight "
            "and normal lifespan. This genotype does not affect survival. The mouse has "
            "no mortality phenotype and appears to have normal longevity. I see no reason "
            "to believe survival is compromised.\n"
            "</think>\n"
            "Answer: A"
        ),
        "answer": "A",
        "genes": ["Apoe"],
        "expect": "ZERO consistency — trace says 'no effect' but answer says 'impairs'",
    },
    {
        "title": "WRONG PATHWAY — Sirt6 claimed as respiratory",
        "trace": (
            "<think>\n"
            "Sirt6 is a key NAD-dependent deacetylase. This knockout shows a role in the "
            "respiratory system function and lung development pathway. Loss of Sirt6 "
            "leads to respiratory system failure and premature death. The phenotype "
            "profile confirms impaired survival.\n"
            "</think>\n"
            "Answer: A"
        ),
        "answer": "A",
        "genes": ["Sirt6"],
        "expect": "LOW system score — Sirt6 is metabolism/aging, not respiratory",
    },
    {
        "title": "PERFECT TRACE — obscure gene, careful reasoning",
        "trace": (
            "<think>\n"
            "This homozygous knockout shows 21 recorded phenotype terms including abnormal "
            "homeostasis and metabolic phenotypes. The presence of decreased body weight, "
            "abnormal glucose homeostasis, and impaired metabolism suggest this genotype "
            "may shorten lifespan. The phenotype burden is substantial and concentrated "
            "in metabolic systems. This likely impairs survival.\n"
            "</think>\n"
            "Answer: A"
        ),
        "answer": "A",
        "genes": ["Lepr"],
        "expect": "HIGH score — no hallucinations, consistent, correct system",
    },
]


def print_bar(score, width=30):
    filled = int(score * width)
    if score >= 0.8:
        color = GREEN
    elif score >= 0.5:
        color = YELLOW
    else:
        color = RED
    bar = color + "█" * filled + DIM + "░" * (width - filled) + RESET
    return f"{bar} {color}{score:.3f}{RESET}"


def main():
    print(f"\n{BOLD}{CYAN}{'═' * 60}{RESET}")
    print(f"{BOLD}{CYAN}  REASONING TRACE SCORER — LIVE DEMO{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")
    print(f"{DIM}  Validating LLM reasoning against 18,035 real MGI genes{RESET}")
    print(f"{DIM}  No API calls — pure programmatic checks{RESET}\n")

    all_scores = []

    for i, ex in enumerate(EXAMPLES, 1):
        print(f"{BOLD}{CYAN}── Example {i}/5: {ex['title']}{RESET}")
        print(f"{DIM}  Genes in prompt: {', '.join(ex['genes'])}{RESET}")
        print(f"{DIM}  Expected: {ex['expect']}{RESET}")

        time.sleep(0.5)

        result = score_trace(ex["trace"], ex["answer"], ex["genes"], use_api=False)
        all_scores.append(result["trace_score"])

        print(f"\n  {BOLD}COMPOSITE SCORE:{RESET}  {print_bar(result['trace_score'])}")

        for key, sub in result["sub_scores"].items():
            label = key.replace("_", " ").title()
            s = sub.get("score")
            if s is None:
                continue
            print(f"    {label:30s} {print_bar(s)}")

            if sub.get("hallucinated_genes"):
                print(f"      {RED}✗ Hallucinated: {', '.join(sub['hallucinated_genes'])}{RESET}")
            if sub.get("contradicts"):
                print(f"      {RED}✗ {sub['reason']}{RESET}")
            if sub.get("unverified"):
                print(f"      {RED}✗ Ungrounded: {', '.join(sub['unverified'])}{RESET}")
            if sub.get("famous_genes_cited"):
                print(f"      {YELLOW}⚠ Famous gene cited: {', '.join(sub['famous_genes_cited'])}{RESET}")

        print()

    mean = sum(all_scores) / len(all_scores)
    print(f"{BOLD}{CYAN}{'═' * 60}{RESET}")
    print(f"  {BOLD}MEAN TRACE SCORE:{RESET}  {print_bar(mean)}")
    print(f"  {DIM}Traces scored: {len(all_scores)} | API calls: 0 | Cost: $0.00{RESET}")
    print(f"{BOLD}{CYAN}{'═' * 60}{RESET}\n")


if __name__ == "__main__":
    main()
