"""Live Scoring API — streams LLM reasoning + Claude analysis step by step.

Run:  python api_server.py
Then: Lovable frontend hits http://localhost:8000

Endpoints:
  GET  /prompts          → list of 5 pre-built prompts
  GET  /score/{id}       → SSE stream of the full scoring pipeline
"""

import asyncio
import json
import re
import sys
import time
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "judge"))

from src import config

app = FastAPI(title="Longevity Trace Scorer")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

PROMPTS = [
    {
        "id": 0,
        "gene": "Npc1",
        "title": "Niemann-Pick cholesterol transporter",
        "description": "Npc1 knockout → neurodegeneration, lipid dysregulation, tremors",
        "difficulty": "Easy — classic lethal phenotype",
        "gold": "A",
        "gold_label": "Impairs survival",
        "genotype_id": "MGI:5293760",
        "condition": "geno_pheno",
        "phenotypes": "ataxia, tremors, lipid dysregulation, foam cell reticulosis, elevated cholesterol, Purkinje cell degeneration, weight loss",
        "prompt_text": (
            "Genotype: Npc1 knockout (Npc1<tm1Mbjg>/Npc1<tm1Mbjg>), homozygous\n"
            "Phenotype profile: ataxia, tremors, lipid dysregulation, foam cell reticulosis, "
            "elevated cholesterol, Purkinje cell degeneration, weight loss\n\n"
            "Does this genotype impair survival?\n"
            "Options: A. Yes, impairs survival  B. No, does not impair survival\n"
            "Think step by step, then answer."
        ),
    },
    {
        "id": 1,
        "gene": "Osmr",
        "title": "Oncostatin M receptor — the trick question",
        "description": "Osmr knockout → metabolic syndrome phenotypes BUT does NOT kill mice",
        "difficulty": "Hard — scary phenotypes, no mortality",
        "gold": "B",
        "gold_label": "Does NOT impair survival",
        "genotype_id": "MGI:2680724",
        "condition": "geno_pheno",
        "phenotypes": "insulin resistance, hepatic steatosis, chronic inflammation, elevated triglycerides, metabolic syndrome features",
        "prompt_text": (
            "Genotype: Osmr knockout (Osmr<tm1Aust>/Osmr<tm1Aust>), homozygous\n"
            "Phenotype profile: insulin resistance, hepatic steatosis, chronic inflammation, "
            "elevated triglycerides, metabolic syndrome features\n\n"
            "Does this genotype impair survival?\n"
            "Options: A. Yes, impairs survival  B. No, does not impair survival\n"
            "Think step by step, then answer."
        ),
    },
    {
        "id": 2,
        "gene": "Ctla4",
        "title": "Immune checkpoint regulator",
        "description": "Ctla4 knockout → unchecked T-cell activation, systemic inflammation",
        "difficulty": "Medium — immune catastrophe",
        "gold": "A",
        "gold_label": "Impairs survival",
        "genotype_id": "MGI:4940107",
        "condition": "geno_pheno",
        "phenotypes": "increased acute inflammation, T-cell hyperactivation, multi-organ lymphocyte infiltration",
        "prompt_text": (
            "Genotype: Ctla4 knockout (Ctla4<tm1All>/Ctla4<tm1All>), homozygous\n"
            "Phenotype profile: increased acute inflammation, T-cell hyperactivation, "
            "multi-organ lymphocyte infiltration\n\n"
            "Does this genotype impair survival?\n"
            "Options: A. Yes, impairs survival  B. No, does not impair survival\n"
            "Think step by step, then answer."
        ),
    },
    {
        "id": 3,
        "gene": "Hidden",
        "title": "Mystery gene — phenotype only (no gene name)",
        "description": "Can the LLM reason from phenotypes alone? No gene identity given.",
        "difficulty": "Hardest — ablation condition, gene hidden",
        "gold": "A",
        "gold_label": "Impairs survival",
        "genotype_id": "MGI:3690623",
        "condition": "pheno_only",
        "phenotypes": "gastrointestinal ulcers, colitis, rectal prolapse, abnormal thrombosis, cachexia, weight loss",
        "prompt_text": (
            "Genotype: [GENE IDENTITY HIDDEN]\n"
            "Phenotype profile: gastrointestinal ulcers, colitis, rectal prolapse, "
            "abnormal thrombosis in lung and kidney, cachexia, weight loss\n\n"
            "Based ONLY on the phenotype profile (no gene identity available), "
            "does this genotype impair survival?\n"
            "Options: A. Yes, impairs survival  B. No, does not impair survival\n"
            "Think step by step, then answer."
        ),
    },
    {
        "id": 4,
        "gene": "Cry1",
        "title": "Circadian clock component — subtle case",
        "description": "Cry1 knockout → circadian disruption, leukocytosis, low albumin",
        "difficulty": "Hard — disrupted clock ≠ death",
        "gold": "B",
        "gold_label": "Does NOT impair survival",
        "genotype_id": "MGI:6408315",
        "condition": "geno_pheno",
        "phenotypes": "circadian rhythm disruption, increased leukocyte count, decreased serum albumin",
        "prompt_text": (
            "Genotype: Cry1 knockout (Cry1<em1(IMPC)H>/Cry1<em1(IMPC)H>), homozygous\n"
            "Phenotype profile: circadian rhythm disruption, increased leukocyte count, "
            "decreased serum albumin\n\n"
            "Does this genotype impair survival?\n"
            "Options: A. Yes, impairs survival  B. No, does not impair survival\n"
            "Think step by step, then answer."
        ),
    },
]


def extract_answer(text):
    region = text.rsplit("</think>", 1)[-1] if "</think>" in text else text
    m = re.search(r"\b(?:answer|option)\s*[:=]?\s*\(?([AB])\)?", region, re.I)
    if m:
        return m.group(1).upper()
    matches = re.findall(r"\b([AB])\b", region[-100:])
    return matches[-1].upper() if matches else "?"


def call_longevity_llm(prompt_text):
    try:
        from openai import OpenAI
        client = OpenAI(base_url=config.LONGEVITY_BASE_URL, api_key=config.MODEL_ACCESS_TOKEN)
        start = time.time()
        resp = client.chat.completions.create(
            model=config.LONGEVITY_MODEL,
            messages=[
                {"role": "system", "content": "You are a biomedical AI specialized in aging biology."},
                {"role": "user", "content": prompt_text},
            ],
            max_tokens=400,
            temperature=0,
        )
        return resp.choices[0].message.content, time.time() - start, None
    except Exception as e:
        return None, 0, str(e)


def call_claude_scorer(trace, gene, phenotypes, pred, gold):
    try:
        from anthropic import Anthropic
        from score_trace import GENE_SYSTEMS
        gene_systems = GENE_SYSTEMS.get(gene.lower(), set())
        known = ", ".join(sorted(gene_systems)) if gene_systems else "unknown"

        prompt = (
            f"You are a mouse genetics expert. Score this LLM's reasoning about a gene knockout.\n\n"
            f"GENE: {gene}\n"
            f"KNOWN MGI SYSTEMS: {known}\n"
            f"PHENOTYPES: {phenotypes}\n"
            f"CORRECT ANSWER: {'A (impairs survival)' if gold == 'A' else 'B (does not impair)'}\n"
            f"MODEL PREDICTED: {'A (impairs survival)' if pred == 'A' else 'B (does not impair)'}\n\n"
            f"MODEL'S REASONING:\n{trace[:2000]}\n\n"
            f"SCORING RUBRIC (rate 0-3):\n"
            f"  0 — Wrong direction OR wrong biological mechanism\n"
            f"  1 — Correct direction, but mechanism missing or wrong\n"
            f"  2 — Correct direction + mechanism, sound logic\n"
            f"  3 — Full chain: molecular → cellular → system → survival outcome\n\n"
            f"Return JSON:\n"
            f'{{"score": 0-3, '
            f'"step_by_step": ["molecular: ...", "cellular: ...", "system: ...", "survival: ..."], '
            f'"fabrications": ["any invented facts"], '
            f'"verdict": "1-2 sentence summary", '
            f'"what_model_got_right": "brief", '
            f'"what_model_got_wrong": "brief"}}'
        )

        client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
        start = time.time()
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=500,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        latency = time.time() - start
        text = msg.content[0].text
        cleaned = re.sub(r"```json\s*", "", text)
        cleaned = re.sub(r"```\s*", "", cleaned)
        m = re.search(r"\{.*\}", cleaned, re.S)
        data = json.loads(m.group(0)) if m else {"score": 0, "verdict": text[:200]}
        data["latency"] = latency
        return data, None
    except Exception as e:
        return None, str(e)


def run_programmatic_checks(trace, pred, gene):
    from score_trace import score_trace
    genes = [gene] if gene != "Hidden" else []
    return score_trace(trace, pred, genes, use_api=False)


@app.get("/prompts")
def get_prompts():
    return [
        {k: v for k, v in p.items() if k != "prompt_text"}
        for p in PROMPTS
    ]


@app.get("/score/{prompt_id}")
async def score_prompt(prompt_id: int):
    if prompt_id < 0 or prompt_id >= len(PROMPTS):
        return {"error": "Invalid prompt ID"}

    p = PROMPTS[prompt_id]

    async def stream():
        # Step 1: Show the prompt
        yield {"event": "step", "data": json.dumps({
            "phase": "prompt",
            "title": "Sending prompt to Longevity-LLM...",
            "content": p["prompt_text"],
            "gene": p["gene"],
            "gold": p["gold"],
            "gold_label": p["gold_label"],
        })}
        await asyncio.sleep(0.5)

        # Step 2: Call Longevity-LLM
        yield {"event": "step", "data": json.dumps({
            "phase": "llm_calling",
            "title": "Longevity-LLM is thinking...",
            "content": "Waiting for model response (~8 seconds)...",
        })}

        trace, latency, err = await asyncio.to_thread(call_longevity_llm, p["prompt_text"])

        if err:
            # Fall back to cached trace
            cached_path = ROOT / "data" / "eval_controlled_240.jsonl"
            trace = None
            if cached_path.exists():
                with open(cached_path) as f:
                    for line in f:
                        r = json.loads(line)
                        if r["genotype_id"] == p["genotype_id"] and r["condition"] == p["condition"]:
                            trace = r.get("raw", "")
                            break
            if not trace:
                yield {"event": "step", "data": json.dumps({
                    "phase": "error",
                    "title": "LLM endpoint unavailable",
                    "content": err,
                })}
                return
            latency = 0
            yield {"event": "step", "data": json.dumps({
                "phase": "llm_fallback",
                "title": "Using cached trace (endpoint unavailable)",
                "content": trace[:200] + "...",
            })}

        pred = extract_answer(trace or "")

        # Step 3: Show LLM response
        yield {"event": "step", "data": json.dumps({
            "phase": "llm_response",
            "title": "Longevity-LLM responded",
            "trace": trace,
            "prediction": pred,
            "prediction_label": "Impairs survival" if pred == "A" else "Does NOT impair" if pred == "B" else "Unknown",
            "correct": pred == p["gold"],
            "latency": round(latency, 1),
        })}
        await asyncio.sleep(0.3)

        # Step 4: Programmatic checks
        yield {"event": "step", "data": json.dumps({
            "phase": "programmatic_start",
            "title": "Running programmatic checks...",
            "content": "Checking gene hallucination, think/answer consistency, pathway grounding",
        })}

        prog = await asyncio.to_thread(run_programmatic_checks, trace, pred, p["gene"])

        yield {"event": "step", "data": json.dumps({
            "phase": "programmatic_done",
            "title": "Programmatic checks complete",
            "trace_score": prog["trace_score"],
            "checks": {
                "gene_hallucination": {
                    "score": prog["sub_scores"]["gene_hallucination"]["score"],
                    "hallucinated": prog["sub_scores"]["gene_hallucination"].get("hallucinated_genes", []),
                },
                "consistency": {
                    "score": prog["sub_scores"]["think_answer_consistency"]["score"],
                    "contradicts": prog["sub_scores"]["think_answer_consistency"].get("contradicts", False),
                    "reason": prog["sub_scores"]["think_answer_consistency"].get("reason", ""),
                },
                "pathway": {
                    "score": prog["sub_scores"]["system_grounding"]["score"],
                    "unverified": prog["sub_scores"]["system_grounding"].get("unverified", []),
                },
            },
        })}
        await asyncio.sleep(0.3)

        # Step 5: Claude analysis
        yield {"event": "step", "data": json.dumps({
            "phase": "claude_calling",
            "title": "Claude Haiku is analyzing the biology...",
            "content": "Verifying mechanisms, checking for fabrications, scoring 0-3...",
        })}

        claude, cerr = await asyncio.to_thread(
            call_claude_scorer, trace, p["gene"], p["phenotypes"], pred, p["gold"]
        )

        if claude:
            yield {"event": "step", "data": json.dumps({
                "phase": "claude_done",
                "title": "Claude Haiku analysis complete",
                "score_0_3": claude.get("score", 0),
                "verdict": claude.get("verdict", ""),
                "step_by_step": claude.get("step_by_step", []),
                "fabrications": claude.get("fabrications", []),
                "what_right": claude.get("what_model_got_right", ""),
                "what_wrong": claude.get("what_model_got_wrong", ""),
                "latency": claude.get("latency", 0),
            })}
        else:
            yield {"event": "step", "data": json.dumps({
                "phase": "claude_error",
                "title": "Claude analysis failed",
                "content": cerr,
            })}

        # Step 6: Final summary
        combined = prog["trace_score"] * 0.5 + (claude.get("score", 0) / 3) * 0.5 if claude else prog["trace_score"]
        yield {"event": "step", "data": json.dumps({
            "phase": "final",
            "title": "Scoring complete",
            "programmatic_score": prog["trace_score"],
            "claude_score_0_3": claude.get("score", 0) if claude else None,
            "combined_score": round(combined, 3),
            "prediction": pred,
            "correct": pred == p["gold"],
            "gold": p["gold"],
            "gene": p["gene"],
        })}

    return EventSourceResponse(stream())


if __name__ == "__main__":
    import uvicorn
    print(f"\n  Longevity Trace Scorer API")
    print(f"  http://localhost:8000")
    print(f"  Endpoints: GET /prompts, GET /score/0..4\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)
