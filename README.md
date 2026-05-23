# Red-Flag Clinical-Reasoning Benchmark

Caltech Longevity Hackathon — **Track 01: LongevityLLM Benchmarking**. We build a
JSONL/ChatML benchmark that tests whether Insilico's **Longevity-LLM (Qwen3.5-9B)**
derives a high-level phenotype (10-yr mortality) from low-level NHANES clinical data —
and whether it reasons about clinical *context* or reacts to scary keywords.

**The deliverable is the dataset** (`benchmark.jsonl`), not a demo. Judged on Utility /
Diversity / Retrieval-Resistance / Statistical-Rigor (20 pts). Full plan in the vault:
`vault/shared/caltech-hackathon-2026/` (build-plan, grading-rubric-spec, task-authoring-worksheet, deck→README).

## Quickstart
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # fill HF_TOKEN (and ANTHROPIC_API_KEY for the bonus judge)

python scripts/smoke_endpoint.py                                   # hour-0: endpoint alive?
python scripts/contamination_probe.py                              # hour-0: model recognize NHANES?
python mock/make_mock.py                                           # build mock records
python validate/validate_jsonl.py mock/mock_records.jsonl --min-per-task 1   # gate works
```

## The contract
`schema/records.py :: BenchmarkRecord` is the single shared interface. A submission line
is `record.model_dump_json()`. **Don't change a field without telling the team.** Build
against `mock/mock_records.jsonl` until the real NHANES cohort lands.

## Ground truth (3 layers — see build-plan.md §2)
- **A / absolute:** real profile → real linked outcome (`binary_survival`, `ordinal_risk`, `regression`)
- **B / relative:** counterfactual red-flag effect (`pairwise_counterfactual`, `set_generation`) — direction + matched-cohort band
- **C / bonus:** reasoning-verification scorer (`src/score/deterministic.py` + `judge.py`)

## Who owns what
| Person | Start here | Builds |
|---|---|---|
| **Anderson** | `src/generate/tasks.py` | task generators, profile render/perturb, model run+parse, scorer, `run_all.py` |
| **CS teammate** | `src/nhanes/build_cohort.py` | NHANES acquire/join/**censoring**, baselines, matched-cohort effects |
| **Bio 1** | `tasks/redflags.csv` | red-flag table: direction + HR band + citation (no git) |
| **Bio 2** | `tasks/context_cases.yaml` | keyword-traps + biological-correctness criteria + citations (no git) |

## How we work (lean — co-located, 32h)
- One repo, short-lived per-person branches, merge to `main` freely; **say it out loud** before merging. No required reviews.
- The **schema is the coordination mechanism**, not a board. Lock it; develop against mock.
- Bio teammates edit `tasks/*` in-repo or a shared doc — no git ceremony required.
- **Validate before every freeze:** `python validate/validate_jsonl.py outputs/benchmark.jsonl` must PASS.
- `data/` and `outputs/` are gitignored. **Never commit `.env` / `HF_TOKEN`.**

## Runnable now vs stubbed
- **Runnable:** smoke test, contamination probe, mock generator, validator, model client, parser, metrics, red-flag loader, bonus scorer.
- **Stubs (locked signatures, `# TODO(owner)`):** `src/nhanes/*`, `src/generate/*`, `src/baselines/*`, `run_all.py` wiring.
