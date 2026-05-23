"""Central config. Read from env; never hard-code the token."""

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader (no dependency). Sets vars not already in the environment."""
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


_load_dotenv(REPO_ROOT / ".env")

DATA_DIR = REPO_ROOT / "data"
OUTPUTS_DIR = REPO_ROOT / "outputs"
TASKS_DIR = REPO_ROOT / "tasks"
MOCK_DIR = REPO_ROOT / "mock"

# --- Model under test: Longevity-LLM (Qwen3.5-9B) on the HF OpenAI-compatible endpoint ---
# Base URL must include the /v1 suffix for the OpenAI-compatible Messages API.
LONGEVITY_BASE_URL = os.environ.get(
    "LONGEVITY_BASE_URL",
    "https://sqrq2pj09htgequ0.us-east-2.aws.endpoints.huggingface.cloud/v1",
)
# This endpoint is served by vLLM and requires the EXACT model id "longevity-llm"
# (NOT "tgi"). Confirmed via GET /v1/models.
LONGEVITY_MODEL = os.environ.get("LONGEVITY_MODEL", "longevity-llm")
# Token to CALL the model endpoint (the actual LLM credential from event materials).
MODEL_ACCESS_TOKEN = os.environ.get("MODEL_ACCESS_TOKEN", "")

# Personal HF READ token — used only for DATASET access (datasets-server / load_dataset).
HF_TOKEN = os.environ.get("HF_TOKEN", "")

# --- Judge (bonus reasoning scorer): the only thing the $50 Anthropic credit pays for ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
JUDGE_MODEL = os.environ.get("JUDGE_MODEL", "claude-haiku-4-5-20251001")  # cheap; cap max_tokens
JUDGE_MAX_TOKENS = int(os.environ.get("JUDGE_MAX_TOKENS", "600"))

# --- Generation constraints (from track-01-spec.md) ---
MAX_PROMPT_TOKENS_CL100K = 30_000   # submission cap (cl100k_base)
MODEL_MAX_LEN = 28_000              # endpoint's REAL ctx (vLLM /v1/models); prompt+completion, model tokenizer
MIN_PROMPTS_PER_TASK = 50
SEED = int(os.environ.get("LRB_SEED", "1234"))   # fix for reproducibility


def require(name: str, value: str) -> str:
    if not value:
        raise RuntimeError(f"{name} is not set. Copy .env.example to .env and fill it in.")
    return value
