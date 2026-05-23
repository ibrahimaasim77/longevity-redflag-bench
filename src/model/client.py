"""Client for Longevity-LLM via the HF OpenAI-compatible endpoint.

Functional. Includes retry/backoff because the endpoint is a shared event credential
(expect cold starts, rate limits, contention). Never let a transient error kill a run.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import List, Optional

from src import config


@dataclass
class ChatResult:
    content: str
    latency_s: float
    ok: bool
    error: Optional[str] = None


def _client():
    # Imported lazily so the rest of the repo works without `openai` installed.
    from openai import OpenAI

    return OpenAI(
        base_url=config.LONGEVITY_BASE_URL,
        api_key=config.require("HF_TOKEN", config.HF_TOKEN),
    )


def chat(
    messages: List[dict],
    *,
    temperature: float = 0.0,
    max_tokens: int = 600,
    retries: int = 3,
    backoff=(5, 15, 45),
) -> ChatResult:
    """Send a ChatML `messages` list. Returns ChatResult; never raises on API error.

    messages: [{"role": "system"|"user"|"assistant", "content": str}, ...]
    """
    client = _client()
    last_err = None
    for attempt in range(retries):
        t0 = time.time()
        try:
            resp = client.chat.completions.create(
                model=config.LONGEVITY_MODEL,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            content = resp.choices[0].message.content or ""
            return ChatResult(content=content, latency_s=time.time() - t0, ok=True)
        except Exception as e:  # noqa: BLE001 — deliberately broad; we log, never crash
            last_err = str(e)
            if attempt < retries - 1:
                time.sleep(backoff[min(attempt, len(backoff) - 1)])
    return ChatResult(content="", latency_s=0.0, ok=False, error=last_err)
