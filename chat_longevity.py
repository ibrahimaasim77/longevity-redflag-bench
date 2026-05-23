"""Interactive Longevity-LLM tester. See thinking, toggle modes, inspect payloads.

Commands (type at the >>> prompt):
  /think on              — enable thinking mode (default)
  /think off             — disable thinking mode (terse number-only answers)
  /sys <text>            — set system prompt (default uses the dataset's)
  /sys reset             — clear the system prompt
  /show                  — show what will be sent on the next call
  /tokens <n>            — set max_tokens (default 1200)
  /quit                  — exit
  <anything else>        — sent as a user message
"""
from __future__ import annotations

import json
import os
import sys

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

BASE_URL = os.environ.get(
    "LONGEVITY_BASE_URL",
    "https://sqrq2pj09htgequ0.us-east-2.aws.endpoints.huggingface.cloud/v1",
)
MODEL = os.environ.get("LONGEVITY_MODEL", "longevity-llm")
# Prefer env var; fall back to the hackathon token (may be revoked).
API_KEY = os.environ.get("HF_TOKEN", "")

# This is the EXACT system prompt the LongevityBench dataset ships for NHANES rows.
# We're not making it up — it comes from the dataset itself.
DEFAULT_SYSTEM = (
    "You are a biomedical AI specialized in aging biology, "
    "trained on genomic, proteomic, and clinical data."
)

client = OpenAI(base_url=BASE_URL, api_key=API_KEY)


class Session:
    def __init__(self):
        self.system = DEFAULT_SYSTEM
        self.thinking = True
        self.max_tokens = 1200

    def build_messages(self, user_text: str):
        msgs = []
        if self.system:
            msgs.append({"role": "system", "content": self.system})
        msgs.append({"role": "user", "content": user_text})
        return msgs

    def build_kwargs(self, user_text: str):
        kw = {
            "model": MODEL,
            "messages": self.build_messages(user_text),
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
        }
        if not self.thinking:
            kw["extra_body"] = {"chat_template_kwargs": {"enable_thinking": False}}
        return kw

    def ask(self, user_text: str):
        kw = self.build_kwargs(user_text)
        print(f"\n[sending: thinking={'on' if self.thinking else 'off'}, "
              f"system={'yes' if self.system else 'no'}, max_tokens={self.max_tokens}]")
        try:
            r = client.chat.completions.create(**kw)
        except Exception as e:
            print(f"API ERROR: {type(e).__name__}: {e}")
            return
        raw = r.choices[0].message.content or ""
        u = r.usage
        print(f"[tokens: prompt={u.prompt_tokens}  completion={u.completion_tokens}]")

        if "</think>" in raw:
            think, _, answer = raw.partition("</think>")
            think = think.replace("<think>", "").strip()
            print(f"\n--- THINKING ({len(think)} chars) ---\n{think}")
            print(f"\n--- ANSWER ---\n{answer.strip()}")
        else:
            print(f"\n--- RESPONSE ---\n{raw}")


def main():
    s = Session()
    print("Longevity-LLM REPL. /quit to exit, /show to inspect payload, /think on|off to toggle.")
    while True:
        try:
            line = input("\n>>> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue

        if line in ("/quit", "/exit"):
            break
        if line == "/show":
            sample = s.build_kwargs("<your next prompt goes here>")
            print(json.dumps(sample, indent=2)[:1500])
            continue
        if line.startswith("/think "):
            v = line.split(maxsplit=1)[1].strip().lower()
            s.thinking = v in ("on", "true", "1", "yes")
            print(f"  thinking = {s.thinking}")
            continue
        if line.startswith("/tokens "):
            try:
                s.max_tokens = int(line.split()[1])
                print(f"  max_tokens = {s.max_tokens}")
            except Exception:
                print("  usage: /tokens 1200")
            continue
        if line == "/sys reset":
            s.system = ""
            print("  system prompt cleared")
            continue
        if line.startswith("/sys "):
            s.system = line[5:].strip()
            print(f"  system = {s.system!r}")
            continue

        s.ask(line)


if __name__ == "__main__":
    main()
