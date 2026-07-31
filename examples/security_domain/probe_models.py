"""Isolate which layer breaks each model.

For every model slug we plan to use, tries three progressively-integrated calls:

  1. RAW  — direct httpx POST to OpenRouter /chat/completions with a simple
            single-turn prompt. If this fails, the slug is wrong or OpenRouter
            says the model is unavailable.

  2. LC   — ChatOpenAI(base_url=openrouter, model=slug).ainvoke(...) with
            plain text response. If RAW passes but LC fails, LangChain's
            OpenAI wrapper is doing something the model doesn't like.

  3. STRUCT — .with_structured_output(SimpleSchema, method='function_calling').
              If LC passes but STRUCT fails, the model doesn't support
              function-calling and we need a different structured-output
              method (json_mode, json_schema, ...).

Zero substrate, zero graph, zero judge — just probe the wire.

Usage:
    python -m examples.security_domain.probe_models
"""

from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

import httpx
from pydantic import BaseModel, Field


REPO_ROOT = Path(__file__).resolve().parent.parent.parent


TASK_POOL_MODELS = [
    "anthropic/claude-haiku-4.5",
    "anthropic/claude-sonnet-5",
    "thinkingmachines/inkling",
]

JUDGE_MODELS = [
    "x-ai/grok-4.3",
    "openai/gpt-5.4-mini",
    "qwen/qwen3.6-flash",
]

BASELINE_MODELS = [
    # Candidate baselines. We need ONE that works — out of task pool
    # (Anthropic + ThinkingMachines) and out of judge panel (xAI + OpenAI + Qwen).
    # Baseline only needs LC to work (plain text); no STRUCT required.
    "google/gemini-2.5-flash",
    "google/gemini-2.5-flash-lite",
    "deepseek/deepseek-v3.1",
    "mistralai/mistral-medium-3",
]


class SimpleAnswer(BaseModel):
    answer: str = Field(description="one-sentence answer")


async def _raw_call(session: httpx.AsyncClient, model: str, key: str) -> tuple[bool, str]:
    """Direct httpx POST — no LangChain."""
    try:
        r = await session.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "You are a terse assistant."},
                    {"role": "user", "content": "What is 2 + 2? One sentence."},
                ],
                "temperature": 0.0,
                "max_tokens": 60,
            },
            timeout=30.0,
        )
        if r.status_code == 200:
            content = r.json()["choices"][0]["message"]["content"]
            return True, content.strip()[:80]
        return False, f"HTTP {r.status_code}: {r.text[:200]}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"


async def _langchain_call(model: str, key: str) -> tuple[bool, str]:
    """ChatOpenAI(base_url=openrouter) — no structured output."""
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            model=model,
            temperature=0.0,
            max_retries=0,
        )
        msg = await llm.ainvoke([
            ("system", "You are a terse assistant."),
            ("human", "What is 2 + 2? One sentence."),
        ])
        content = getattr(msg, "content", str(msg))
        return True, str(content).strip()[:80]
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"


async def _structured_call(model: str, key: str) -> tuple[bool, str]:
    """.with_structured_output(SimpleAnswer, method='function_calling')."""
    try:
        from langchain_openai import ChatOpenAI
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=key,
            model=model,
            temperature=0.0,
            max_retries=0,
        )
        structured = llm.with_structured_output(SimpleAnswer, method="function_calling")
        r = await structured.ainvoke([
            ("system", "Return STRICT JSON."),
            ("human", "What is 2 + 2? Answer in one sentence."),
        ])
        return True, f"answer={r.answer!r}"
    except Exception as e:
        return False, f"{type(e).__name__}: {str(e)[:200]}"


async def probe_one(session: httpx.AsyncClient, label: str, model: str, key: str, structured: bool):
    raw_ok, raw_msg = await _raw_call(session, model, key)
    lc_ok, lc_msg = await _langchain_call(model, key)
    st_ok, st_msg = (True, "not-tested") if not structured else await _structured_call(model, key)

    def mark(ok):
        return "\033[92m✓\033[0m" if ok else "\033[91m✗\033[0m"

    print(f"\n  {label}  {model}")
    print(f"    {mark(raw_ok)} RAW    :  {raw_msg}")
    print(f"    {mark(lc_ok)} LC     :  {lc_msg}")
    if structured:
        print(f"    {mark(st_ok)} STRUCT :  {st_msg}")
    return {"model": model, "raw": raw_ok, "lc": lc_ok, "structured": st_ok if structured else None}


async def main():
    # Load OpenRouter key
    try:
        from dotenv import load_dotenv
        env_path = REPO_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            print(f"  loaded env from {env_path}")
        else:
            load_dotenv()
    except ImportError:
        pass
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    print()
    print(f"  ┌─── model probe ───────────────────────────────────────")
    print(f"  │ Testing each slug at 3 levels:")
    print(f"  │   RAW    — direct httpx POST to OpenRouter")
    print(f"  │   LC     — ChatOpenAI(base_url=openrouter).ainvoke plain-text")
    print(f"  │   STRUCT — LC + .with_structured_output(schema, method='function_calling')")
    print(f"  └──────────────────────────────────────────────────────")

    results = []
    async with httpx.AsyncClient() as session:
        print(f"\n  === TASK POOL (need STRUCT to work) ===")
        for m in TASK_POOL_MODELS:
            results.append(await probe_one(session, "task ", m, key, structured=True))

        print(f"\n  === JUDGES (need STRUCT to work — panel parses JSON) ===")
        for m in JUDGE_MODELS:
            results.append(await probe_one(session, "judge", m, key, structured=True))

        print(f"\n  === BASELINE (only needs LC — plain text answer) ===")
        for m in BASELINE_MODELS:
            results.append(await probe_one(session, "base ", m, key, structured=False))

    # Summary
    print(f"\n  === summary ===")
    fails = [r for r in results if not r["raw"] or not r["lc"] or (r["structured"] is False)]
    if not fails:
        print("    all models pass all applicable layers ✓")
    else:
        for r in fails:
            layers_broken = []
            if not r["raw"]: layers_broken.append("RAW")
            if not r["lc"]: layers_broken.append("LC")
            if r["structured"] is False: layers_broken.append("STRUCT")
            print(f"    ✗ {r['model']:<40}  broken at: {', '.join(layers_broken)}")


if __name__ == "__main__":
    asyncio.run(main())
