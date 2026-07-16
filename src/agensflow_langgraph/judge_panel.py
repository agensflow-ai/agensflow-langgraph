"""3-judge cross-family panel — adapter side (runs in the MAS process).

Mirrors the logic of `agensflow-mcp/engine/relative_judge.py`. The two
implementations exist because they run in different processes: the server side
serves Claude Code / codex clients that call `POST /judge/relative`; this side
runs inside the LangGraph process, where the adapter's `@agensflow(judge=...)`
decorator invokes it directly without a server round-trip.

A shared contract test (`tests/integration/test_judge_panel_contract.py`) feeds
the same synthetic OpenRouter response to BOTH implementations and asserts they
produce identical quality outputs — that catches drift.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import statistics
import sys

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
JUDGE_TIMEOUT_S = 25.0
_DIFF_CAP = 12000

AXES = ("correctness", "completeness", "precision", "robustness")

RELATIVE_RUBRIC = (
    "You are comparing two candidate solutions (A and B) to the SAME task. Score "
    "each solution INDEPENDENTLY on these axes, each in [0.0, 1.0]:\n"
    "  1. correctness  - does the response actually do what the task asks?\n"
    "  2. completeness - are all criteria covered, including stated edge cases?\n"
    "  3. precision    - concise, no dead content, no needless verbosity?\n"
    "  4. robustness   - handles edge cases / failure modes implied by the task?\n"
    "Use the comparison across A and B to ANCHOR the scale: the better solution should be near 1.0, "
    "the worse near 0.0; they can land close if quality is similar. Also produce a holistic 'score' in "
    "[0.0, 1.0] reflecting overall judgment, plus a one-sentence reason.\n"
    "REQUIRED — every solution MUST include all four axis_scores; do not omit any. Respond with ONLY "
    'this JSON object, no prose:\n'
    '{"A": {"axis_scores": {"correctness": <f>, "completeness": <f>, "precision": <f>, '
    '"robustness": <f>}, "score": <f>, "reason": "<one sentence>"},\n'
    ' "B": {"axis_scores": {"correctness": <f>, "completeness": <f>, "precision": <f>, '
    '"robustness": <f>}, "score": <f>, "reason": "<one sentence>"}}'
)


def _clamp01(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if v < 0.0 else (1.0 if v > 1.0 else v)


def _composed(axis_scores: dict) -> float:
    return sum(_clamp01(axis_scores.get(a)) for a in AXES) / len(AXES)


def _parse_side(obj) -> dict | None:
    if not isinstance(obj, dict):
        return None
    axes = obj.get("axis_scores")
    if not isinstance(axes, dict) or not all(a in axes for a in AXES):
        return None
    norm = {a: _clamp01(axes[a]) for a in AXES}
    score = _clamp01(obj.get("score", _composed(norm)))
    reason = str(obj.get("reason", "")).strip()[:300]
    return {"axis_scores": norm, "score": score, "reason": reason}


def _parse_verdict(text: str) -> tuple[dict | None, dict | None, str]:
    excerpt = (text or "").strip()[:300]
    obj = None
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        m = re.search(r"\{.*\}", text or "", re.DOTALL)
        if m:
            try:
                obj = json.loads(m.group(0))
            except (ValueError, TypeError):
                obj = None
    if not isinstance(obj, dict):
        return None, None, excerpt
    return _parse_side(obj.get("A")), _parse_side(obj.get("B")), excerpt


def _aggregate(candidate_scalars: list[float]) -> float:
    return statistics.mean(candidate_scalars) if candidate_scalars else 0.0


async def _ask(key: str, model: str, task: str, sol_a: str, sol_b: str):
    import httpx

    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": RELATIVE_RUBRIC},
            {"role": "user", "content": (
                f"Task:\n{task}\n\n=== Solution A ===\n{sol_a[:_DIFF_CAP]}\n\n"
                f"=== Solution B ===\n{sol_b[:_DIFF_CAP]}\n\nScore both A and B per the rubric.")},
        ],
        "temperature": 0.0, "max_tokens": 600,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {key}", "Content-Type": "application/json",
        "HTTP-Referer": "https://agensflow.ai", "X-Title": "AgensFlow Relative Judge (adapter)",
    }
    debug = bool(os.environ.get("OPENROUTER_JUDGE_DEBUG"))
    try:
        async with httpx.AsyncClient(timeout=JUDGE_TIMEOUT_S) as client:
            resp = await client.post(OPENROUTER_URL, headers=headers, json=payload)
            if resp.status_code != 200:
                if debug:
                    print(f"[judge {model}] HTTP {resp.status_code}: {resp.text[:200]}",
                          file=sys.stderr)
                return None, None, f"HTTP {resp.status_code}"
            content = resp.json().get("choices", [{}])[0].get("message", {}).get("content", "")
            A, B, excerpt = _parse_verdict(content)
            return A, B, excerpt
    except Exception as e:  # noqa: BLE001
        if debug:
            print(f"[judge {model}] error: {type(e).__name__}: {str(e)[:200]}", file=sys.stderr)
        return None, None, f"error: {type(e).__name__}"


async def _judge(key: str, model: str, task: str, candidate: str, baseline: str) -> dict:
    A1, B1, _ = await _ask(key, model, task, candidate, baseline)
    A2, B2, _ = await _ask(key, model, task, baseline, candidate)
    cand1, base1 = A1, B1
    cand2, base2 = B2, A2
    cand_scalars = [_composed(s["axis_scores"]) for s in (cand1, cand2) if s]
    base_scalars = [_composed(s["axis_scores"]) for s in (base1, base2) if s]
    if not cand_scalars:
        return {"model": model, "verdict": "unavailable",
                "Q_candidate": None, "Q_baseline": None,
                "axis_means_candidate": {}, "axis_means_baseline": {}}

    def _axis_mean(sides, axis):
        vals = [s["axis_scores"][axis] for s in sides if s]
        return statistics.mean(vals) if vals else None

    return {
        "model": model, "verdict": "scored",
        "Q_candidate": statistics.mean(cand_scalars),
        "Q_baseline": statistics.mean(base_scalars) if base_scalars else None,
        "axis_means_candidate": {a: _axis_mean([cand1, cand2], a) for a in AXES},
        "axis_means_baseline": {a: _axis_mean([base1, base2], a) for a in AXES},
    }


async def relative_quality(
    task: str,
    candidate: str,
    baseline: str,
    *,
    openrouter_key: str,
    models: tuple[str, ...],
) -> tuple[float, dict[str, float]]:
    """Run the panel and return (composed_quality, per_axis_scores).

    Fully async — the adapter's decorator calls this inside a running event loop.
    Returns quality=0.0 if all judges failed; caller decides whether to submit
    that reward or skip.
    """
    judges = list(await asyncio.gather(
        *[_judge(openrouter_key, m, task, candidate, baseline) for m in models]
    ))
    scored = [j for j in judges if j["verdict"] == "scored"]
    Q = _aggregate([j["Q_candidate"] for j in scored])
    per_axis: dict[str, float] = {}
    for a in AXES:
        vals = [j["axis_means_candidate"].get(a) for j in scored
                if j["axis_means_candidate"].get(a) is not None]
        if vals:
            per_axis[a] = float(statistics.mean(vals))
    return Q, per_axis
