"""LLM-judge fallback — opt-in, OpenRouter-backed by default.

Default is OFF: `judge=False` (or None) on the decorator disables. Users opt in with:
    judge=True                                          # use env AGENSFLOW_JUDGE
    judge="openrouter:openai/gpt-4o-mini"               # single-model
    judge={"panel": ["x-ai/grok-4.3",                    # 3-judge cross-family panel
                     "openai/gpt-5.4-mini",
                     "qwen/qwen3.6-flash"]}
    judge=my_callable                                    # user-provided scorer

The judge runs after the node returns, submits reward asynchronously via
`/reward/submit`. Graph latency is never blocked on the judge.

Rubric: generic quality axes (correctness, completeness, precision, robustness).
Server-side scoring is opaque to the client.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Literal
from uuid import UUID

import httpx

from agensflow_langgraph.contracts import RewardSubmission


_OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_DEFAULT_TIMEOUT = 20.0


@dataclass(frozen=True)
class JudgeConfig:
    kind: Literal["openrouter", "panel", "callable"]
    model: str | None = None
    panel_models: tuple[str, ...] | None = None
    fn: Callable[..., Any] | None = None
    openrouter_key: str | None = None


def resolve_judge(judge_spec) -> JudgeConfig | None:
    """Convert the decorator's `judge=` argument into a runtime config, or None."""
    if judge_spec is False or judge_spec is None:
        return None

    if judge_spec is True:
        # Env var opt-in — user must set AGENSFLOW_JUDGE + OPENROUTER_API_KEY
        model = os.environ.get("AGENSFLOW_JUDGE")
        if not model:
            return None
        judge_spec = model

    if callable(judge_spec):
        return JudgeConfig(kind="callable", fn=judge_spec)

    # Panel form: {"panel": [model_a, model_b, model_c]}
    if isinstance(judge_spec, dict):
        panel = judge_spec.get("panel")
        if not panel:
            return None
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            return None
        return JudgeConfig(
            kind="panel", panel_models=tuple(panel), openrouter_key=key,
        )

    if isinstance(judge_spec, str):
        model = judge_spec.removeprefix("openrouter:")
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            return None
        return JudgeConfig(kind="openrouter", model=model, openrouter_key=key)

    return None


_RUBRIC = """You are grading an AI assistant's response on four axes. Return ONLY JSON, no prose.

Score each axis 0-1:
  * correctness   — is the response factually right?
  * completeness  — does it fully address the request?
  * precision     — is it concise and on-point (not padded, not verbose)?
  * robustness    — would it hold up on edge cases the user didn't specify?

Return: {"quality": <0-1 aggregate>, "axes": {"correctness": ..., "completeness": ...,
"precision": ..., "robustness": ...}, "reasoning": "one sentence"}

Input to the assistant:
{input_}

Assistant response:
{output_}
"""


async def submit_judge_reward(
    *,
    decision_id: UUID,
    input_messages: Any,
    output_message: Any,
    action: str,
    judge_cfg: JudgeConfig,
    server_url: str | None,
    tenant_key: str | None,
) -> None:
    """Run the judge (async) and submit RewardSubmission. Never raises."""
    try:
        if judge_cfg.kind == "callable":
            score = await _await_maybe(
                judge_cfg.fn(input_messages, output_message, action)
            )
            quality, axes, reasoning = _normalize_score(score)
        elif judge_cfg.kind == "panel":
            # 3-judge cross-family panel — imported lazily to keep the
            # single-model path free of the extra module.
            from agensflow_langgraph.judge_panel import relative_quality

            task_text = _render_task(input_messages)
            candidate_text = _render_response(output_message)
            baseline_text = ""   # no baseline for adapter path; panel judges A alone
            q, axes_dict = await relative_quality(
                task=task_text,
                candidate=candidate_text,
                baseline=baseline_text,
                openrouter_key=judge_cfg.openrouter_key or "",
                models=judge_cfg.panel_models or (),
            )
            quality = q
            axes = axes_dict or None
            reasoning = f"panel Q over {len(judge_cfg.panel_models or ())} judges"
        else:
            score = await _openrouter_judge(
                input_messages,
                output_message,
                judge_cfg.model or "openai/gpt-4o-mini",
                judge_cfg.openrouter_key or "",
            )
            quality, axes, reasoning = _normalize_score(score)
    except Exception:
        return  # Best-effort. If the judge fails, no reward is recorded (fine).

    if quality is None:
        return

    from agensflow_langgraph.client import get_client

    client = get_client(server_url, tenant_key)
    await client.a_submit_reward(
        RewardSubmission(
            decision_id=decision_id,
            quality=quality,
            quality_source="judge",
            quality_axes=axes,
            quality_reasoning=reasoning,
        )
    )


async def _openrouter_judge(
    input_messages: Any, output_message: Any, model: str, api_key: str
) -> dict:
    prompt = _RUBRIC.format(
        input_=json.dumps(input_messages, default=str)[:4000],
        output_=json.dumps(output_message, default=str)[:2000],
    )
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.0,
        "max_tokens": 400,
        "response_format": {"type": "json_object"},
    }
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://agensflow.ai",
        "X-Title": "AgensFlow LangGraph Judge",
    }
    async with httpx.AsyncClient(timeout=_DEFAULT_TIMEOUT) as c:
        r = await c.post(_OPENROUTER_URL, headers=headers, json=payload)
        r.raise_for_status()
        content = r.json()["choices"][0]["message"]["content"]
    return json.loads(content)


async def _await_maybe(x: Any) -> Any:
    """Awaits `x` if it's awaitable, else returns it unchanged."""
    import inspect

    if inspect.isawaitable(x):
        return await x
    return x


def _render_task(input_messages: Any) -> str:
    """Panel path: collapse the input message list into a task-description string.

    The panel rubric expects a `task` describing what the assistant was asked to do.
    We approximate that from the most recent human/user message in the input list."""
    if isinstance(input_messages, list):
        for msg in reversed(input_messages):
            if isinstance(msg, dict):
                role = msg.get("role", "").lower()
                if role in ("user", "human"):
                    return str(msg.get("content", ""))[:2000]
    if isinstance(input_messages, dict):
        return str(input_messages.get("content", ""))[:2000]
    return str(input_messages)[:2000] if input_messages else ""


def _render_response(output_message: Any) -> str:
    """Panel path: extract the assistant's response text."""
    if isinstance(output_message, dict):
        return str(output_message.get("content", ""))[:6000]
    return str(output_message)[:6000] if output_message else ""


def _normalize_score(score: Any) -> tuple[float | None, dict[str, float] | None, str | None]:
    """Normalize whatever the judge returned into (quality, axes, reasoning)."""
    if isinstance(score, dict):
        q = score.get("quality")
        if q is None:
            axes = score.get("axes")
            if isinstance(axes, dict) and axes:
                # aggregate = mean of axes
                q = sum(float(v) for v in axes.values() if isinstance(v, (int, float))) / len(
                    axes
                )
            else:
                return None, None, None
        return (
            _clip01(float(q)),
            {k: _clip01(float(v)) for k, v in (score.get("axes") or {}).items()} or None,
            score.get("reasoning"),
        )
    if isinstance(score, (int, float)):
        return _clip01(float(score)), None, None
    return None, None, None


def _clip01(x: float) -> float:
    return max(0.0, min(1.0, x))
