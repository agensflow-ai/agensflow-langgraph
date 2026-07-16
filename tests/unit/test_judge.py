"""Judge resolution: default OFF, opt-in via decorator, callable + OpenRouter."""

from __future__ import annotations

import os

from agensflow_langgraph.judge import (
    JudgeConfig,
    _normalize_score,
    resolve_judge,
)


def test_default_disabled_via_false():
    assert resolve_judge(False) is None
    assert resolve_judge(None) is None


def test_callable_wraps_as_callable_kind():
    def my_judge(inp, out, action):
        return 0.7

    cfg = resolve_judge(my_judge)
    assert cfg is not None
    assert cfg.kind == "callable"
    assert cfg.fn is my_judge


def test_string_spec_needs_openrouter_key(monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    cfg = resolve_judge("openrouter:openai/gpt-4o-mini")
    assert cfg is None  # no key ⇒ disabled


def test_string_spec_with_key(monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-fake")
    cfg = resolve_judge("openrouter:openai/gpt-4o-mini")
    assert cfg is not None
    assert cfg.kind == "openrouter"
    assert cfg.model == "openai/gpt-4o-mini"


def test_true_needs_env_agensflow_judge(monkeypatch):
    monkeypatch.delenv("AGENSFLOW_JUDGE", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-fake")
    # No AGENSFLOW_JUDGE set ⇒ resolve returns None (opt-in requires the model spec)
    assert resolve_judge(True) is None

    monkeypatch.setenv("AGENSFLOW_JUDGE", "openai/gpt-4o-mini")
    cfg = resolve_judge(True)
    assert cfg is not None
    assert cfg.model == "openai/gpt-4o-mini"


def test_normalize_scalar():
    q, axes, reason = _normalize_score(0.85)
    assert q == 0.85
    assert axes is None
    assert reason is None


def test_normalize_dict_with_quality():
    q, axes, reason = _normalize_score(
        {"quality": 0.9, "axes": {"correctness": 0.95, "precision": 0.85}, "reasoning": "good"}
    )
    assert q == 0.9
    assert axes == {"correctness": 0.95, "precision": 0.85}
    assert reason == "good"


def test_normalize_dict_axes_only_averages():
    q, axes, reason = _normalize_score(
        {"axes": {"correctness": 0.6, "completeness": 0.8}}
    )
    assert q == 0.7
    assert axes == {"correctness": 0.6, "completeness": 0.8}


def test_normalize_clips_out_of_range():
    q, _, _ = _normalize_score(1.5)
    assert q == 1.0
    q, _, _ = _normalize_score(-0.2)
    assert q == 0.0


def test_normalize_junk_returns_none():
    q, axes, reason = _normalize_score("not a number")
    assert q is None
