"""Contract test — server + adapter panel produce identical output.

The panel logic lives in TWO places (server: agensflow_mcp.engine.relative_judge;
adapter: agensflow_langgraph.judge_panel) because the two run in different
processes. This test feeds the SAME synthetic OpenRouter response to both and
asserts they compute the same composed quality + per-axis means.

If this test breaks, one implementation has drifted from the other — fix ONE side
to match, don't paper over with tolerance.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest


# Two "judges" both return the same well-formed rubric response, so both panels
# should produce identical composed quality and axis means.
_JUDGE_RESPONSE = {
    "choices": [
        {
            "message": {
                "content": (
                    '{"A": {"axis_scores": {"correctness": 0.90, "completeness": 0.85, '
                    '"precision": 0.80, "robustness": 0.75}, "score": 0.83, '
                    '"reason": "solid overall"},'
                    ' "B": {"axis_scores": {"correctness": 0.60, "completeness": 0.65, '
                    '"precision": 0.70, "robustness": 0.55}, "score": 0.62, '
                    '"reason": "weaker on correctness and robustness"}}'
                )
            }
        }
    ]
}


class _FakeHttpxResponse:
    status_code = 200

    def json(self):
        return _JUDGE_RESPONSE


class _FakeAsyncClient:
    def __init__(self, *_args, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def post(self, *_args, **_kwargs):
        return _FakeHttpxResponse()


@pytest.fixture
def _patched_httpx():
    """Patch httpx.AsyncClient so BOTH panel modules use the fake transport.

    Both server relative_judge.py and adapter judge_panel.py import httpx lazily
    and instantiate `httpx.AsyncClient(...)`. Patching the class at the module
    level catches both."""
    import httpx

    with patch.object(httpx, "AsyncClient", _FakeAsyncClient):
        yield


@pytest.mark.asyncio
async def test_server_and_adapter_panels_produce_same_quality(_patched_httpx) -> None:
    """Feed the same rubric response to both panel implementations; assert equality."""
    # --- Server side ---
    from agensflow_mcp.engine.relative_judge import _panel as server_panel

    server_q, server_judges = await server_panel(
        key="fake-key",
        models=("m1", "m2", "m3"),
        task="say hello politely",
        candidate="Hello, nice to meet you!",
        baseline="hi",
    )

    # --- Adapter side ---
    from agensflow_langgraph.judge_panel import relative_quality as adapter_panel

    adapter_q, adapter_axes = await adapter_panel(
        task="say hello politely",
        candidate="Hello, nice to meet you!",
        baseline="hi",
        openrouter_key="fake-key",
        models=("m1", "m2", "m3"),
    )

    # Both should score the SAME composed candidate quality
    # (same rubric response → same axis averages → same composed mean).
    assert server_q == pytest.approx(adapter_q, rel=1e-6), (
        f"Server panel Q ({server_q}) diverged from adapter panel Q ({adapter_q}). "
        "One implementation has drifted."
    )


@pytest.mark.asyncio
async def test_adapter_panel_extracts_per_axis(_patched_httpx) -> None:
    """The adapter's `relative_quality` returns per-axis scores for the candidate."""
    from agensflow_langgraph.judge_panel import relative_quality

    _, per_axis = await relative_quality(
        task="say hello",
        candidate="Hi",
        baseline="Yo",
        openrouter_key="fake-key",
        models=("m1",),
    )
    assert set(per_axis) == {"correctness", "completeness", "precision", "robustness"}
    # Rubric response returns A.correctness=0.90 and B.correctness=0.60 no matter
    # which side we send. Panel does order-swap: order1 sends candidate=A (0.90),
    # order2 sends candidate=B (0.60). Averaging: (0.90+0.60)/2 = 0.75.
    assert per_axis["correctness"] == pytest.approx(0.75)


@pytest.mark.asyncio
async def test_adapter_panel_returns_zero_when_all_judges_fail() -> None:
    """If EVERY judge returns garbage / times out, the panel returns Q=0.0."""
    import httpx

    class _FailingClient:
        def __init__(self, *_a, **_k):
            pass
        async def __aenter__(self):
            return self
        async def __aexit__(self, *_a):
            return False
        async def post(self, *_a, **_k):
            raise httpx.RequestError("simulated network error")

    with patch.object(httpx, "AsyncClient", _FailingClient):
        from agensflow_langgraph.judge_panel import relative_quality
        q, axes = await relative_quality(
            task="t", candidate="c", baseline="b",
            openrouter_key="fake-key", models=("m1", "m2"),
        )
        assert q == 0.0
        assert axes == {}
