"""Deterministic-mock demo — no LLM calls, no API keys, no OpenRouter judge.

Shows the whole substrate loop with mock models that return canned responses at
known cost. Every node's "quality" is a fixed synthetic score per action, so the
substrate learns per-node routing deterministically over ~15 runs.

Runs against a locally-started `agensflow-mcp` server (default: http://localhost:8000).

Usage:
    # Terminal 1 — start the policy server
    cd ../../agensflow-mcp
    source .venv/bin/activate && uvicorn agensflow_mcp.app:app --port 8000

    # Terminal 2 — issue a key + run the demo
    export AGENSFLOW_SERVER_URL=http://localhost:8000
    export AGENSFLOW_API_KEY=$(curl -sX POST http://localhost:8000/auth/anonymous | \
                              python -c "import sys,json;print(json.load(sys.stdin)['api_key'])")
    python graph.py --runs 20

Expected output: after ~15 runs the substrate converges each node to its optimal
action (see SYNTHETIC_QUALITY below).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from typing import Any

from agensflow_langgraph import agensflow, arecord_reward

# ---- Synthetic quality landscape — the ground truth we're expecting the substrate ----
# ---- to discover. Each (node, action) has a fixed quality; the substrate should ------
# ---- prefer the highest-quality action per node.                                ------

SYNTHETIC_QUALITY = {
    "classify_intent":     {"cheap": 0.95, "balanced": 0.85, "deep": 0.75},  # cheap wins
    "retrieve_context":    {"cheap": 0.50, "balanced": 0.90, "deep": 0.85},  # balanced wins
    "synthesize_response": {"cheap": 0.20, "balanced": 0.70, "deep": 0.95},  # deep wins
}


class MockModel:
    """A canned Runnable — LangChain-compatible enough for the decorator to bind."""

    def __init__(self, name: str) -> None:
        self.name = name

    def with_config(self, callbacks=None, metadata=None, **_):
        # Return self — the callbacks don't fire because we don't inherit from
        # BaseChatModel; that's fine, this demo doesn't test cost capture.
        return self

    async def ainvoke(self, x, config=None):
        return {"model_used": self.name}


pool = {
    "cheap":    MockModel("cheap"),
    "balanced": MockModel("balanced"),
    "deep":     MockModel("deep"),
}


@agensflow(pool=pool)
async def classify_intent(state: dict, model, config=None) -> dict:
    r = await model.ainvoke(state.get("messages", []))
    return {"messages": [{"role": "assistant", "content": r["model_used"]}]}


@agensflow(pool=pool)
async def retrieve_context(state: dict, model, config=None) -> dict:
    r = await model.ainvoke(state.get("messages", []))
    return {"messages": [{"role": "assistant", "content": r["model_used"]}]}


@agensflow(pool=pool)
async def synthesize_response(state: dict, model, config=None) -> dict:
    r = await model.ainvoke(state.get("messages", []))
    return {"messages": [{"role": "assistant", "content": r["model_used"]}]}


NODES = [
    ("classify_intent", classify_intent),
    ("retrieve_context", retrieve_context),
    ("synthesize_response", synthesize_response),
]


async def run_one_iteration(iter_i: int) -> dict[str, str]:
    """Run all three nodes once. For each, submit synthetic reward based on the
    chosen action. Return {node_name: chosen_action}."""
    chosen: dict[str, str] = {}
    for node_name, node_fn in NODES:
        thread_id = f"iter_{iter_i}_{node_name}"
        config = {
            "metadata": {"langgraph_node": node_name},
            "configurable": {"thread_id": thread_id},
        }
        state = {"messages": [{"role": "user", "content": f"iter {iter_i}"}]}
        result = await node_fn(state, config=config)
        action = result["messages"][0]["content"]
        chosen[node_name] = action
        quality = SYNTHETIC_QUALITY[node_name][action]
        await arecord_reward(quality=quality, thread_id=thread_id)
    return chosen


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=20)
    args = ap.parse_args()

    if not os.environ.get("AGENSFLOW_API_KEY"):
        print(
            "  Set AGENSFLOW_API_KEY before running. See the docstring at the top.",
            file=sys.stderr,
        )
        return 2

    print(f"  Running {args.runs} iterations against {os.environ.get('AGENSFLOW_SERVER_URL', 'localhost:8000')}")
    print("  Expected converged routing:")
    for node, quality_map in SYNTHETIC_QUALITY.items():
        argmax = max(quality_map, key=lambda k: quality_map[k])
        print(f"    {node:<22} → {argmax} (quality {quality_map[argmax]})")
    print()

    history = []
    for i in range(args.runs):
        chosen = await run_one_iteration(i)
        history.append(chosen)
        picks = "  ".join(f"{k}={v}" for k, v in chosen.items())
        print(f"  [{i+1:>2}/{args.runs}]  {picks}")

    # Summary: last 5 iterations' choices per node
    print("\n  Last 5 iterations (convergence check):")
    for node, _ in NODES:
        recent = [h[node] for h in history[-5:]]
        expected = max(SYNTHETIC_QUALITY[node], key=lambda k: SYNTHETIC_QUALITY[node][k])
        hits = sum(1 for r in recent if r == expected)
        marker = "✓" if hits >= 4 else "…"
        print(f"    {marker} {node:<22} {recent}  (expected {expected})")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
