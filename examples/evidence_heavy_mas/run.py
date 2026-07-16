"""CLI entrypoint for the evidence_heavy MAS example.

Usage (from the agensflow-langgraph repo root):

    export OPENROUTER_API_KEY=sk-or-...
    export AGENSFLOW_SERVER_URL=http://localhost:8000
    export AGENSFLOW_API_KEY=agf_...

    python -m examples.evidence_heavy_mas.run --tasks 5 --runs 3

Runs `runs` iterations of each of the first `tasks` benchmark tasks, prints
per-run action picks + rubric score + cost, and a summary at the end showing
which action the substrate converged to for each node.

Optional:
    --stream            Use graph.astream_events() instead of ainvoke — verifies
                        callback fire-rate under streaming.
    --export-policy P   After all runs, GET /policy/me and write to P as JSON.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

# Support both `python -m examples.evidence_heavy_mas.run` and direct execution
# (`python examples/evidence_heavy_mas/run.py`) — the latter needs a small path fix.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples.evidence_heavy_mas.graph import build_graph, build_pools  # type: ignore[no-redef]
    from examples.evidence_heavy_mas.tasks import TASKS, score_answer  # type: ignore[no-redef]
else:
    from .graph import build_graph, build_pools
    from .tasks import TASKS, score_answer

from agensflow_langgraph import arecord_reward


async def _run_one(compiled, task, iter_idx: int, use_stream: bool) -> dict:
    """Run the graph once for one task. Returns per-node action picks + judge score."""
    thread_id = f"{task.id}_iter{iter_idx}_{uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    initial_state = {
        "user_task": task.question,
        "revision_count": 0,
        "trace": [],
    }
    t_start = time.monotonic()
    if use_stream:
        # Exercise the streaming path — we still just need the terminal state.
        final_state = None
        async for event in compiled.astream(initial_state, config=config):
            for _node_name, node_state in event.items():
                final_state = {**(final_state or {}), **node_state}
    else:
        final_state = await compiled.ainvoke(initial_state, config=config)
    latency = time.monotonic() - t_start

    final_answer = (final_state or {}).get("final_answer", "")
    quality, axes, reasoning = score_answer(task, final_answer)
    await arecord_reward(
        quality=quality, thread_id=thread_id, axes=axes, reasoning=reasoning
    )
    return {
        "thread_id": thread_id,
        "task_id": task.id,
        "difficulty": task.difficulty,
        "actions": [t for t in (final_state or {}).get("trace", [])],
        "revisions": (final_state or {}).get("revision_count", 0),
        "final_answer": final_answer,
        "quality": quality,
        "quality_axes": axes,
        "quality_reasoning": reasoning,
        "latency_s": latency,
    }


def _print_run(i: int, total: int, r: dict) -> None:
    print(
        f"  [{i:>3}/{total}] {r['task_id']:<28} diff={r['difficulty']:<6} "
        f"q={r['quality']:.2f}  lat={r['latency_s']:.1f}s  revs={r['revisions']}"
    )
    for step in r["actions"]:
        print(f"        {step['node']:<10} → {step['action']}")


def _summary(results: list[dict]) -> None:
    print("\n  ══════ Summary ══════")
    picks_by_sig: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        for step in r["actions"]:
            picks_by_sig[step["node"]][step["action"]] += 1
    for sig, action_counts in picks_by_sig.items():
        total = sum(action_counts.values())
        best_action, best_count = max(action_counts.items(), key=lambda x: x[1])
        share = 100.0 * best_count / total
        others = ", ".join(f"{a}={c}" for a, c in action_counts.items() if a != best_action)
        print(f"    {sig:<10} argmax={best_action:<10} ({share:>4.0f}%, n={total})   {others}")
    q_by_diff: dict[str, list[float]] = defaultdict(list)
    for r in results:
        q_by_diff[r["difficulty"]].append(r["quality"])
    print("")
    for diff, qs in q_by_diff.items():
        mean = sum(qs) / len(qs)
        print(f"    quality[{diff:<6}] mean={mean:.2f}  (n={len(qs)})")


async def _export_policy(server_url: str, tenant_key: str, out_path: Path) -> None:
    """Read the tenant's current policy from `/policy/me` and write to JSON."""
    import httpx

    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.get(
            f"{server_url.rstrip('/')}/policy/me",
            headers={"Authorization": f"Bearer {tenant_key}"},
        )
        r.raise_for_status()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(r.json(), indent=2))
    print(f"\n  ✓ policy exported to {out_path}")


async def main() -> int:
    ap = argparse.ArgumentParser(prog="evidence_heavy_mas")
    ap.add_argument("--tasks", type=int, default=5, help="how many tasks to use (max 5)")
    ap.add_argument("--runs", type=int, default=3, help="iterations per task")
    ap.add_argument("--stream", action="store_true", help="use astream() path")
    ap.add_argument(
        "--export-policy",
        type=Path,
        default=None,
        help="write the tenant's policy JSON to this path after all runs",
    )
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("  OPENROUTER_API_KEY not set — needed for real LLM calls", file=sys.stderr)
        return 2
    if not os.environ.get("AGENSFLOW_API_KEY"):
        print("  AGENSFLOW_API_KEY not set — needed to talk to the policy server", file=sys.stderr)
        return 2

    pools = build_pools()
    compiled = build_graph(pools)
    selected_tasks = TASKS[: args.tasks]
    total = len(selected_tasks) * args.runs

    print(
        f"  running {len(selected_tasks)} task(s) × {args.runs} iter(s) = {total} graph invocations"
    )
    print(f"  policy server: {os.environ.get('AGENSFLOW_SERVER_URL', 'http://localhost:8000')}")
    print(f"  stream mode:   {'yes' if args.stream else 'no (ainvoke)'}\n")

    results = []
    idx = 0
    for i_run in range(args.runs):
        for task in selected_tasks:
            idx += 1
            r = await _run_one(compiled, task, i_run, args.stream)
            results.append(r)
            _print_run(idx, total, r)

    _summary(results)

    if args.export_policy:
        await _export_policy(
            os.environ.get("AGENSFLOW_SERVER_URL", "http://localhost:8000"),
            os.environ["AGENSFLOW_API_KEY"],
            args.export_policy,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
