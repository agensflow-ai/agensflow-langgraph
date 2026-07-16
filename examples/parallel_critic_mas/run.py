"""CLI for the parallel_critic MAS. Same shape as evidence_heavy_mas/run.py.

    export OPENROUTER_API_KEY=sk-or-...
    export AGENSFLOW_SERVER_URL=http://localhost:8000
    export AGENSFLOW_API_KEY=agf_...

    python -m examples.parallel_critic_mas.run --tasks 5 --runs 8
    python -m examples.parallel_critic_mas.run --tasks 5 --runs 8 --export-policy examples/starter_policies/parallel_critic_v1.json
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from uuid import uuid4

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from examples.parallel_critic_mas.graph import build_graph, build_pools  # type: ignore[no-redef]
    from examples.parallel_critic_mas.tasks import TASKS, score_answer  # type: ignore[no-redef]
else:
    from .graph import build_graph, build_pools
    from .tasks import TASKS, score_answer

from agensflow_langgraph import arecord_reward, export_policy


async def _run_one(compiled, task, iter_idx: int) -> dict:
    thread_id = f"{task.id}_iter{iter_idx}_{uuid4().hex[:8]}"
    config = {"configurable": {"thread_id": thread_id}}
    initial = {"user_task": task.question, "trace": []}
    t_start = time.monotonic()
    final = await compiled.ainvoke(initial, config=config)
    latency = time.monotonic() - t_start

    quality, axes, reasoning = score_answer(task, final.get("final_answer", ""))
    await arecord_reward(quality=quality, thread_id=thread_id, axes=axes,
                          reasoning=reasoning)
    return {
        "task_id": task.id,
        "difficulty": task.difficulty,
        "actions": final.get("trace", []),
        "final_answer": final.get("final_answer", ""),
        "quality": quality,
        "latency_s": latency,
    }


def _print_run(i: int, total: int, r: dict) -> None:
    print(
        f"  [{i:>3}/{total}] {r['task_id']:<28} diff={r['difficulty']:<6} "
        f"q={r['quality']:.2f}  lat={r['latency_s']:.1f}s"
    )
    for step in r["actions"]:
        print(f"        {step['node']:<10} → {step['action']}")


def _summary(results: list[dict]) -> None:
    print("\n  ══════ Summary ══════")
    picks: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for r in results:
        for step in r["actions"]:
            picks[step["node"]][step["action"]] += 1
    for sig, counts in picks.items():
        total = sum(counts.values())
        best, best_n = max(counts.items(), key=lambda x: x[1])
        others = ", ".join(f"{a}={c}" for a, c in counts.items() if a != best)
        print(f"    {sig:<10} argmax={best:<10} ({100*best_n/total:>4.0f}%, n={total})   {others}")
    q_by_diff: dict[str, list[float]] = defaultdict(list)
    for r in results:
        q_by_diff[r["difficulty"]].append(r["quality"])
    for d, qs in q_by_diff.items():
        print(f"    quality[{d:<6}] mean={sum(qs)/len(qs):.2f} (n={len(qs)})")


async def main() -> int:
    ap = argparse.ArgumentParser(prog="parallel_critic_mas")
    ap.add_argument("--tasks", type=int, default=5)
    ap.add_argument("--runs", type=int, default=3)
    ap.add_argument("--export-policy", type=Path, default=None)
    args = ap.parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        print("  OPENROUTER_API_KEY not set", file=sys.stderr)
        return 2
    if not os.environ.get("AGENSFLOW_API_KEY"):
        print("  AGENSFLOW_API_KEY not set", file=sys.stderr)
        return 2

    pools = build_pools()
    compiled = build_graph(pools)
    tasks = TASKS[: args.tasks]
    total = len(tasks) * args.runs

    print(f"  parallel_critic MAS: {len(tasks)} task(s) × {args.runs} iter(s) = {total}")
    print(f"  policy server: {os.environ.get('AGENSFLOW_SERVER_URL', 'http://localhost:8000')}\n")

    results = []
    idx = 0
    for i in range(args.runs):
        for task in tasks:
            idx += 1
            r = await _run_one(compiled, task, i)
            results.append(r)
            _print_run(idx, total, r)

    _summary(results)

    if args.export_policy:
        info = export_policy(args.export_policy)
        print(f"\n  ✓ exported {info['n_signatures']} signatures / "
              f"{info['n_actions']} actions → {info['path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
