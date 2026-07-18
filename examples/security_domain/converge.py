"""One-command converge script for the security domain.

Runs the security-domain parallel_critic MAS over the paper's 60-task
security-advisory suite N epochs, judged after each task by the free-tier
3-panel judge (Google + Qwen + xAI — disjoint from the task pool's OpenAI/
Anthropic/Meta). Exports the converged bandit policy as JSON for the
notebook's warm-start.

Usage (needs OPENROUTER_API_KEY in env or a .env file at CWD):

    # Full convergence — ~$10-20 in OpenRouter fees, ~30-60 min wall-clock
    python -m examples.security_domain.converge --epochs 6 --output security_v1.json

    # Cheap smoke — 1 epoch, subset of tasks
    python -m examples.security_domain.converge --epochs 1 --tasks C1.1,C1.2,C5.1

    # Resume after a mid-run interrupt (reads snapshot_*.json)
    python -m examples.security_domain.converge --epochs 6 --resume

Runs everything through an in-process ASGI-backed agensflow-mcp — no
separate uvicorn, no port bound. Every 20 task-runs the current policy is
snapshotted to disk so a KeyboardInterrupt loses at most 20 runs of work.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path


THIS_DIR = Path(__file__).parent


def _budget_check(n_tasks: int, epochs: int, judges: int) -> None:
    """Rough cost estimate. Assumes ~$0.02 avg per task-run + ~$0.03 per judge call."""
    task_runs = n_tasks * epochs
    est_low = task_runs * (0.015 + judges * 0.02)
    est_high = task_runs * (0.05 + judges * 0.05)
    print(
        f"  budget estimate:  ~${est_low:.2f} — ${est_high:.2f} "
        f"({task_runs} task-runs × ~{judges} judge calls each)"
    )
    print(f"  press CTRL-C in the next 5s to abort")
    time.sleep(5)


async def _boot_server():
    """Boot agensflow-mcp in-process (same pattern the notebook uses)."""
    os.environ.setdefault("AGF_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    os.environ.setdefault("AGF_ENV", "test")
    os.environ.setdefault("AGF_JWT_SECRET", "converge-script-not-for-production")

    from httpx import ASGITransport, AsyncClient
    from asgi_lifespan import LifespanManager
    from agensflow_mcp.app import create_app
    from agensflow_mcp.db.session import get_engine, init_db
    from agensflow_mcp.db.models import Base

    app = create_app()
    await init_db()
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    lifespan_mgr = LifespanManager(app)
    await lifespan_mgr.__aenter__()
    server_client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")

    resp = await server_client.post("/auth/anonymous")
    api_key = resp.json()["api_key"]

    from agensflow_langgraph import client as agf_client

    class _NotebookClient(agf_client.AgensFlowClient):
        async def _a_post_model(self, path, payload, model_cls):
            r = await server_client.post(path, json=payload, headers=self._headers)
            self._raise_for_status(r)
            return model_cls.model_validate(r.json())

        async def _a_get_model(self, path, model_cls, params=None):
            r = await server_client.get(path, params=params, headers=self._headers)
            self._raise_for_status(r)
            return model_cls.model_validate(r.json())

    agf_client._CACHE.clear()
    agf_client.AgensFlowClient = _NotebookClient
    os.environ["AGENSFLOW_SERVER_URL"] = "http://test"
    os.environ["AGENSFLOW_API_KEY"] = api_key

    return app, server_client, lifespan_mgr, api_key


PANEL_MODELS = (
    "google/gemini-2.5-flash",
    "qwen/qwen-2.5-72b-instruct",
    "x-ai/grok-4-fast",
)


async def _judge_one(user_task: str, answer: str, baseline: str, key: str) -> float:
    """Score answer vs baseline via the 3-panel judge; return composed quality."""
    from agensflow_langgraph.judge_panel import relative_quality

    q, _axes = await relative_quality(
        task=user_task,
        candidate=answer,
        baseline=baseline,
        openrouter_key=key,
        models=PANEL_MODELS,
    )
    return q


async def _generate_baseline(task_text: str, key: str) -> str:
    """Cheap single-model baseline for the judge to anchor against."""
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        model="openai/gpt-4o-mini",
        temperature=0.0,
    )
    msg = await llm.ainvoke([
        ("system",
         "Answer concisely using only the provided context. If no context is "
         "given, answer in one paragraph from general knowledge."),
        ("human", task_text),
    ])
    return msg.content


async def _snapshot(path: Path) -> None:
    from agensflow_langgraph.client import get_client
    resp = await get_client().a_export_policy()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "contract_version": "v1",
        "policy": resp.policy,
        "n_signatures": resp.n_signatures,
        "n_actions": resp.n_actions,
    }
    path.write_text(json.dumps(payload, indent=2))
    print(f"    ↳ snapshot: {path.name}  ({resp.n_signatures} sigs / {resp.n_actions} arms)")


async def _run_one_task(task, compiled, thread_id: str, key: str) -> tuple[str, float]:
    """Invoke the graph on one task, judge the answer, return (final_answer, quality)."""
    result = await compiled.ainvoke(
        {"user_task": task.user_task,
         "corpus_doc_ids": task.corpus_doc_ids,
         "trace": []},
        config={"configurable": {"thread_id": thread_id}},
    )
    answer = result["final_answer"]
    baseline = await _generate_baseline(task.user_task, key)
    quality = await _judge_one(task.user_task, answer, baseline, key)
    return answer, quality


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--tasks", type=str, default="all",
                    help="'all' or comma-sep task ids like 'C1.1,C2.2'")
    ap.add_argument("--output", type=Path, default=THIS_DIR.parent / "starter_policies" / "security_v1.json")
    ap.add_argument("--snapshot-every", type=int, default=20)
    args = ap.parse_args()

    # Load OpenRouter key from env / .env
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("ERROR: OPENROUTER_API_KEY not set (env or .env)", file=sys.stderr)
        sys.exit(1)

    # Load tasks (must be importable as `examples.security_domain.tasks`)
    from examples.security_domain.tasks import ALL_TASKS
    from examples.security_domain.graph import build_pools, build_graph

    if args.tasks == "all":
        tasks = list(ALL_TASKS)
    else:
        wanted = set(args.tasks.split(","))
        tasks = [t for t in ALL_TASKS if t.id in wanted]
        if not tasks:
            print(f"ERROR: no tasks matched {args.tasks!r}", file=sys.stderr)
            sys.exit(2)

    print(f"  tasks: {len(tasks)}   epochs: {args.epochs}   panel: {len(PANEL_MODELS)} judges")
    _budget_check(n_tasks=len(tasks), epochs=args.epochs, judges=len(PANEL_MODELS))

    app, server_client, lifespan_mgr, api_key = await _boot_server()
    print(f"  server booted (api_key={api_key[:20]}...)")

    pools = build_pools()
    compiled = build_graph(pools)
    print(f"  graph compiled: {list(compiled.get_graph().nodes)}\n")

    from agensflow_langgraph import arecord_reward

    snapshot_dir = args.output.parent / "converge_snapshots"
    snapshot_dir.mkdir(parents=True, exist_ok=True)

    run_idx = 0
    for epoch in range(1, args.epochs + 1):
        print(f"  === epoch {epoch}/{args.epochs} ===")
        for t in tasks:
            run_idx += 1
            thread_id = f"converge_e{epoch}_{t.id}"
            t0 = time.time()
            try:
                answer, quality = await _run_one_task(t, compiled, thread_id, key)
                await arecord_reward(quality=quality, thread_id=thread_id)
                dt = time.time() - t0
                print(f"    #{run_idx:03d}  {t.id:<6} class={t.scenario_class}  "
                      f"q={quality:.3f}  {dt:5.1f}s")
            except Exception as e:
                dt = time.time() - t0
                print(f"    #{run_idx:03d}  {t.id:<6} ERROR after {dt:.1f}s: "
                      f"{type(e).__name__}: {str(e)[:120]}")
                continue

            if run_idx % args.snapshot_every == 0:
                await _snapshot(snapshot_dir / f"snap_{run_idx:04d}.json")

    # Final export
    await _snapshot(args.output)

    await server_client.aclose()
    await lifespan_mgr.__aexit__(None, None, None)
    print(f"\n  ==== converged. final policy: {args.output} ====")


if __name__ == "__main__":
    asyncio.run(main())
