"""Verify the paper-translated warm-start actually converges + adapts sensibly.

Runs on top of the imported priors from `examples/starter_policies/security_v1.json`
(which came from `translate_paper_pkl.py`). This is the pre-ship proof point:
does the substrate act coherently when we drive real tasks through the modern
model bindings, or does the transfer break under a model swap?

Two invocation modes:

    # Smoke — 20 tasks × 1 epoch, ~$1-3, ~5-10 min
    python -m examples.security_domain.verify_warmstart --tasks 20 --epochs 1

    # Real convergence — 60 tasks × 2 epochs, ~$6-12, ~30-60 min
    python -m examples.security_domain.verify_warmstart --tasks all --epochs 2

Outputs a full JSON report at `examples/security_domain/reports/verify_<ts>.json`
plus a human-readable summary at end of run, including:

  * per-arm delta (visits Δ + reward_mean Δ vs paper priors)
  * per-scenario-class routing distribution (which arms won for C1 vs C7 etc.)
  * pathological-arm check (dead arms, high failure_count)
  * judge stability (per-task axis disagreement across 3 panel members)
  * rank-change vs paper (paper's best solver arm still best under modern models?)
  * PASS/FAIL banner

Hard PASS/FAIL checks (in _hard_checks below):
  * every arm gets at least 1 visit
  * no arm ends at reward_mean=0.0 with 3+ visits
  * failure_count / visits < 0.5 for every arm
  * at least one revision loop triggered somewhere in the run

Reads OPENROUTER_API_KEY from env or repo-root .env
(agensflow-langgraph/.env).
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import json
import os
import statistics
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
DEFAULT_STARTER = REPO_ROOT / "examples" / "starter_policies" / "security_v1.json"
REPORTS_DIR = THIS_DIR / "reports"


PANEL_MODELS = (
    "x-ai/grok-4.3",
    "openai/gpt-5.4-mini",
    "qwen/qwen3.6-flash",
)
BASELINE_MODEL = "thinkingmachines/inkling"


def _budget_estimate(n_tasks: int, epochs: int) -> tuple[float, float]:
    """Rough $ estimate. Assumes ~$0.05 avg per task-run, ~$0.03 per judge, ~$0.01 per baseline."""
    task_runs = n_tasks * epochs
    low = task_runs * (0.03 + len(PANEL_MODELS) * 0.02 + 0.005)
    high = task_runs * (0.10 + len(PANEL_MODELS) * 0.05 + 0.02)
    return low, high


async def _boot_server():
    os.environ.setdefault("AGF_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    os.environ.setdefault("AGF_ENV", "test")
    os.environ.setdefault("AGF_JWT_SECRET", "verify-not-for-production")

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

    class _VerifyClient(agf_client.AgensFlowClient):
        async def _a_post_model(self, path, payload, model_cls):
            r = await server_client.post(path, json=payload, headers=self._headers)
            self._raise_for_status(r)
            return model_cls.model_validate(r.json())

        async def _a_get_model(self, path, model_cls, params=None):
            r = await server_client.get(path, params=params, headers=self._headers)
            self._raise_for_status(r)
            return model_cls.model_validate(r.json())

    agf_client._CACHE.clear()
    agf_client.AgensFlowClient = _VerifyClient
    os.environ["AGENSFLOW_SERVER_URL"] = "http://test"
    os.environ["AGENSFLOW_API_KEY"] = api_key

    return app, server_client, lifespan_mgr, api_key


async def _baseline_of(task_text: str, key: str) -> str:
    from langchain_openai import ChatOpenAI

    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=key,
        model=BASELINE_MODEL, temperature=0.0,
        default_headers={
            "HTTP-Referer": "https://agensflow.ai",
            "X-Title": "AgensFlow security_domain verify",
        },
    )
    msg = await llm.ainvoke([
        ("system",
         "Answer concisely using only the provided context. If none is given, "
         "answer from general knowledge in one paragraph."),
        ("human", task_text),
    ])
    return msg.content


async def _judge(task_text: str, cand: str, base: str, key: str) -> tuple[float, dict, dict]:
    """Return (composed_quality, per_axis, per_judge_scores)."""
    from agensflow_langgraph.judge_panel import relative_quality, _judge as _one_judge

    # Also gather per-judge scores individually so we can measure agreement
    per_judge: dict[str, dict] = {}
    for m in PANEL_MODELS:
        try:
            result = await _one_judge(key, m, task_text, cand, base)
            per_judge[m] = {
                "verdict": result.get("verdict"),
                "Q_candidate": result.get("Q_candidate"),
                "axis_means_candidate": result.get("axis_means_candidate", {}),
            }
        except Exception as e:
            per_judge[m] = {"verdict": "error", "error": str(e)[:200]}

    q, per_axis = await relative_quality(
        task=task_text, candidate=cand, baseline=base,
        openrouter_key=key, models=PANEL_MODELS,
    )
    return q, per_axis, per_judge


async def _snapshot():
    from agensflow_langgraph.client import get_client
    resp = await get_client().a_export_policy()
    return copy.deepcopy(resp.policy)


def _compute_delta(before: dict, after: dict) -> list[dict]:
    """Return per-arm delta rows sorted by node then arm-key."""
    rows: list[dict] = []
    all_nodes = sorted(set(before) | set(after))
    for node in all_nodes:
        b_arms = before.get(node, {})
        a_arms = after.get(node, {})
        for arm in sorted(set(b_arms) | set(a_arms)):
            b = b_arms.get(arm, {})
            a = a_arms.get(arm, {})
            vb = int(b.get("visits", 0))
            va = int(a.get("visits", 0))
            mb = float(b.get("reward_mean", 0.0))
            ma = float(a.get("reward_mean", 0.0))
            fb = int(b.get("failure_count", 0))
            fa = int(a.get("failure_count", 0))
            rows.append({
                "node": node, "arm": arm,
                "visits_before": vb, "visits_after": va, "visits_delta": va - vb,
                "reward_mean_before": mb, "reward_mean_after": ma,
                "reward_mean_delta": ma - mb,
                "failure_before": fb, "failure_after": fa,
                "failure_delta": fa - fb,
            })
    return rows


def _hard_checks(rows: list[dict], task_metrics: list[dict], epochs: int) -> tuple[bool, list[str]]:
    """Return (all_pass, list_of_issues). Scale-aware: doesn't demand exhaustive
    arm coverage when task count is smaller than the action space."""
    issues: list[str] = []
    from collections import defaultdict as _dd

    by_node: dict[str, list[dict]] = _dd(list)
    for r in rows:
        by_node[r["node"]].append(r)

    total_tasks = len(task_metrics)

    # 1. Each pool must have at least ceil(min(pool_size, N/2)) arms explored.
    #    For a 9-arm pool over 20 tasks, expect at least 5 arms tried;
    #    over 120 tasks, all 9. Below that = pipeline suspicion.
    for node, arms in by_node.items():
        explored = sum(1 for a in arms if a["visits_delta"] > 0)
        pool = len(arms)
        # min explored we'd expect at this task scale
        expected = min(pool, max(1, total_tasks // 4))
        if explored < expected:
            issues.append(
                f"POOL {node}: only {explored}/{pool} arms explored "
                f"in {total_tasks} runs (expected ≥{expected})"
            )

    # 2. UCB1 sanity: an unexplored arm should not have HIGHER paper prior than
    #    the LOWEST-prior explored arm. If it does, exploration isn't happening
    #    where it should. Tolerance 0.02 for exploration-bonus noise.
    for node, arms in by_node.items():
        unvisited = [a for a in arms if a["visits_delta"] == 0]
        visited = [a for a in arms if a["visits_delta"] > 0]
        if not unvisited or not visited:
            continue
        max_unvisited_prior = max(a["reward_mean_before"] for a in unvisited)
        min_visited_prior = min(a["reward_mean_before"] for a in visited)
        if max_unvisited_prior > min_visited_prior + 0.02:
            skipped = next(a for a in unvisited if a["reward_mean_before"] == max_unvisited_prior)
            issues.append(
                f"UCB INVERSION in {node}: unvisited {skipped['arm']} "
                f"(prior {max_unvisited_prior:.3f}) beats explored arm at {min_visited_prior:.3f}"
            )

    # 3. Dead arms — 3+ visits, reward_mean=0
    dead = [f"{r['node']}/{r['arm']} (v={r['visits_after']}, μ=0)"
            for r in rows if r["visits_after"] >= 3 and r["reward_mean_after"] == 0.0]
    if dead:
        issues.append(f"DEAD arms: {', '.join(dead[:5])}")

    # 4. High failure rate — failure/visits > 0.5
    high_fail = [f"{r['node']}/{r['arm']} ({r['failure_after']}/{r['visits_after']})"
                 for r in rows if r["visits_after"] > 0
                 and r["failure_after"] / r["visits_after"] > 0.5]
    if high_fail:
        issues.append(f"HIGH-FAILURE arms: {', '.join(high_fail[:5])}")

    # 5. At least one revision loop triggered (verifier gate working) — multi-epoch only
    total_revs = sum(m.get("revisions", 0) for m in task_metrics)
    if epochs > 1 and total_revs == 0:
        issues.append(f"NO REVISIONS across {len(task_metrics)} runs — verifier_gate never triggered")

    # 6. Every panel judge scored >80% of tasks
    per_judge_success = {m: 0 for m in PANEL_MODELS}
    for tm in task_metrics:
        for m, j in (tm.get("per_judge") or {}).items():
            if j.get("verdict") == "scored":
                per_judge_success[m] += 1
    for m, s in per_judge_success.items():
        rate = s / total_tasks if total_tasks else 0
        if rate < 0.8:
            issues.append(f"JUDGE {m} scored only {s}/{total_tasks} tasks (<80%)")

    return (len(issues) == 0), issues


def _class_routing(task_metrics: list[dict]) -> dict[str, Counter]:
    """Per-scenario-class breakdown: which solver arms won."""
    per_class: dict[str, Counter] = defaultdict(Counter)
    for tm in task_metrics:
        sc = tm.get("scenario_class")
        actions = {step["node"]: step["action"] for step in tm.get("trace", [])}
        if sc and "solver" in actions:
            per_class[sc][actions["solver"]] += 1
    return per_class


def _judge_agreement(task_metrics: list[dict]) -> dict[str, float]:
    """Per-task max-min disagreement across panel judges."""
    disagreements: list[float] = []
    for tm in task_metrics:
        pj = tm.get("per_judge") or {}
        scores = [j.get("Q_candidate") for j in pj.values()
                  if isinstance(j.get("Q_candidate"), (int, float))]
        if len(scores) >= 2:
            disagreements.append(max(scores) - min(scores))
    if not disagreements:
        return {"n": 0}
    return {
        "n": len(disagreements),
        "mean": statistics.mean(disagreements),
        "median": statistics.median(disagreements),
        "max": max(disagreements),
        "p90": sorted(disagreements)[int(len(disagreements) * 0.9)] if len(disagreements) > 1 else disagreements[0],
    }


# Transient errors we should retry with backoff (provider outage, rate limits,
# routing hiccups). Permanent errors (400 bad model, 401 bad auth) bypass retry.
_TRANSIENT_MARKERS = (
    "RateLimitError", "InternalServerError", "APITimeoutError",
    "APIConnectionError", "ConnectError", "ReadTimeout",
    " 429", " 502", " 503", " 504", " 520", " 522", " 524",
)


def _is_transient(exc: Exception) -> bool:
    label = f"{type(exc).__name__} {str(exc)}"
    return any(m in label for m in _TRANSIENT_MARKERS)


async def _retry(fn, *args, max_attempts: int = 4, base_delay: float = 3.0, cap: float = 45.0, label: str = ""):
    """Exponential-backoff retry on transient LLM errors. Bails immediately on permanent errors."""
    for attempt in range(1, max_attempts + 1):
        try:
            return await fn(*args)
        except Exception as e:
            if not _is_transient(e) or attempt == max_attempts:
                raise
            delay = min(cap, base_delay * (2 ** (attempt - 1))) + (attempt * 0.5)  # jittered
            print(f"      ↻ {label} transient ({type(e).__name__}), retry {attempt}/{max_attempts - 1} in {delay:.1f}s")
            await asyncio.sleep(delay)


async def _run_one(task, compiled, key: str) -> dict:
    from agensflow_langgraph import arecord_reward

    thread_id = f"verify_{task.id}_{int(time.time() * 1000)}"
    t0 = time.time()
    revisions = None

    async def _invoke_graph():
        return await compiled.ainvoke(
            {"user_task": task.user_task,
             "corpus_doc_ids": task.corpus_doc_ids,
             "trace": [], "revision_count": 0},
            config={"configurable": {"thread_id": thread_id}},
        )

    try:
        result = await _retry(_invoke_graph, label=f"{task.id} graph")
        revisions = result.get("revision_count", 0)
        answer = result["final_answer"]
        trace = result.get("trace", [])
    except Exception as e:
        return {
            "task_id": task.id, "scenario_class": task.scenario_class,
            "status": "graph_error", "error": f"{type(e).__name__}: {str(e)[:200]}",
            "elapsed_s": time.time() - t0,
        }

    try:
        baseline = await _retry(_baseline_of, task.user_task, key, label=f"{task.id} baseline")
    except Exception as e:
        return {
            "task_id": task.id, "scenario_class": task.scenario_class,
            "status": "baseline_error", "error": f"{type(e).__name__}: {str(e)[:200]}",
            "elapsed_s": time.time() - t0,
        }

    try:
        quality, per_axis, per_judge = await _retry(
            _judge, task.user_task, answer, baseline, key, label=f"{task.id} judge"
        )
    except Exception as e:
        return {
            "task_id": task.id, "scenario_class": task.scenario_class,
            "status": "judge_error", "error": f"{type(e).__name__}: {str(e)[:200]}",
            "elapsed_s": time.time() - t0,
        }

    await arecord_reward(quality=quality, thread_id=thread_id)

    return {
        "task_id": task.id, "scenario_class": task.scenario_class,
        "status": "ok", "quality": quality,
        "per_axis": per_axis, "per_judge": per_judge,
        "trace": trace, "revisions": revisions,
        "answer_len": len(answer), "baseline_len": len(baseline),
        "elapsed_s": time.time() - t0,
    }


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=str, default="20",
                    help="'all' | integer count | csv task ids like 'C1.1,C2.2' | class 'C1'")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--starter", type=Path, default=DEFAULT_STARTER)
    ap.add_argument("--snapshot-every", type=int, default=15)
    args = ap.parse_args()

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
        print(
            f"ERROR: OPENROUTER_API_KEY not set.\n"
            f"  Put it in the repo-root .env at: {REPO_ROOT / '.env'}",
            file=sys.stderr,
        )
        sys.exit(1)

    # Verify starter is present
    if not args.starter.exists():
        print(f"ERROR: starter file not found at {args.starter}\n"
              f"  Generate with: python -m examples.security_domain.translate_paper_pkl",
              file=sys.stderr)
        sys.exit(2)

    from examples.security_domain.tasks import ALL_TASKS
    from examples.security_domain.graph import build_pools, build_graph

    # Task selection
    if args.tasks == "all":
        tasks = list(ALL_TASKS)
    elif args.tasks.startswith("C") and "." not in args.tasks and "," not in args.tasks:
        tasks = [t for t in ALL_TASKS if t.scenario_class == args.tasks]
    elif "," in args.tasks or "." in args.tasks:
        wanted = set(args.tasks.split(","))
        tasks = [t for t in ALL_TASKS if t.id in wanted]
    else:
        try:
            n = int(args.tasks)
        except ValueError:
            print(f"ERROR: --tasks must be 'all', an integer, a CSV of task ids, or a class like 'C1'",
                  file=sys.stderr)
            sys.exit(3)
        # Take one task per scenario class, cycle until we have n
        by_class: dict[str, list] = defaultdict(list)
        for t in ALL_TASKS:
            by_class[t.scenario_class].append(t)
        classes = sorted(by_class)
        tasks = []
        while len(tasks) < n:
            for c in classes:
                if by_class[c]:
                    tasks.append(by_class[c].pop(0))
                    if len(tasks) == n:
                        break

    if not tasks:
        print("ERROR: no tasks selected", file=sys.stderr)
        sys.exit(4)

    print()
    print(f"  ┌─── verify_warmstart ──────────────────────────────────")
    print(f"  │ starter:  {args.starter.name}")
    print(f"  │ tasks:    {len(tasks)}  ({args.tasks})")
    print(f"  │ epochs:   {args.epochs}")
    print(f"  │ judges:   {len(PANEL_MODELS)} panel ({', '.join(PANEL_MODELS)})")
    print(f"  │ baseline: {BASELINE_MODEL}")
    low, high = _budget_estimate(len(tasks), args.epochs)
    print(f"  │ estimate: ${low:.2f} — ${high:.2f}")
    print(f"  └──────────────────────────────────────────────────────")
    print(f"  press CTRL-C in the next 5s to abort")
    time.sleep(5)

    # Boot + import
    app, server_client, lifespan_mgr, api_key = await _boot_server()
    from agensflow_langgraph import aimport_policy
    imp = await aimport_policy(args.starter)
    print(f"  ✓ imported starter: {imp['signatures_merged']} sigs / {imp['actions_merged']} arms")

    # Snapshot BEFORE
    policy_before = await _snapshot()
    print(f"  ✓ policy_before snapshotted")

    # Build graph
    pools = build_pools()
    compiled = build_graph(pools)

    # Run
    task_metrics: list[dict] = []
    started = time.time()
    idx = 0
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    report_path = REPORTS_DIR / f"verify_{ts}.json"

    INTER_TASK_DELAY_S = 3.0   # small pause between tasks — reduces rate-limit hits

    print()
    for epoch in range(1, args.epochs + 1):
        print(f"  === epoch {epoch}/{args.epochs} ({len(tasks)} tasks) ===")
        for t in tasks:
            idx += 1
            m = await _run_one(t, compiled, key)
            task_metrics.append({"epoch": epoch, "idx": idx, **m})
            if m["status"] == "ok":
                print(f"    #{idx:03d}  {t.id:<6} {t.scenario_class}  q={m['quality']:.3f}  "
                      f"revs={m['revisions'] or 0}  {m['elapsed_s']:5.1f}s  ")
            else:
                print(f"    #{idx:03d}  {t.id:<6} {t.scenario_class}  {m['status']}: "
                      f"{m.get('error', '?')[:80]}")

            if idx % args.snapshot_every == 0:
                snap = await _snapshot()
                snapshot_path = REPORTS_DIR / f"snap_{ts}_{idx:04d}.json"
                snapshot_path.write_text(json.dumps({"policy": snap}, indent=2))

            # Brief pause between tasks — gentler on rate limits
            if idx < len(tasks) * args.epochs:
                await asyncio.sleep(INTER_TASK_DELAY_S)

    # Snapshot AFTER
    policy_after = await _snapshot()
    delta_rows = _compute_delta(policy_before, policy_after)
    hard_pass, hard_issues = _hard_checks(delta_rows, task_metrics, args.epochs)
    class_routing = _class_routing(task_metrics)
    judge_agreement = _judge_agreement(task_metrics)

    ok_count = sum(1 for m in task_metrics if m["status"] == "ok")
    err_count = len(task_metrics) - ok_count
    mean_q = statistics.mean([m["quality"] for m in task_metrics if m["status"] == "ok"]) if ok_count else 0.0
    total_elapsed = time.time() - started

    report = {
        "meta": {
            "timestamp": ts,
            "starter": str(args.starter),
            "n_tasks": len(tasks),
            "n_epochs": args.epochs,
            "total_runs": len(task_metrics),
            "ok_count": ok_count,
            "err_count": err_count,
            "mean_quality": mean_q,
            "total_elapsed_s": total_elapsed,
            "panel_models": list(PANEL_MODELS),
            "baseline_model": BASELINE_MODEL,
        },
        "hard_pass": hard_pass,
        "hard_issues": hard_issues,
        "delta": delta_rows,
        "class_routing": {c: dict(v) for c, v in class_routing.items()},
        "judge_agreement": judge_agreement,
        "task_metrics": task_metrics,
        "policy_before": policy_before,
        "policy_after": policy_after,
    }
    report_path.write_text(json.dumps(report, indent=2))

    # ---- Console summary ----
    print()
    print(f"  ┌─── verify_warmstart complete ────────────────────────")
    print(f"  │ ok / total:  {ok_count} / {len(task_metrics)}")
    print(f"  │ mean quality (ok runs): {mean_q:.3f}")
    print(f"  │ wall clock: {total_elapsed:.0f}s")
    print(f"  │ report:     {report_path}")
    print(f"  └──────────────────────────────────────────────────────")

    print()
    print(f"  === per-arm delta ===")
    print(f"    {'node':<10} {'arm':<20} {'visits Δ':>10} {'μ before':>10} {'μ after':>10} {'μ Δ':>10}")
    for r in delta_rows:
        arrow = "↑" if r["reward_mean_delta"] > 0.005 else ("↓" if r["reward_mean_delta"] < -0.005 else "·")
        print(f"    {r['node']:<10} {r['arm']:<20} {r['visits_delta']:>+10} "
              f"{r['reward_mean_before']:>10.3f} {r['reward_mean_after']:>10.3f} "
              f"{r['reward_mean_delta']:>+10.3f} {arrow}")

    if class_routing:
        print()
        print(f"  === per-scenario-class solver-arm distribution ===")
        for sc in sorted(class_routing):
            top = class_routing[sc].most_common(3)
            print(f"    {sc}  → {', '.join(f'{k}={v}' for k, v in top)}")

    if judge_agreement.get("n"):
        print()
        print(f"  === 3-panel judge agreement ===")
        ja = judge_agreement
        print(f"    per-task max-min score gap: mean={ja['mean']:.3f}  "
              f"median={ja['median']:.3f}  p90={ja['p90']:.3f}  max={ja['max']:.3f}")

    print()
    banner = " PASS " if hard_pass else " FAIL "
    marker = "═" * 20
    print(f"  {marker}{banner}{marker}")
    if not hard_pass:
        for issue in hard_issues:
            print(f"    ✗ {issue}")
    print()

    await server_client.aclose()
    await lifespan_mgr.__aexit__(None, None, None)

    sys.exit(0 if hard_pass else 5)


if __name__ == "__main__":
    asyncio.run(main())
