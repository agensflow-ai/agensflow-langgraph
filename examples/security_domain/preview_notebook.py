"""Headless run of the notebook's Sections 4/6/7/8 for fast iteration.

Mirrors what quickstart_adapted.ipynb executes end-to-end (skipping cleanup +
prose cells), so you can iterate on Section 6-8 content without opening
Jupyter. Same cost as one Colab click (~$0.35 with the adapted starter).

Usage:
    # against security_v1_adapted.json (default — strongest per-class signal)
    python -m examples.security_domain.preview_notebook

    # or against paper-only priors:
    python -m examples.security_domain.preview_notebook --starter security_v1.json

    # subset one section for cheaper iteration:
    python -m examples.security_domain.preview_notebook --only 6      # just the routing table (free)
    python -m examples.security_domain.preview_notebook --only 7      # + 3 per-class tasks (~$0.15)
    python -m examples.security_domain.preview_notebook --only 8      # + Pareto (~$0.20)

Reads OPENROUTER_API_KEY from repo-root .env.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent


PANEL_MODELS = ("x-ai/grok-4.3", "openai/gpt-5.4-mini", "qwen/qwen3.6-flash")


# Approximate OpenRouter prices per 1M tokens (USD). Sourced from OpenRouter's
# public pricing pages; a few cents off is fine for the Pareto demo's
# relative-cost comparison. Update as prices change.
PRICING = {
    "anthropic/claude-haiku-4.5":       {"in": 0.80,  "out": 4.00},
    "anthropic/claude-sonnet-5":        {"in": 3.00,  "out": 15.00},
    "thinkingmachines/inkling":         {"in": 0.15,  "out": 0.60},
    "openai/gpt-5.4-nano":              {"in": 0.10,  "out": 0.40},
    "openai/gpt-5.4-mini":              {"in": 0.30,  "out": 1.20},
}


def _approx_cost(model_id: str, tokens_in: int, tokens_out: int) -> float:
    """Fallback cost calc: tokens × per-model rate. Returns 0 if model unknown."""
    p = PRICING.get(model_id)
    if not p:
        return 0.0
    return (tokens_in * p["in"] + tokens_out * p["out"]) / 1_000_000


async def _boot_server():
    os.environ.setdefault("AGF_DATABASE_URL", "sqlite+aiosqlite:///:memory:")
    os.environ.setdefault("AGF_ENV", "test")
    os.environ.setdefault("AGF_JWT_SECRET", "preview-not-for-production")

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

    class _PreviewClient(agf_client.AgensFlowClient):
        async def _a_post_model(self, path, payload, model_cls):
            r = await server_client.post(path, json=payload, headers=self._headers)
            self._raise_for_status(r)
            return model_cls.model_validate(r.json())

        async def _a_get_model(self, path, model_cls, params=None):
            r = await server_client.get(path, params=params, headers=self._headers)
            self._raise_for_status(r)
            return model_cls.model_validate(r.json())

    agf_client._CACHE.clear()
    agf_client.AgensFlowClient = _PreviewClient
    os.environ["AGENSFLOW_SERVER_URL"] = "http://test"
    os.environ["AGENSFLOW_API_KEY"] = api_key
    return app, server_client, lifespan_mgr


async def _section_6():
    """Display current routing preferences per (node, regime)."""
    from collections import defaultdict
    from agensflow_langgraph.client import get_client

    print("\n" + "=" * 70)
    print("  SECTION 6  ·  What did the substrate actually learn?")
    print("=" * 70 + "\n")

    policy = (await get_client().a_export_policy()).policy

    # Signatures are `f"{node}:{regime}"` — group by node prefix
    by_node: dict[str, dict[str, dict]] = defaultdict(dict)
    for sig, arms in policy.items():
        if ":" in sig:
            node, regime = sig.split(":", 1)
        else:
            node, regime = sig, "default"
        by_node[node][regime] = arms

    node_order = ("planner", "memory", "solver", "verifier", "evaluator")
    for node in node_order:
        regimes = by_node.get(node, {})
        if not regimes:
            continue
        print(f"  {node}:")
        for regime in sorted(regimes):
            arms = regimes[regime]
            ranked = sorted(arms.items(), key=lambda kv: -kv[1].get("reward_mean", 0))
            print(f"    [{regime}]")
            for arm, stats in ranked:
                v = int(stats.get("visits", 0))
                m = stats.get("reward_mean", 0)
                marker = " ← preferred" if arm == ranked[0][0] else ""
                print(f"      {arm:<18} v={v:>3}  μ={m:.3f}{marker}")
        print()


async def _section_7(compiled, ALL_TASKS):
    """Run 3 different-class tasks through the substrate."""
    print("\n" + "=" * 70)
    print("  SECTION 7  ·  Per-task-class routing on real tasks")
    print("=" * 70 + "\n")

    demo_ids = ["C1.1", "C7.1", "C3.1"]
    records = []
    for tid in demo_ids:
        t = next(x for x in ALL_TASKS if x.id == tid)
        print(f"  ── Task {t.id} (class {t.scenario_class}) ──")
        print(f"  Q: {t.user_task[:110]}{'…' if len(t.user_task) > 110 else ''}\n")
        t0 = time.monotonic()
        r = await compiled.ainvoke(
            {"user_task": t.user_task, "corpus_doc_ids": t.corpus_doc_ids, "scenario_class": t.scenario_class, "trace": []},
            config={"configurable": {"thread_id": f"preview_7_{t.id}"}},
        )
        elapsed = time.monotonic() - t0
        routing = {step["node"]: step["action"] for step in r["trace"]}
        records.append({"task": t, "routing": routing, "answer": r["final_answer"], "elapsed_s": elapsed})
        print(f"  A: {r['final_answer'][:180]}{'…' if len(r['final_answer']) > 180 else ''}\n")
        print(f"  routing: " + " → ".join(f"{n}={routing.get(n, '?')}"
                                          for n in ("planner", "memory", "solver", "verifier", "evaluator")))
        print(f"  wall clock: {elapsed:.1f}s\n")

    print("  === Per-task routing summary ===")
    hdr = f'  {"task":<8} {"class":<7} ' + " ".join(f"{n:<12}" for n in ("planner", "memory", "solver", "verifier", "evaluator"))
    print(hdr)
    for rec in records:
        routing = rec["routing"]
        print(f"  {rec['task'].id:<8} {rec['task'].scenario_class:<7} " +
              " ".join(f"{routing.get(n, '?'):<12}" for n in ("planner", "memory", "solver", "verifier", "evaluator")))


async def _section_8(compiled, ALL_TASKS, key, api_key, server_client):
    """Substrate vs all-cheapest vs all-most-capable on one representative task."""
    from langchain_openai import ChatOpenAI
    from langgraph.graph import END, START, StateGraph
    from langgraph.checkpoint.memory import InMemorySaver
    from agensflow_langgraph.judge_panel import relative_quality
    from agensflow_langgraph.callbacks import CostCapture
    from examples.security_domain.graph import (
        SecurityMASState, _render_corpus_subset,
    )
    from examples.security_domain.prompts import (
        PLANNER_SYS, MEMORY_SYS, VERIFIER_SYS, EVALUATOR_SYS, SOLVER_SYSTEMS,
        EvidenceItem, PlannerOutput, MemoryOutput, SolverOutput, VerifierOutput,
        EvaluatorOutput, format_planner_input, format_memory_input,
        format_solver_input, format_verifier_input, format_evaluator_input,
    )

    print("\n" + "=" * 70)
    print("  SECTION 8  ·  Pareto — substrate vs all-cheapest vs all-most-capable")
    print("=" * 70 + "\n")

    # Some OpenRouter providers return errors on function_calling AND json_schema
    # (e.g. sonnet-5's stringified nested JSON). Cascade: fc → js → raw invoke +
    # manual JSON extraction + Pydantic construction. Only raise if all three fail.
    def _pinned(model_id, schema):
        import json as _json, re as _re
        llm = ChatOpenAI(
            base_url="https://openrouter.ai/api/v1", api_key=key,
            model=model_id, temperature=0.0, max_retries=2,
            default_headers={"HTTP-Referer": "https://agensflow.ai", "X-Title": "AgensFlow preview"},
        )
        fc = llm.with_structured_output(schema, method="function_calling")
        js = llm.with_structured_output(schema, method="json_schema")

        async def _raw_parse(msgs, config=None):
            """Third fallback: raw text response, extract first JSON object, construct schema."""
            resp = await llm.ainvoke(msgs, config=config)
            text = getattr(resp, "content", None) or str(resp)
            # Extract the largest {...} block, strip common wrappers
            m = _re.search(r"\{.*\}", text, _re.DOTALL)
            if not m:
                raise ValueError(f"no JSON object in raw response: {text[:200]}")
            payload = _json.loads(m.group(0))
            return schema.model_validate(payload)

        class _FallthroughShim:
            async def ainvoke(self, msgs, config=None):
                try:
                    return await fc.ainvoke(msgs, config=config)
                except Exception as e_fc:
                    try:
                        return await js.ainvoke(msgs, config=config)
                    except Exception:
                        try:
                            return await _raw_parse(msgs, config=config)
                        except Exception:
                            raise e_fc

        return _FallthroughShim()

    def _build_pinned(model_id, *, solver_skill="concise", memory_skip=False, verifier_skip=False):
        """Pinned baseline. If skips or a specific solver_skill are set, use them —
        letting us build a fair "all-best BUT with substrate's topology & skill picks"
        comparison that isolates model-tier as the only difference from the substrate."""
        p, m, s, v, e = (_pinned(model_id, sch) for sch in
                         (PlannerOutput, MemoryOutput, SolverOutput, VerifierOutput, EvaluatorOutput))

        # Node functions accept + forward `config` so LangGraph's outer
        # callbacks (CostCapture) propagate down to each model.ainvoke.
        async def _planner(state, config=None):
            r = await p.ainvoke([("system", PLANNER_SYS), ("human", format_planner_input(state["user_task"]))], config=config)
            return {"goal": r.goal, "subproblem": r.subproblem}
        async def _memory(state, config=None):
            if memory_skip:
                return {"evidence": []}
            r = await m.ainvoke([("system", MEMORY_SYS.format(corpus=_render_corpus_subset(state.get("corpus_doc_ids", [])))),
                                 ("human", format_memory_input(state["subproblem"]))], config=config)
            return {"evidence": [ev.model_dump() for ev in r.evidence]}
        async def _solver(state, config=None):
            ev = [EvidenceItem(**e) for e in state.get("evidence", [])]
            system = SOLVER_SYSTEMS.get(solver_skill, SOLVER_SYSTEMS["concise"])
            r = await s.ainvoke([("system", system),
                                 ("human", format_solver_input(state["subproblem"], ev))], config=config)
            return {"draft_answer": r.draft_answer, "solver_reasoning": r.reasoning}
        async def _verifier(state, config=None):
            if verifier_skip:
                return {"verifier_verdict": "supported", "ungrounded_claims": []}
            ev = [EvidenceItem(**el) for el in state.get("evidence", [])]
            r = await v.ainvoke([("system", VERIFIER_SYS),
                                 ("human", format_verifier_input(state["subproblem"], state["draft_answer"], ev))], config=config)
            return {"verifier_verdict": r.verdict, "ungrounded_claims": list(r.ungrounded_claims)}
        async def _evaluator(state, config=None):
            r = await e.ainvoke([("system", EVALUATOR_SYS),
                                 ("human", format_evaluator_input(state["goal"], state["draft_answer"], state.get("verifier_verdict", "unknown")))], config=config)
            return {"final_answer": r.final_answer, "evaluator_reasoning": r.merged_reasoning}

        g = StateGraph(SecurityMASState)
        for n, fn in [("planner", _planner), ("memory", _memory), ("solver", _solver),
                      ("verifier", _verifier), ("evaluator", _evaluator)]:
            g.add_node(n, fn)
        for a, b in [(START, "planner"), ("planner", "memory"), ("memory", "solver"),
                     ("solver", "verifier"), ("verifier", "evaluator"), ("evaluator", END)]:
            g.add_edge(a, b)
        return g.compile(checkpointer=InMemorySaver())

    representative = next(t for t in ALL_TASKS if t.id == "C7.1")
    print(f"  Task: {representative.id} — {representative.user_task[:120]}...\n")

    pinned_cheap = _build_pinned("thinkingmachines/inkling")
    pinned_best  = _build_pinned("anthropic/claude-sonnet-5")

    # Substrate runs FIRST — we capture its routing to build a fair comparison
    # graph ("all-best-substrate-topology") that uses substrate's skip + skill
    # picks but pins model to sonnet-5. That isolates model-tier savings from
    # skip+skill savings so buyers can attribute the substrate's cost win.
    strategies = [
        ("substrate",    compiled,      None),   # cost via _substrate_cost_from_server
        ("all-cheapest", pinned_cheap,  "thinkingmachines/inkling"),
        ("all-best",     pinned_best,   "anthropic/claude-sonnet-5"),
    ]

    baseline_llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1", api_key=key,
        model="thinkingmachines/inkling", temperature=0.0,
        default_headers={"HTTP-Referer": "https://agensflow.ai", "X-Title": "AgensFlow preview baseline"},
    )
    baseline_answer = (await baseline_llm.ainvoke([
        ("system", "Answer concisely."),
        ("human", representative.user_task),
    ])).content

    # Map (signature-prefix, arm) → OpenRouter model slug. Skip arms cost $0.
    def _arm_to_model(signature: str, arm: str) -> str | None:
        node = signature.split(":", 1)[0] if signature else signature
        if arm == "skip":
            return None
        if node == "planner" or node == "evaluator":
            return "anthropic/claude-sonnet-5"
        if node == "memory":
            return "thinkingmachines/inkling"  # "use"
        if node == "verifier":
            return {"fast": "thinkingmachines/inkling",
                    "haiku": "anthropic/claude-haiku-4.5"}.get(arm)
        if node == "solver":
            # Arm is "{skill}-{tier}" e.g. "concise-haiku"
            tier = arm.split("-", 1)[-1] if "-" in arm else arm
            return {"haiku": "anthropic/claude-haiku-4.5",
                    "fast":  "thinkingmachines/inkling",
                    "mini":  "anthropic/claude-sonnet-5"}.get(tier)
        return None

    async def _substrate_cost_from_server(thread_id: str) -> tuple[float, int, int]:
        """Sum cost + tokens across the 5 most recent decisions on the server.
        Cost computed from tokens × PRICING per arm's bound model — CostCapture
        doesn't compute cost for modern slugs without a pricing table."""
        resp = await server_client.get(
            "/langgraph/decisions?limit=50",
            headers={"Authorization": f"Bearer {api_key}"},
        )
        decisions = resp.json().get("decisions", [])
        recent = decisions[:5]
        cost = 0.0
        t_in = 0
        t_out = 0
        for d in recent:
            sig = d.get("signature", "")
            arm = d.get("action", "")
            ti = int(d.get("tokens_input") or 0)
            to = int(d.get("tokens_output") or 0)
            model = _arm_to_model(sig, arm)
            arm_cost = _approx_cost(model, ti, to) if model else 0.0
            cost += arm_cost
            t_in += ti
            t_out += to
        return cost, t_in, t_out

    rows = []
    substrate_routing: dict[str, str] = {}
    for name, graph, model_id in strategies:
        cap = CostCapture()
        thread_id = f"preview_8_{name}"
        t0 = time.monotonic()
        try:
            r = await graph.ainvoke(
                {"user_task": representative.user_task,
                 "corpus_doc_ids": representative.corpus_doc_ids,
                 "scenario_class": representative.scenario_class,
                 "trace": [], "revision_count": 0},
                config={"configurable": {"thread_id": thread_id}, "callbacks": [cap]},
            )
        except Exception as ex:
            print(f"  {name:<14} ERROR: {type(ex).__name__}: {str(ex)[:120]}")
            rows.append({"strategy": name, "quality": None, "cost_usd": 0.0,
                         "tokens_in": 0, "tokens_out": 0, "latency_s": 0.0,
                         "error": str(ex)[:200]})
            continue
        elapsed = time.monotonic() - t0

        # Capture substrate's per-node routing so we can build the "fair"
        # all-best comparison that reuses substrate's skip + skill choices.
        if name == "substrate":
            substrate_routing = {step["node"]: step["action"] for step in r.get("trace", [])}

        # Substrate: cost from server (its own callbacks eat outer CostCapture)
        # Pinned:    tokens from CostCapture × PRICING table
        if name == "substrate":
            cost, t_in, t_out = await _substrate_cost_from_server(thread_id)
        else:
            t_in, t_out = cap.input_tokens, cap.output_tokens
            cost = _approx_cost(model_id, t_in, t_out) if model_id else cap.cost_usd

        try:
            q, _axes = await relative_quality(
                task=representative.user_task, candidate=r["final_answer"], baseline=baseline_answer,
                openrouter_key=key, models=PANEL_MODELS,
            )
        except Exception as ex:
            print(f"  {name:<14} judge ERROR: {type(ex).__name__}: {str(ex)[:120]}")
            q = 0.0
        rows.append({
            "strategy": name, "quality": q,
            "tokens_in": t_in, "tokens_out": t_out,
            "cost_usd": cost, "latency_s": elapsed,
            "answer_len": len(r["final_answer"]),
        })
        print(f"  {name:<14} q={q:.3f}  tokens=in{t_in}/out{t_out}  "
              f"cost=${cost:.4f}  lat={elapsed:.1f}s")

    # ---- Fair-attribution baseline: all-best model but substrate's topology + skill ----
    if substrate_routing:
        memory_skip = substrate_routing.get("memory") == "skip"
        verifier_skip = substrate_routing.get("verifier") == "skip"
        solver_arm = substrate_routing.get("solver", "concise-mini")
        solver_skill = solver_arm.split("-", 1)[0] if "-" in solver_arm else "concise"
        fair_best = _build_pinned("anthropic/claude-sonnet-5",
                                  solver_skill=solver_skill,
                                  memory_skip=memory_skip,
                                  verifier_skip=verifier_skip)
        print(f"\n  [fair-attribution baseline: sonnet-5 everywhere, "
              f"substrate's skip+skill choices: solver_skill={solver_skill}, "
              f"memory_skip={memory_skip}, verifier_skip={verifier_skip}]")
        cap = CostCapture()
        t0 = time.monotonic()
        try:
            r = await fair_best.ainvoke(
                {"user_task": representative.user_task,
                 "corpus_doc_ids": representative.corpus_doc_ids,
                 "scenario_class": representative.scenario_class,
                 "trace": [], "revision_count": 0},
                config={"configurable": {"thread_id": "preview_8_all-best-substrate-topology"},
                        "callbacks": [cap]},
            )
            elapsed = time.monotonic() - t0
            t_in, t_out = cap.input_tokens, cap.output_tokens
            cost = _approx_cost("anthropic/claude-sonnet-5", t_in, t_out)
            q, _axes = await relative_quality(
                task=representative.user_task, candidate=r["final_answer"], baseline=baseline_answer,
                openrouter_key=key, models=PANEL_MODELS,
            )
            rows.append({
                "strategy": "best-fair", "quality": q,
                "tokens_in": t_in, "tokens_out": t_out, "cost_usd": cost,
                "latency_s": elapsed, "answer_len": len(r["final_answer"]),
            })
            print(f"  {'best-fair':<14} q={q:.3f}  tokens=in{t_in}/out{t_out}  "
                  f"cost=${cost:.4f}  lat={elapsed:.1f}s")
        except Exception as ex:
            print(f"  {'best-fair':<14} ERROR: {type(ex).__name__}: {str(ex)[:120]}")
            rows.append({"strategy": "best-fair", "quality": None, "cost_usd": 0.0,
                         "tokens_in": 0, "tokens_out": 0, "latency_s": 0.0,
                         "error": str(ex)[:200]})

    if rows:
        best_row = next((r for r in rows if r.get("strategy") == "all-best"
                         and r.get("quality") is not None), None)

        print("\n  === Pareto: substrate vs baselines ===")
        print(f"  {'strategy':<14} {'quality':>8} {'cost':>10} {'latency':>9}"
              + (f" {'q%':>6} {'$%':>6}" if best_row else ""))
        for r in rows:
            if r.get("quality") is None:
                print(f"  {r['strategy']:<14} (failed: {r.get('error', 'unknown')[:80]})")
                continue
            line = (f"  {r['strategy']:<14} {r['quality']:>8.3f} "
                    f"${r['cost_usd']:>9.4f} {r['latency_s']:>8.1f}s")
            if best_row:
                q_pct = 100 * r["quality"] / max(best_row["quality"], 1e-6)
                c_pct = 100 * r["cost_usd"] / max(best_row["cost_usd"], 1e-9)
                line += f" {q_pct:>5.1f}% {c_pct:>5.1f}%"
            print(line)

        print("\n  ── What the columns mean ──")
        print("    substrate    : full learned routing (model tier + skill card + skip)")
        print("    all-cheapest : inkling everywhere, always-use, concise skill")
        print("    all-best     : sonnet-5 everywhere, always-use, concise skill")
        print("    best-fair    : sonnet-5 everywhere, but substrate's skip + skill choices")
        print("                   (isolates model-tier savings from skip+skill savings)")

        # Attribution breakdown — clear signed deltas
        sub = next((r for r in rows if r["strategy"] == "substrate"
                    and r.get("quality") is not None), None)
        fair = next((r for r in rows if r["strategy"] == "best-fair"
                     and r.get("quality") is not None), None)
        cheap = next((r for r in rows if r["strategy"] == "all-cheapest"
                      and r.get("quality") is not None), None)
        print("\n  ── Attribution ──")
        if sub and best_row:
            saved_c = 1 - sub["cost_usd"] / max(best_row["cost_usd"], 1e-9)
            q_delta = (sub["quality"] - best_row["quality"]) / max(best_row["quality"], 1e-6)
            q_verb = "gains" if q_delta >= 0 else "trades off"
            print(f"    substrate vs all-best (naive): saves {saved_c*100:.0f}% cost, "
                  f"{q_verb} {abs(q_delta)*100:.0f}% quality")
        if best_row and fair:
            skip_saved = 1 - fair["cost_usd"] / max(best_row["cost_usd"], 1e-9)
            print(f"    ├─ from skip+skill picks alone: saves {skip_saved*100:.0f}% cost "
                  f"(best-fair vs all-best)")
        if sub and fair:
            tier_saved = 1 - sub["cost_usd"] / max(fair["cost_usd"], 1e-9)
            print(f"    └─ from model-tier picks alone: saves {tier_saved*100:.0f}% cost "
                  f"(substrate vs best-fair)")
        if sub and cheap:
            q_gain = (sub["quality"] - cheap["quality"]) / max(cheap["quality"], 1e-6)
            print(f"    substrate vs all-cheapest: {q_gain*100:+.0f}% quality "
                  f"(spends ${sub['cost_usd'] - cheap['cost_usd']:.4f} more)")
        print("\n  ── Caveat ──")
        print("    Single task, single trial. Substrate performance vs baselines")
        print("    depends heavily on the task shape + judge stability. This demo is")
        print("    an existence proof of the routing story, not a benchmark result.")


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starter", type=str, default="security_v1_adapted.json",
                    help="Starter policy filename inside examples/starter_policies/")
    ap.add_argument("--only", type=str, default="6,7,8",
                    help="Comma-sep list of section numbers to run (6, 7, 8). Default: all.")
    args = ap.parse_args()

    try:
        from dotenv import load_dotenv
        env_path = REPO_ROOT / ".env"
        if env_path.exists():
            load_dotenv(env_path)
    except ImportError:
        pass
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        print("ERROR: OPENROUTER_API_KEY not set", file=sys.stderr)
        sys.exit(1)

    starter_path = REPO_ROOT / "examples" / "starter_policies" / args.starter
    if not starter_path.exists():
        print(f"ERROR: starter not found: {starter_path}", file=sys.stderr)
        sys.exit(2)

    sections = set(args.only.split(","))

    print(f"  starter:  {starter_path.name}")
    print(f"  sections: {sorted(sections)}")

    app, server_client, lifespan_mgr = await _boot_server()

    from agensflow_langgraph import aimport_policy
    imp = await aimport_policy(starter_path)
    print(f"  ✓ imported: {imp['signatures_merged']} sigs / {imp['actions_merged']} arms")

    from examples.security_domain.tasks import ALL_TASKS
    from examples.security_domain.graph import build_pools, build_graph
    pools = build_pools()
    compiled = build_graph(pools)

    if "6" in sections:
        await _section_6()
    if "7" in sections:
        await _section_7(compiled, ALL_TASKS)
    if "8" in sections:
        await _section_8(compiled, ALL_TASKS, key, os.environ["AGENSFLOW_API_KEY"], server_client)

    await server_client.aclose()
    await lifespan_mgr.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(main())
