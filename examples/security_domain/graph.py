"""security_domain MAS — linear topology with skill × model factored solver.

Faithful to the paper's e09 action space:
  * planner  → 1 arm
  * memory   → 2 arms: use, skip
  * solver   → 9 arms: 3 skill cards × 3 model tiers
  * verifier → 3 arms: fast, haiku, skip
  * evaluator → 1 arm

Model tier binding (arm-key suffix → OpenRouter model):
  * haiku → anthropic/claude-haiku-4.5
  * fast  → thinkingmachines/inkling
  * mini  → anthropic/claude-sonnet-5   (paper's "mini" tier upgraded to
                                         a modern Anthropic model — keeps
                                         Anthropic + ThinkingMachines as the
                                         only two task-pool families, so the
                                         judge panel's xAI/OpenAI/Qwen is
                                         fully family-disjoint)

Topology:
  START → planner → memory-or-skip → solver → verifier-or-skip
                                                   ↓
                                        gate: supported/partial → evaluator → END
                                              unsupported (≤2 revs) → solver
"""

from __future__ import annotations

import operator
import os
from typing import Annotated, Any, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agensflow_langgraph import agensflow

from .corpus import CORPUS, get_corpus_subset
from .prompts import (
    EVALUATOR_SYS,
    EvaluatorOutput,
    EvidenceItem,
    MEMORY_SYS,
    MemoryOutput,
    PLANNER_SYS,
    PlannerOutput,
    SOLVER_SYSTEMS,
    SolverOutput,
    VERIFIER_SYS,
    VerifierOutput,
    format_evaluator_input,
    format_memory_input,
    format_planner_input,
    format_solver_input,
    format_verifier_input,
)


MODEL_BINDING = {
    "haiku": "anthropic/claude-haiku-4.5",
    "fast":  "thinkingmachines/inkling",
    "mini":  "anthropic/claude-sonnet-5",
}

SKILL_CARDS = ("concise", "cot", "evidence")

# Scenario-class → regime mapping (matches the paper's e09 encoding).
# The substrate keys per-(node, regime) so C1 tasks and C7 tasks route to
# different arms even though they hit the same nodes.
SCENARIO_TO_REGIME = {
    "C1": "straightforward", "C6": "straightforward", "C8": "straightforward",
    "C2": "evidence_heavy",   "C3": "evidence_heavy",   "C4": "evidence_heavy",  "C7": "evidence_heavy",
    "C5": "ambiguous",
}


def regime_signature(state, config, node_name) -> str:
    """Signature callable: returns f"{node}:{regime}".

    Regime comes from `state["regime"]` if set explicitly, otherwise from
    `state["scenario_class"]` via SCENARIO_TO_REGIME, otherwise defaults to
    `evidence_heavy` (paper's majority class). Enables per-regime routing —
    the substrate tracks separate arm stats for `solver:evidence_heavy` vs
    `solver:straightforward` vs `solver:ambiguous`.
    """
    regime = state.get("regime")
    if regime is None:
        sc = state.get("scenario_class")
        regime = SCENARIO_TO_REGIME.get(sc, "evidence_heavy")
    return f"{node_name}:{regime}"


class SecurityMASState(TypedDict, total=False):
    user_task: str
    corpus_doc_ids: list[str]
    scenario_class: str      # C1..C8 — feeds regime_signature
    regime: str              # explicit override; overrides scenario_class mapping
    goal: str
    subproblem: str
    evidence: list[dict]
    draft_answer: str
    solver_reasoning: str
    verifier_verdict: str
    ungrounded_claims: list[str]
    revision_count: int
    final_answer: str
    evaluator_reasoning: str
    messages: Annotated[list, add_messages]
    trace: Annotated[list, operator.add]


def _or(model_id: str, schema=None):
    """OpenRouter-backed ChatOpenAI.

    The `default_headers` are load-bearing: OpenRouter uses HTTP-Referer +
    X-Title for app-tier attribution and routing. Without them, requests are
    served from a shared-throughput pool with materially tighter rate limits.
    """
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set — required for security_domain.")
    llm = ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        model=model_id,
        temperature=0.0,
        max_retries=2,
        default_headers={
            "HTTP-Referer": "https://agensflow.ai",
            "X-Title": "AgensFlow security_domain",
        },
    )
    return llm.with_structured_output(schema, method="function_calling") if schema else llm


def _skip_runnable(default_output: Any):
    """A no-op Runnable that returns `default_output` regardless of input.

    Used as the arm binding for 'skip' choices — when the substrate picks
    skip, the node produces this default and moves on with no LLM call.
    """
    from langchain_core.runnables import RunnableLambda
    return RunnableLambda(lambda _: default_output)


def build_pools() -> dict[str, dict[str, Any]]:
    """Return per-node pools matching the paper's e09 action space.

    Solver keys use `{skill}-{model_tier}` format so the paper-pkl translator
    can map paper's `solver_{skill}_{tier}` → OSS `{skill}-{tier}`.
    """
    planner_pool = {
        "default": _or(MODEL_BINDING["mini"], PlannerOutput),
    }

    memory_pool = {
        "use":  _or(MODEL_BINDING["fast"], MemoryOutput),
        "skip": _skip_runnable(MemoryOutput(evidence=[])),
    }

    solver_pool: dict[str, Any] = {}
    for skill in SKILL_CARDS:
        for tier in ("haiku", "fast", "mini"):
            key = f"{skill}-{tier}"
            solver_pool[key] = _or(MODEL_BINDING[tier], SolverOutput)

    verifier_pool = {
        "fast":  _or(MODEL_BINDING["fast"], VerifierOutput),
        "haiku": _or(MODEL_BINDING["haiku"], VerifierOutput),
        "skip":  _skip_runnable(VerifierOutput(verdict="supported", ungrounded_claims=[])),
    }

    evaluator_pool = {
        "default": _or(MODEL_BINDING["mini"], EvaluatorOutput),
    }

    return {
        "planner":   planner_pool,
        "memory":    memory_pool,
        "solver":    solver_pool,
        "verifier":  verifier_pool,
        "evaluator": evaluator_pool,
    }


def _render_corpus_subset(doc_ids: list[str]) -> str:
    docs = get_corpus_subset(doc_ids) if doc_ids else CORPUS
    return "\n\n".join(f"[{d.id}]\n{d.text}" for d in docs)


def build_graph(pools: dict[str, dict[str, Any]], *, max_revisions: int = 2):
    def _trace(node: str, model: Any) -> dict:
        cfg = getattr(model, "config", None) or {}
        meta = cfg.get("metadata") or {}
        return {"node": node, "action": meta.get("agensflow_action", "<fell-open>")}

    @agensflow(pool=pools["planner"], signature=regime_signature)
    async def planner(state: SecurityMASState, model, config=None):
        r: PlannerOutput = await model.ainvoke([
            ("system", PLANNER_SYS),
            ("human", format_planner_input(state["user_task"])),
        ])
        return {"goal": r.goal, "subproblem": r.subproblem,
                "trace": [_trace("planner", model)]}

    @agensflow(pool=pools["memory"], signature=regime_signature)
    async def memory(state: SecurityMASState, model, config=None):
        payload_msgs = [
            ("system", MEMORY_SYS.format(corpus=_render_corpus_subset(
                state.get("corpus_doc_ids", [])))),
            ("human", format_memory_input(state["subproblem"])),
        ]
        r = await model.ainvoke(payload_msgs)
        return {"evidence": [e.model_dump() for e in r.evidence],
                "trace": [_trace("memory", model)]}

    @agensflow(pool=pools["solver"], signature=regime_signature)
    async def solver(state: SecurityMASState, model, config=None):
        cfg = getattr(model, "config", None) or {}
        action = (cfg.get("metadata") or {}).get("agensflow_action", "concise-haiku")
        skill = action.split("-", 1)[0] if "-" in action else "concise"
        system = SOLVER_SYSTEMS.get(skill, SOLVER_SYSTEMS["concise"])

        ev = [EvidenceItem(**e) for e in state.get("evidence", [])]
        r: SolverOutput = await model.ainvoke([
            ("system", system),
            ("human", format_solver_input(state["subproblem"], ev)),
        ])
        return {"draft_answer": r.draft_answer,
                "solver_reasoning": r.reasoning,
                "trace": [_trace("solver", model)]}

    @agensflow(pool=pools["verifier"], signature=regime_signature)
    async def verifier(state: SecurityMASState, model, config=None):
        ev = [EvidenceItem(**e) for e in state.get("evidence", [])]
        r: VerifierOutput = await model.ainvoke([
            ("system", VERIFIER_SYS),
            ("human", format_verifier_input(state["subproblem"],
                                            state["draft_answer"], ev)),
        ])
        return {"verifier_verdict": r.verdict,
                "ungrounded_claims": list(r.ungrounded_claims),
                "trace": [_trace("verifier", model)]}

    def verifier_gate(state: SecurityMASState) -> str:
        verdict = state.get("verifier_verdict", "supported")
        revs = state.get("revision_count", 0)
        if verdict == "unsupported" and revs < max_revisions:
            return "bump_revision"
        return "evaluator"

    async def bump_revision(state: SecurityMASState):
        return {"revision_count": state.get("revision_count", 0) + 1}

    @agensflow(pool=pools["evaluator"], signature=regime_signature)
    async def evaluator(state: SecurityMASState, model, config=None):
        r: EvaluatorOutput = await model.ainvoke([
            ("system", EVALUATOR_SYS),
            ("human", format_evaluator_input(
                state["goal"], state["draft_answer"],
                state.get("verifier_verdict", "unknown"))),
        ])
        return {"final_answer": r.final_answer,
                "evaluator_reasoning": r.merged_reasoning,
                "trace": [_trace("evaluator", model)]}

    graph = StateGraph(SecurityMASState)
    graph.add_node("planner",       planner)
    graph.add_node("memory",        memory)
    graph.add_node("solver",        solver)
    graph.add_node("verifier",      verifier)
    graph.add_node("bump_revision", bump_revision)
    graph.add_node("evaluator",     evaluator)

    graph.add_edge(START,           "planner")
    graph.add_edge("planner",       "memory")
    graph.add_edge("memory",        "solver")
    graph.add_edge("solver",        "verifier")
    graph.add_conditional_edges(
        "verifier", verifier_gate,
        {"bump_revision": "bump_revision", "evaluator": "evaluator"},
    )
    graph.add_edge("bump_revision", "solver")
    graph.add_edge("evaluator", END)

    return graph.compile(checkpointer=InMemorySaver())
