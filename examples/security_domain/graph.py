"""security_domain MAS — same parallel_critic topology, but the MEMORY node
retrieves from a per-task subset of the security-advisory corpus.

Reuses the schemas + prompts from `examples.parallel_critic_mas.prompts`
verbatim — only the memory node's corpus threading differs (per-task
subset instead of one global corpus render).

State adds one field vs parallel_critic_mas:
  * `corpus_doc_ids: list[str]`  — which docs to expose to memory for
    this task (set from SecurityTask.corpus_doc_ids at invocation).
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

from ..parallel_critic_mas.prompts import (
    CRITIC_SYS,
    CriticOutput,
    EVALUATOR_SYS,
    EvaluatorOutput,
    EvidenceItem,
    MEMORY_SYS,
    MemoryOutput,
    PLANNER_SYS,
    PlannerOutput,
    SOLVER_SYS,
    SolverOutput,
    VERIFIER_SYS,
    VerifierOutput,
    format_critic_input,
    format_evaluator_input,
    format_memory_input,
    format_planner_input,
    format_solver_input,
    format_verifier_input,
)
from .corpus import CORPUS, get_corpus_subset


class SecurityMASState(TypedDict, total=False):
    user_task: str
    corpus_doc_ids: list[str]     # NEW: per-task doc subset

    goal: str
    subproblem: str
    evidence: list[dict]
    draft_answer: str
    solver_reasoning: str
    reasoning_score: float
    reasoning_issues: list[str]
    verifier_verdict: str
    ungrounded_claims: list[str]
    final_answer: str
    evaluator_reasoning: str

    messages: Annotated[list, add_messages]
    trace: Annotated[list, operator.add]


def _or(model_id: str) -> ChatOpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError("OPENROUTER_API_KEY not set — required for security_domain.")
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        model=model_id,
        temperature=0.0,
        max_retries=2,
    )


def build_pools() -> dict[str, dict[str, Any]]:
    def wrap(model_id: str, schema):
        return _or(model_id).with_structured_output(schema, method="function_calling")

    # Same shape as parallel_critic_mas — this is intentional so the
    # converged security_v1.json is portable to the notebook's identical
    # topology.
    return {
        "planner": {
            "fast": wrap("openai/gpt-4o-mini", PlannerOutput),
            "deep": wrap("anthropic/claude-sonnet-4", PlannerOutput),
        },
        "memory": {
            "cheap": wrap("meta-llama/llama-3.3-70b-instruct", MemoryOutput),
            "solid": wrap("openai/gpt-4o-mini", MemoryOutput),
        },
        "solver": {
            "cheap":    wrap("meta-llama/llama-3.3-70b-instruct", SolverOutput),
            "balanced": wrap("openai/gpt-4o", SolverOutput),
            "deep":     wrap("anthropic/claude-sonnet-4", SolverOutput),
        },
        "critic": {
            "cheap": wrap("openai/gpt-4o-mini", CriticOutput),
            "solid": wrap("openai/gpt-4o", CriticOutput),
        },
        "verifier": {
            "cheap": wrap("openai/gpt-4o-mini", VerifierOutput),
            "solid": wrap("openai/gpt-4o", VerifierOutput),
        },
        "evaluator": {
            "default": wrap("openai/gpt-4o-mini", EvaluatorOutput),
        },
    }


def _render_corpus_subset(doc_ids: list[str]) -> str:
    docs = get_corpus_subset(doc_ids) if doc_ids else CORPUS
    return "\n\n".join(f"[{d.id}]\n{d.text}" for d in docs)


def build_graph(pools: dict[str, dict[str, Any]]):
    def _trace(node: str, model: Any) -> dict:
        cfg = getattr(model, "config", None) or {}
        meta = cfg.get("metadata") or {}
        return {"node": node, "action": meta.get("agensflow_action", "<fell-open>")}

    @agensflow(pool=pools["planner"])
    async def planner(state: SecurityMASState, model, config=None):
        r: PlannerOutput = await model.ainvoke([
            ("system", PLANNER_SYS),
            ("human", format_planner_input(state["user_task"])),
        ])
        return {"goal": r.goal, "subproblem": r.subproblem, "trace": [_trace("planner", model)]}

    @agensflow(pool=pools["memory"])
    async def memory(state: SecurityMASState, model, config=None):
        corpus_str = _render_corpus_subset(state.get("corpus_doc_ids", []))
        r: MemoryOutput = await model.ainvoke([
            ("system", MEMORY_SYS.format(corpus=corpus_str)),
            ("human", format_memory_input(state["subproblem"])),
        ])
        return {"evidence": [e.model_dump() for e in r.evidence],
                "trace": [_trace("memory", model)]}

    @agensflow(pool=pools["solver"])
    async def solver(state: SecurityMASState, model, config=None):
        ev = [EvidenceItem(**e) for e in state.get("evidence", [])]
        r: SolverOutput = await model.ainvoke([
            ("system", SOLVER_SYS),
            ("human", format_solver_input(state["subproblem"], ev)),
        ])
        return {"draft_answer": r.draft_answer,
                "solver_reasoning": r.reasoning,
                "trace": [_trace("solver", model)]}

    @agensflow(pool=pools["critic"])
    async def critic(state: SecurityMASState, model, config=None):
        r: CriticOutput = await model.ainvoke([
            ("system", CRITIC_SYS),
            ("human", format_critic_input(state["subproblem"],
                                          state["draft_answer"],
                                          state["solver_reasoning"])),
        ])
        return {"reasoning_score": r.reasoning_score,
                "reasoning_issues": list(r.reasoning_issues),
                "trace": [_trace("critic", model)]}

    @agensflow(pool=pools["verifier"])
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

    @agensflow(pool=pools["evaluator"])
    async def evaluator(state: SecurityMASState, model, config=None):
        r: EvaluatorOutput = await model.ainvoke([
            ("system", EVALUATOR_SYS),
            ("human", format_evaluator_input(state["goal"], state["draft_answer"],
                                             state.get("reasoning_score", 0.0),
                                             state.get("verifier_verdict", "unknown"))),
        ])
        return {"final_answer": r.final_answer,
                "evaluator_reasoning": r.merged_reasoning,
                "trace": [_trace("evaluator", model)]}

    graph = StateGraph(SecurityMASState)
    graph.add_node("planner",   planner)
    graph.add_node("memory",    memory)
    graph.add_node("solver",    solver)
    graph.add_node("critic",    critic)
    graph.add_node("verifier",  verifier)
    graph.add_node("evaluator", evaluator)

    graph.add_edge(START,       "planner")
    graph.add_edge("planner",   "memory")
    graph.add_edge("memory",    "solver")
    graph.add_edge("solver",    "critic")
    graph.add_edge("solver",    "verifier")
    graph.add_edge("critic",    "evaluator")
    graph.add_edge("verifier",  "evaluator")
    graph.add_edge("evaluator", END)

    return graph.compile(checkpointer=InMemorySaver())
