"""parallel_critic MAS — 5 decorated nodes with critic + verifier IN PARALLEL.

Topology differs from evidence_heavy_mas:

     START
       │
       ▼
   ┌───────┐
   │planner│
   └───┬───┘
       ▼
   ┌───────┐
   │memory │
   └───┬───┘
       ▼
   ┌───────┐
   │solver │
   └───┬───┘
       │
       ├──────────┬───────────┐   ← fan-out (LangGraph runs both concurrently)
       ▼          ▼
   ┌───────┐  ┌───────┐
   │critic │  │verify │
   └───┬───┘  └───┬───┘
       │          │
       └────┬─────┘                ← fan-in (evaluator sees both signals)
            ▼
       ┌─────────┐
       │evaluator│
       └────┬────┘
            ▼
           END

Exercises LangGraph's parallel execution + fan-in — a capability
evidence_heavy_mas doesn't touch.
"""

from __future__ import annotations

import os
from typing import Annotated, Any, TypedDict

from langchain_openai import ChatOpenAI
import operator

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agensflow_langgraph import agensflow

from .prompts import (
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


class MASState(TypedDict, total=False):
    user_task: str
    goal: str
    subproblem: str
    evidence: list[dict]
    draft_answer: str
    solver_reasoning: str

    # From parallel critic + verifier — LangGraph's TypedDict handles concurrent
    # writes to DIFFERENT keys automatically. Critic writes reasoning_score;
    # verifier writes verifier_verdict. Neither writes the other's keys, so
    # there's no reducer conflict.
    reasoning_score: float
    reasoning_issues: list[str]
    verifier_verdict: str
    ungrounded_claims: list[str]

    final_answer: str
    evaluator_reasoning: str

    messages: Annotated[list, add_messages]
    # trace uses list-concat reducer because critic + verifier run in parallel;
    # both append their entries in the same graph step and LangGraph needs a
    # merge strategy for concurrent writes to the same key.
    trace: Annotated[list, operator.add]


def _or(model_id: str) -> ChatOpenAI:
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set — required for parallel_critic MAS."
        )
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


def build_graph(pools: dict[str, dict[str, Any]]):
    def _trace_entry(node: str, model: Any) -> dict:
        cfg = getattr(model, "config", None) or {}
        meta = cfg.get("metadata") or {}
        return {"node": node, "action": meta.get("agensflow_action", "<fell-open>")}

    @agensflow(pool=pools["planner"])
    async def planner(state: MASState, model, config=None) -> dict:
        r: PlannerOutput = await model.ainvoke(
            [
                ("system", PLANNER_SYS),
                ("human", format_planner_input(state["user_task"])),
            ]
        )
        return {
            "goal": r.goal,
            "subproblem": r.subproblem,
            "trace": [_trace_entry("planner", model)],
        }

    @agensflow(pool=pools["memory"])
    async def memory(state: MASState, model, config=None) -> dict:
        from .documents import render_corpus

        r: MemoryOutput = await model.ainvoke(
            [
                ("system", MEMORY_SYS.format(corpus=render_corpus())),
                ("human", format_memory_input(state["subproblem"])),
            ]
        )
        return {
            "evidence": [e.model_dump() for e in r.evidence],
            "trace": [_trace_entry("memory", model)],
        }

    @agensflow(pool=pools["solver"])
    async def solver(state: MASState, model, config=None) -> dict:
        ev = [EvidenceItem(**e) for e in state.get("evidence", [])]
        r: SolverOutput = await model.ainvoke(
            [
                ("system", SOLVER_SYS),
                ("human", format_solver_input(state["subproblem"], ev)),
            ]
        )
        return {
            "draft_answer": r.draft_answer,
            "solver_reasoning": r.reasoning,
            "trace": [_trace_entry("solver", model)],
        }

    # -- Parallel branch: critic and verifier both consume solver's output ----
    # LangGraph runs them concurrently because both are downstream of solver
    # AND both feed into evaluator (fan-in via the edges below).

    @agensflow(pool=pools["critic"])
    async def critic(state: MASState, model, config=None) -> dict:
        r: CriticOutput = await model.ainvoke(
            [
                ("system", CRITIC_SYS),
                (
                    "human",
                    format_critic_input(
                        state["subproblem"],
                        state["draft_answer"],
                        state["solver_reasoning"],
                    ),
                ),
            ]
        )
        return {
            "reasoning_score": r.reasoning_score,
            "reasoning_issues": list(r.reasoning_issues),
            "trace": [_trace_entry("critic", model)],
        }

    @agensflow(pool=pools["verifier"])
    async def verifier(state: MASState, model, config=None) -> dict:
        ev = [EvidenceItem(**e) for e in state.get("evidence", [])]
        r: VerifierOutput = await model.ainvoke(
            [
                ("system", VERIFIER_SYS),
                (
                    "human",
                    format_verifier_input(
                        state["subproblem"], state["draft_answer"], ev
                    ),
                ),
            ]
        )
        return {
            "verifier_verdict": r.verdict,
            "ungrounded_claims": list(r.ungrounded_claims),
            "trace": [_trace_entry("verifier", model)],
        }

    # -- Fan-in: evaluator sees BOTH critic + verifier outputs ---------------

    @agensflow(pool=pools["evaluator"])
    async def evaluator(state: MASState, model, config=None) -> dict:
        r: EvaluatorOutput = await model.ainvoke(
            [
                ("system", EVALUATOR_SYS),
                (
                    "human",
                    format_evaluator_input(
                        state["goal"],
                        state["draft_answer"],
                        state.get("reasoning_score", 0.0),
                        state.get("verifier_verdict", "unknown"),
                    ),
                ),
            ]
        )
        return {
            "final_answer": r.final_answer,
            "evaluator_reasoning": r.merged_reasoning,
            "trace": [_trace_entry("evaluator", model)],
        }

    # Wire the graph
    graph = StateGraph(MASState)
    graph.add_node("planner", planner)
    graph.add_node("memory", memory)
    graph.add_node("solver", solver)
    graph.add_node("critic", critic)
    graph.add_node("verifier", verifier)
    graph.add_node("evaluator", evaluator)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "memory")
    graph.add_edge("memory", "solver")
    # Fan-out: solver → critic AND verifier (LangGraph runs both concurrently)
    graph.add_edge("solver", "critic")
    graph.add_edge("solver", "verifier")
    # Fan-in: BOTH must complete before evaluator runs
    graph.add_edge("critic", "evaluator")
    graph.add_edge("verifier", "evaluator")
    graph.add_edge("evaluator", END)

    return graph.compile(checkpointer=InMemorySaver())
