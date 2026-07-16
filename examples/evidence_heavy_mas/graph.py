"""The evidence_heavy MAS as a real LangGraph — 5 decorated nodes, conditional
verifier gate, revision loop, InMemorySaver checkpointer.

Every model call goes through OpenRouter via `ChatOpenAI(base_url=...)`. Each
node is decorated with `@agensflow(pool={...})` — the substrate learns which
pool member to route each node's calls to as the demo runs across the task set.

Pool composition per node:
  planner   : {"fast": gpt-4o-mini, "deep": claude-sonnet-4}
  memory    : {"cheap": qwen-2.5-72b, "solid": gpt-4o-mini}
  solver    : {"cheap": qwen-2.5-72b, "balanced": gpt-4o, "deep": claude-sonnet-4}
  verifier  : {"cheap": gpt-4o-mini, "solid": gpt-4o}
  evaluator : {"default": gpt-4o-mini}    (1-arm pool — the demo isn't trying
                                            to route this stage)

Signature per node is auto-derived from `langgraph_node`. The substrate learns
per-node routing over ~20 total task runs (5 tasks × 4 runs).
"""

from __future__ import annotations

import os
from typing import Annotated, Any, TypedDict

from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agensflow_langgraph import agensflow

from .documents import render_corpus
from .prompts import (
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
    format_evaluator_input,
    format_memory_input,
    format_planner_input,
    format_solver_input,
    format_verifier_input,
)

MAX_REVISIONS = 2  # verifier can send back to solver at most this many times


# --------------------------------------------------------------------------- #
# Graph state
# --------------------------------------------------------------------------- #


class MASState(TypedDict, total=False):
    # Inputs
    user_task: str

    # Handoffs
    goal: str
    subproblem: str
    constraints: list[str]
    evidence: list[dict]  # serialized EvidenceItem
    draft_answer: str
    cited_evidence_ids: list[str]
    verifier_verdict: str
    verifier_critique: str
    revision_count: int

    # Output
    final_answer: str
    evaluator_reasoning: str

    # Diagnostics — accumulated by nodes
    messages: Annotated[list, add_messages]
    trace: list[dict]  # {"node": str, "action": str} per node execution


# --------------------------------------------------------------------------- #
# Model pool
# --------------------------------------------------------------------------- #


def _or(model_id: str) -> ChatOpenAI:
    """Return a ChatOpenAI configured to hit OpenRouter for the given model.

    Requires `OPENROUTER_API_KEY` in env. One env var unlocks Anthropic / OpenAI /
    Qwen / Grok / Gemini via OpenRouter's OpenAI-compatible endpoint."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise RuntimeError(
            "OPENROUTER_API_KEY not set. This example makes real LLM calls via "
            "OpenRouter — set the env var before running."
        )
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=key,
        model=model_id,
        temperature=0.0,
        max_retries=2,
    )


def build_pools() -> dict[str, dict[str, Any]]:
    """Construct the per-node pools. Kept in a function so imports don't fail
    when OPENROUTER_API_KEY isn't set (needed for pytest discovery).

    Uses `method="function_calling"` for structured output — some models (e.g.
    Anthropic via OpenRouter) return JSON wrapped in ```markdown fences``` when
    left to freely emit; the function-calling path forces a clean schema."""
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
        "verifier": {
            "cheap": wrap("openai/gpt-4o-mini", VerifierOutput),
            "solid": wrap("openai/gpt-4o", VerifierOutput),
        },
        "evaluator": {
            "default": wrap("openai/gpt-4o-mini", EvaluatorOutput),
        },
    }


# --------------------------------------------------------------------------- #
# Nodes — decorated at build time so pools can be closure-scoped
# --------------------------------------------------------------------------- #


def build_graph(pools: dict[str, dict[str, Any]]):
    """Compile the MAS graph with the given per-node pools.

    Returns the compiled graph (with InMemorySaver checkpointer). The example
    invokes it once per task via `graph.ainvoke(state, config)`."""

    def _trace_entry(node: str, model: Any) -> dict:
        """Pull the substrate's chosen action off the bound model's config.

        The decorator does `pool[action].with_config(metadata={agensflow_action:...})`
        before injecting `model` into the node function; the RunnableBinding exposes
        this via `.config["metadata"]`, so we can surface it for display / logging.
        """
        cfg = getattr(model, "config", None) or {}
        meta = cfg.get("metadata") or {}
        return {"node": node, "action": meta.get("agensflow_action", "<fell-open>")}

    @agensflow(pool=pools["planner"])
    async def planner(state: MASState, model, config=None) -> dict:
        result: PlannerOutput = await model.ainvoke(
            [
                ("system", PLANNER_SYS),
                ("human", format_planner_input(state["user_task"])),
            ]
        )
        return {
            "goal": result.goal,
            "subproblem": result.subproblem,
            "constraints": list(result.constraints),
            "trace": state.get("trace", []) + [_trace_entry("planner", model)],
        }

    @agensflow(pool=pools["memory"])
    async def memory(state: MASState, model, config=None) -> dict:
        # Bake the corpus into the system prompt — real production would use a
        # vector store; this example keeps it simple + deterministic.
        result: MemoryOutput = await model.ainvoke(
            [
                ("system", MEMORY_SYS.format(corpus=render_corpus())),
                ("human", format_memory_input(state["subproblem"])),
            ]
        )
        return {
            "evidence": [e.model_dump() for e in result.evidence],
            "trace": state.get("trace", []) + [_trace_entry("memory", model)],
        }

    @agensflow(pool=pools["solver"])
    async def solver(state: MASState, model, config=None) -> dict:
        evidence = [EvidenceItem(**e) for e in state.get("evidence", [])]
        critique = state.get("verifier_critique") or None
        result: SolverOutput = await model.ainvoke(
            [
                ("system", SOLVER_SYS),
                (
                    "human",
                    format_solver_input(
                        state["subproblem"],
                        state.get("constraints", []),
                        evidence,
                        critique=critique,
                    ),
                ),
            ]
        )
        return {
            "draft_answer": result.draft_answer,
            "cited_evidence_ids": list(result.cited_evidence_ids),
            "trace": state.get("trace", []) + [_trace_entry("solver", model)],
        }

    @agensflow(pool=pools["verifier"])
    async def verifier(state: MASState, model, config=None) -> dict:
        evidence = [EvidenceItem(**e) for e in state.get("evidence", [])]
        result: VerifierOutput = await model.ainvoke(
            [
                ("system", VERIFIER_SYS),
                (
                    "human",
                    format_verifier_input(
                        state["subproblem"], state["draft_answer"], evidence
                    ),
                ),
            ]
        )
        return {
            "verifier_verdict": result.verdict,
            "verifier_critique": result.critique,
            "trace": state.get("trace", []) + [_trace_entry("verifier", model)],
        }

    @agensflow(pool=pools["evaluator"])
    async def evaluator(state: MASState, model, config=None) -> dict:
        result: EvaluatorOutput = await model.ainvoke(
            [
                ("system", EVALUATOR_SYS),
                (
                    "human",
                    format_evaluator_input(
                        state["goal"],
                        state["draft_answer"],
                        state.get("verifier_verdict", "unknown"),
                    ),
                ),
            ]
        )
        return {
            "final_answer": result.final_answer,
            "evaluator_reasoning": result.reasoning,
            "trace": state.get("trace", []) + [_trace_entry("evaluator", model)],
        }

    # ---- Conditional routing after verifier -----------------------------
    def verifier_gate(state: MASState) -> str:
        """Route based on verifier verdict + revision cap. Exercises
        add_conditional_edges — the load-bearing LangGraph feature."""
        verdict = state.get("verifier_verdict", "unsupported")
        revisions = state.get("revision_count", 0)
        if verdict in ("supported", "partial"):
            return "evaluator"
        if revisions >= MAX_REVISIONS:
            return "evaluator"  # Give up — go with the current draft
        return "solver"  # Send back for revision

    async def bump_revision(state: MASState, config=None) -> dict:
        """Small helper node in the revision path — increments the counter so
        the gate can enforce MAX_REVISIONS. This node isn't decorated (no LLM
        call), it just mutates state before looping back to solver."""
        return {"revision_count": state.get("revision_count", 0) + 1}

    # ---- Compile --------------------------------------------------------
    graph = StateGraph(MASState)
    graph.add_node("planner", planner)
    graph.add_node("memory", memory)
    graph.add_node("solver", solver)
    graph.add_node("verifier", verifier)
    graph.add_node("evaluator", evaluator)
    graph.add_node("bump_revision", bump_revision)

    graph.add_edge(START, "planner")
    graph.add_edge("planner", "memory")
    graph.add_edge("memory", "solver")
    graph.add_edge("solver", "verifier")
    graph.add_conditional_edges(
        "verifier",
        verifier_gate,
        {"evaluator": "evaluator", "solver": "bump_revision"},
    )
    graph.add_edge("bump_revision", "solver")
    graph.add_edge("evaluator", END)

    return graph.compile(checkpointer=InMemorySaver())
