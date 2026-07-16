"""Per-agent system prompts + Pydantic response schemas.

Each agent gets a strict Pydantic output class so the graph state stays
well-typed. Response schemas are bound via `.with_structured_output(cls)` on
the pool models before decoration — the decorator wraps the resulting
Runnable, not the raw ChatModel.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Response schemas
# --------------------------------------------------------------------------- #


class PlannerOutput(BaseModel):
    """The planner decomposes the user's question into a canonical form."""

    goal: str = Field(description="one-sentence restatement of the user's goal")
    subproblem: str = Field(
        description="the specific factual question the solver must answer"
    )
    constraints: list[str] = Field(
        default_factory=list,
        description="hard constraints on the answer (e.g. length, format, evidence use)",
    )


class EvidenceItem(BaseModel):
    doc_id: str
    snippet: str


class MemoryOutput(BaseModel):
    """The memory node returns evidence snippets relevant to the subproblem."""

    evidence: list[EvidenceItem] = Field(
        description="verbatim snippets from the provided documents (max 5)",
        max_length=5,
    )
    retrieval_notes: str = Field(default="", description="brief note on relevance")


class SolverOutput(BaseModel):
    """The solver drafts an answer grounded in evidence."""

    draft_answer: str = Field(description="the drafted answer, 1-4 sentences")
    cited_evidence_ids: list[str] = Field(
        default_factory=list,
        description="doc_ids of evidence items used",
    )


class VerifierOutput(BaseModel):
    """The verifier judges whether the draft is grounded in the evidence."""

    verdict: Literal["supported", "partial", "unsupported"] = Field(
        description="supported = fully grounded; partial = some gaps; unsupported = requires revision"
    )
    uncertain_claims: list[str] = Field(
        default_factory=list,
        description="specific claims in the draft that aren't grounded in evidence",
    )
    critique: str = Field(
        default="",
        description="brief critique the solver can use to revise (leave empty if supported)",
    )


class EvaluatorOutput(BaseModel):
    """The evaluator produces the final user-facing answer."""

    final_answer: str = Field(description="the user-facing answer")
    reasoning: str = Field(description="one sentence: why this is the final answer")


# --------------------------------------------------------------------------- #
# System prompts
# --------------------------------------------------------------------------- #


PLANNER_SYS = (
    "You are the PLANNER. Given a user's question, restate the goal in one sentence, "
    "extract the specific factual subproblem the answer must address, and list any "
    "hard constraints (length limits, evidence-use requirements, format). Return "
    "STRICT JSON matching the schema."
)


MEMORY_SYS = (
    "You are the MEMORY node. You have a small document corpus. Given the "
    "subproblem, retrieve at most 5 short snippets that DIRECTLY bear on the "
    "answer. Each snippet must be a verbatim substring from the provided document "
    "(no paraphrasing). Return STRICT JSON matching the schema.\n\n"
    "Corpus:\n{corpus}"
)


SOLVER_SYS = (
    "You are the SOLVER. Given the subproblem, constraints, and evidence snippets, "
    "draft an answer that satisfies the constraints and cites the evidence you used "
    "by doc_id. If a critique from the verifier is provided, address it in the new "
    "draft. Keep the answer to 1-4 sentences unless the constraints say otherwise. "
    "Return STRICT JSON matching the schema."
)


VERIFIER_SYS = (
    "You are the VERIFIER. Given the subproblem, the draft answer, and the evidence "
    "the solver used, decide whether every substantive claim in the draft is "
    "supported by the evidence. Verdict = 'supported' if fully grounded; 'partial' "
    "if minor gaps; 'unsupported' if the draft must be revised. When 'partial' or "
    "'unsupported', list the specific uncertain claims and provide a short critique "
    "the solver can use. Return STRICT JSON matching the schema."
)


EVALUATOR_SYS = (
    "You are the EVALUATOR. Given the goal, the verified draft answer, and the "
    "verifier's verdict, produce the final user-facing answer. Do not add "
    "information beyond what the draft contains. Return STRICT JSON matching "
    "the schema."
)


def format_planner_input(user_task: str) -> str:
    return f"User question:\n{user_task}"


def format_memory_input(subproblem: str) -> str:
    return f"Subproblem to retrieve evidence for:\n{subproblem}"


def format_solver_input(
    subproblem: str,
    constraints: list[str],
    evidence: list[EvidenceItem],
    critique: str | None = None,
) -> str:
    parts = [f"Subproblem:\n{subproblem}", "Constraints:\n- " + "\n- ".join(constraints or ["(none)"])]
    ev_block = "\n".join(f"[{e.doc_id}] {e.snippet}" for e in evidence)
    parts.append(f"Evidence:\n{ev_block}")
    if critique:
        parts.append(f"Verifier critique (previous draft rejected):\n{critique}")
    return "\n\n".join(parts)


def format_verifier_input(
    subproblem: str, draft_answer: str, evidence: list[EvidenceItem]
) -> str:
    ev_block = "\n".join(f"[{e.doc_id}] {e.snippet}" for e in evidence)
    return (
        f"Subproblem:\n{subproblem}\n\nDraft answer:\n{draft_answer}\n\n"
        f"Evidence:\n{ev_block}"
    )


def format_evaluator_input(goal: str, draft_answer: str, verdict: str) -> str:
    return (
        f"Goal:\n{goal}\n\nDraft answer (verifier verdict = {verdict}):\n{draft_answer}"
    )
