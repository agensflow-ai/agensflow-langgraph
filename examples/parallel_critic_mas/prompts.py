"""Prompts + Pydantic schemas for the parallel_critic_mas.

Topology differs from evidence_heavy_mas:
  * The critic and verifier run IN PARALLEL after the solver.
  * The evaluator merges both signals.

Critic asks "is this answer *reasoning* sound?" (independent judgment on
merit of the argument).
Verifier asks "is this answer *evidence-grounded*?" (checks vs. the retrieved
snippets). Both landing independently gives the evaluator two orthogonal
signals to synthesize.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Schemas
# --------------------------------------------------------------------------- #


class PlannerOutput(BaseModel):
    goal: str = Field(description="one-sentence restatement of the user's goal")
    subproblem: str = Field(description="the specific factual question")


class EvidenceItem(BaseModel):
    doc_id: str
    snippet: str


class MemoryOutput(BaseModel):
    evidence: list[EvidenceItem] = Field(max_length=5)


class SolverOutput(BaseModel):
    draft_answer: str = Field(description="the drafted answer, 1-4 sentences")
    reasoning: str = Field(description="one sentence explaining the reasoning")


class CriticOutput(BaseModel):
    """The critic judges the REASONING (independent of evidence)."""

    reasoning_score: float = Field(
        ge=0.0, le=1.0, description="0=nonsense, 1=airtight reasoning"
    )
    reasoning_issues: list[str] = Field(
        default_factory=list, description="specific reasoning gaps if any"
    )


class VerifierOutput(BaseModel):
    """The verifier judges GROUNDING (whether claims match the evidence)."""

    verdict: Literal["supported", "partial", "unsupported"] = Field(
        description="supported = all claims grounded; partial = minor gaps; "
        "unsupported = requires revision"
    )
    ungrounded_claims: list[str] = Field(default_factory=list)


class EvaluatorOutput(BaseModel):
    """Evaluator merges critic + verifier signals into a final answer."""

    final_answer: str = Field(description="the user-facing answer")
    merged_reasoning: str = Field(
        description="one sentence: how the two signals combined into this verdict"
    )


# --------------------------------------------------------------------------- #
# System prompts
# --------------------------------------------------------------------------- #


PLANNER_SYS = (
    "You are the PLANNER. Given a user's question, restate the goal in one "
    "sentence and extract the specific factual subproblem. Return STRICT JSON."
)


MEMORY_SYS = (
    "You are the MEMORY node. You have a small document corpus. Given the "
    "subproblem, retrieve at most 5 verbatim snippets from the documents that "
    "DIRECTLY bear on the answer. Return STRICT JSON.\n\nCorpus:\n{corpus}"
)


SOLVER_SYS = (
    "You are the SOLVER. Given the subproblem and evidence snippets, draft an "
    "answer in 1-4 sentences and provide a one-sentence reasoning explanation. "
    "Cite evidence by doc_id when helpful. Return STRICT JSON."
)


CRITIC_SYS = (
    "You are the CRITIC. Given the subproblem and the solver's draft answer + "
    "reasoning, judge the quality of the reasoning INDEPENDENT of evidence. "
    "Does the argument hang together? Are there logical gaps? Score 0-1 and "
    "list specific issues. Return STRICT JSON."
)


VERIFIER_SYS = (
    "You are the VERIFIER. Given the subproblem, the draft answer, and the "
    "evidence, decide whether every substantive claim is supported by the "
    "evidence. verdict = 'supported' if all claims grounded; 'partial' if "
    "minor gaps; 'unsupported' if the draft needs revision. Return STRICT JSON."
)


EVALUATOR_SYS = (
    "You are the EVALUATOR. You've received two independent signals: the "
    "critic's reasoning score and the verifier's grounding verdict. Produce "
    "the final user-facing answer. If both signals are strong, ship the draft "
    "as-is. If only one is strong, hedge appropriately. Explain in one "
    "sentence how you merged the signals. Return STRICT JSON."
)


def format_planner_input(user_task: str) -> str:
    return f"User question:\n{user_task}"


def format_memory_input(subproblem: str) -> str:
    return f"Subproblem:\n{subproblem}"


def format_solver_input(
    subproblem: str, evidence: list[EvidenceItem]
) -> str:
    ev = "\n".join(f"[{e.doc_id}] {e.snippet}" for e in evidence)
    return f"Subproblem:\n{subproblem}\n\nEvidence:\n{ev}"


def format_critic_input(subproblem: str, draft: str, reasoning: str) -> str:
    return (
        f"Subproblem:\n{subproblem}\n\nDraft answer:\n{draft}\n\n"
        f"Solver reasoning:\n{reasoning}"
    )


def format_verifier_input(
    subproblem: str, draft: str, evidence: list[EvidenceItem]
) -> str:
    ev = "\n".join(f"[{e.doc_id}] {e.snippet}" for e in evidence)
    return f"Subproblem:\n{subproblem}\n\nDraft:\n{draft}\n\nEvidence:\n{ev}"


def format_evaluator_input(
    goal: str,
    draft: str,
    reasoning_score: float,
    verifier_verdict: str,
) -> str:
    return (
        f"Goal:\n{goal}\n\nDraft answer:\n{draft}\n\n"
        f"Critic reasoning score: {reasoning_score:.2f}\n"
        f"Verifier grounding verdict: {verifier_verdict}"
    )
