"""Schemas + prompts for the security_domain MAS.

Solver has 3 skill cards (concise / cot / evidence) — the choice of skill card
is one dimension of the solver arm key (the other dimension is the model tier:
haiku / fast / mini).

Skill cards, matching the paper's naming:
  * concise   — short, direct answer. 1-3 sentences.
  * cot       — chain-of-thought reasoning before the answer.
  * evidence  — every claim cited by doc_id from the retrieved evidence.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class PlannerOutput(BaseModel):
    goal: str = Field(description="one-sentence restatement of the user's goal")
    subproblem: str = Field(description="the specific factual question")


class EvidenceItem(BaseModel):
    doc_id: str
    snippet: str


class MemoryOutput(BaseModel):
    evidence: list[EvidenceItem] = Field(max_length=5)


class SolverOutput(BaseModel):
    draft_answer: str = Field(description="the drafted answer")
    reasoning: str = Field(description="one sentence explaining how you arrived at the answer")


class VerifierOutput(BaseModel):
    verdict: Literal["supported", "partial", "unsupported"] = Field(
        description="supported = all claims grounded; partial = minor gaps; "
        "unsupported = requires revision"
    )
    ungrounded_claims: list[str] = Field(default_factory=list)


class EvaluatorOutput(BaseModel):
    final_answer: str = Field(description="the user-facing answer")
    merged_reasoning: str = Field(description="one sentence: verdict summary")


# --------------------------------------------------------------------------- #
# System prompts — stage-scoped except for the solver, which has three cards
# --------------------------------------------------------------------------- #


PLANNER_SYS = (
    "You are the PLANNER. Given a user's question, restate the goal in one "
    "sentence and extract the specific factual subproblem. Return STRICT JSON."
)


MEMORY_SYS = (
    "You are the MEMORY node. You have a small corpus of security advisories. "
    "Given the subproblem, retrieve at most 5 verbatim snippets from the "
    "documents that DIRECTLY bear on the answer. Return STRICT JSON.\n\n"
    "Corpus:\n{corpus}"
)


VERIFIER_SYS = (
    "You are the VERIFIER. Check whether the draft answer's claims are "
    "supported by the retrieved evidence snippets. Verdict is one of "
    "'supported' (all claims grounded), 'partial' (minor gaps), or "
    "'unsupported' (needs revision). Return STRICT JSON."
)


EVALUATOR_SYS = (
    "You are the EVALUATOR. Merge the solver's draft answer with the verifier "
    "verdict into a final user-facing answer. If the verdict is 'unsupported' "
    "and revisions are exhausted, return the draft with a caveat. Return "
    "STRICT JSON."
)


# --- Solver skill cards ---------------------------------------------------- #

SOLVER_SYS_CONCISE = (
    "You are the SOLVER (concise card). Given the subproblem and evidence "
    "snippets, produce a SHORT direct answer in 1-3 sentences and a "
    "one-sentence reasoning explanation. No preamble, no bullet lists. "
    "Return STRICT JSON."
)


SOLVER_SYS_COT = (
    "You are the SOLVER (chain-of-thought card). Given the subproblem and "
    "evidence, first think step-by-step through the reasoning inside the "
    "'reasoning' field (2-4 concise steps), then produce the answer in "
    "'draft_answer' (1-4 sentences). Return STRICT JSON."
)


SOLVER_SYS_EVIDENCE = (
    "You are the SOLVER (evidence-cite card). Given the subproblem and "
    "evidence snippets, produce an answer where EVERY factual claim is "
    "immediately followed by a bracketed [doc_id] citation from the "
    "provided evidence. The 'reasoning' field explains how you selected "
    "which snippets support which claims. Return STRICT JSON."
)


SOLVER_SYSTEMS = {
    "concise":  SOLVER_SYS_CONCISE,
    "cot":      SOLVER_SYS_COT,
    "evidence": SOLVER_SYS_EVIDENCE,
}


# --- Prompt formatting helpers -------------------------------------------- #


def format_planner_input(user_task: str) -> str:
    return f"User question: {user_task}"


def format_memory_input(subproblem: str) -> str:
    return f"Subproblem: {subproblem}"


def format_solver_input(subproblem: str, evidence: list[EvidenceItem]) -> str:
    lines = [f"Subproblem: {subproblem}", "", "Evidence:"]
    for e in evidence:
        lines.append(f"  [{e.doc_id}] {e.snippet}")
    return "\n".join(lines)


def format_verifier_input(subproblem: str, draft: str, evidence: list[EvidenceItem]) -> str:
    lines = [f"Subproblem: {subproblem}", f"Draft answer: {draft}", "", "Evidence:"]
    for e in evidence:
        lines.append(f"  [{e.doc_id}] {e.snippet}")
    return "\n".join(lines)


def format_evaluator_input(goal: str, draft: str, verdict: str) -> str:
    return (
        f"Goal: {goal}\n"
        f"Draft answer: {draft}\n"
        f"Verifier verdict: {verdict}"
    )
