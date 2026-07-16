"""Pydantic contracts — client side.

MIRROR of the server's `agensflow_mcp.http_routes.langgraph_contracts`. Kept in sync
by a shared JSON-fixture contract test in both repos. Every model carries
`contract_version: Literal["v1"]` so schema evolution stays safe.

Field-naming rule: all field names are generic. No private-layer vocabulary (proprietary metrics,
cross-judge audit, etc.) ever leaves the OSS side — even prospectively.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import BaseModel, Field


# --------------------------------------------------------------------------- #
# Client → server
# --------------------------------------------------------------------------- #


class NodeContext(BaseModel):
    contract_version: Literal["v1"] = "v1"
    signature: str = Field(..., max_length=512)
    langgraph_node: Optional[str] = Field(default=None, max_length=128)
    checkpoint_ns_canonical: Optional[str] = Field(default=None, max_length=512)
    thread_id: Optional[str] = Field(default=None, max_length=128)
    step: Optional[int] = None


class RoutingRequest(BaseModel):
    contract_version: Literal["v1"] = "v1"
    context: NodeContext
    action_pool_keys: list[str] = Field(..., min_length=1, max_length=64)
    idempotency_key: str = Field(..., max_length=128)


class ExecutionResult(BaseModel):
    contract_version: Literal["v1"] = "v1"
    decision_id: UUID
    action: str = Field(..., max_length=64)
    cost_usd: float = Field(..., ge=0.0)
    latency_s: float = Field(..., ge=0.0)
    input_tokens: int = Field(..., ge=0)
    output_tokens: int = Field(..., ge=0)
    thinking_tokens: Optional[int] = Field(default=None, ge=0)
    input_messages: Optional[list[dict]] = None
    output_message: Optional[dict] = None
    error: Optional[str] = None
    error_type: Optional[str] = Field(default=None, max_length=32)
    langsmith_run_id: Optional[str] = Field(default=None, max_length=64)


class RewardSubmission(BaseModel):
    contract_version: Literal["v1"] = "v1"
    decision_id: UUID
    quality: float = Field(..., ge=0.0, le=1.0)
    quality_source: Literal["explicit", "verifier", "judge"]
    quality_axes: Optional[dict[str, float]] = None
    quality_reasoning: Optional[str] = None


class PolicyImportRequest(BaseModel):
    contract_version: Literal["v1"] = "v1"
    policy: dict[str, dict[str, dict[str, float]]]
    source_hash: Optional[str] = Field(default=None, max_length=64)


# --------------------------------------------------------------------------- #
# Server → client
# --------------------------------------------------------------------------- #


class RoutingResponse(BaseModel):
    contract_version: Literal["v1"] = "v1"
    decision_id: UUID
    action: str
    policy_snapshot: dict[str, float]
    ucb_scores: dict[str, float | str]
    reasoning: str
    idempotency_hit: bool


class ExecutionAck(BaseModel):
    contract_version: Literal["v1"] = "v1"
    decision_id: UUID
    status: Literal["executed", "unknown"]


class RewardAck(BaseModel):
    contract_version: Literal["v1"] = "v1"
    decision_id: UUID
    status: Literal["rewarded", "unknown"]
    reward_value: Optional[float] = None


class DecisionRecordPublic(BaseModel):
    contract_version: Literal["v1"] = "v1"
    decision_id: UUID
    signature: str
    action: str
    status: Literal["selected", "executed", "rewarded"]
    quality: Optional[float] = None
    reward_value: Optional[float] = None
    cost_usd: Optional[float] = None
    latency_s: Optional[float] = None
    tokens_input: Optional[int] = None
    tokens_output: Optional[int] = None
    thinking_tokens: Optional[int] = None
    error_type: Optional[str] = None
    selected_at: datetime
    executed_at: Optional[datetime] = None
    rewarded_at: Optional[datetime] = None


class DecisionListResponse(BaseModel):
    contract_version: Literal["v1"] = "v1"
    decisions: list[DecisionRecordPublic]
    total: int


class PolicyImportResponse(BaseModel):
    contract_version: Literal["v1"] = "v1"
    signatures_merged: int
    actions_merged: int


class PolicyExportResponse(BaseModel):
    contract_version: Literal["v1"] = "v1"
    policy: dict[str, dict[str, dict[str, float]]]
    n_signatures: int
    n_actions: int
