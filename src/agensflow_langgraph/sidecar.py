"""Thread-local sidecar for pairing explicit rewards with decisions.

Why not mutate the graph state? Because many LangGraph states are `TypedDict` or
Pydantic — undeclared keys may be dropped or raise. The sidecar stashes recent
decision_ids in process-local memory keyed by thread_id.

Concurrency model: safe for asyncio (single event loop) and safe for CPython
threads (GIL protects dict ops). NOT safe across processes; users running a
distributed graph should pass explicit `decision_id` to `record_reward`.
"""

from __future__ import annotations

import threading
from collections import defaultdict, deque
from typing import Optional
from uuid import UUID

from agensflow_langgraph.contracts import RewardSubmission

_MAX_PER_THREAD = 64  # cap the sidecar so it doesn't grow unbounded on long-running threads


class DecisionSidecar:
    def __init__(self) -> None:
        self._by_thread: dict[str, deque[UUID]] = defaultdict(
            lambda: deque(maxlen=_MAX_PER_THREAD)
        )
        self._lock = threading.Lock()

    def record(self, thread_id: str | None, decision_id: UUID) -> None:
        key = thread_id or "-"
        with self._lock:
            self._by_thread[key].append(decision_id)

    def last(self, thread_id: str | None) -> list[UUID]:
        key = thread_id or "-"
        with self._lock:
            return list(self._by_thread.get(key, ()))

    def clear(self, thread_id: str | None) -> None:
        key = thread_id or "-"
        with self._lock:
            self._by_thread.pop(key, None)


_side = DecisionSidecar()


def _get_sidecar() -> DecisionSidecar:
    return _side


def record_reward(
    *,
    quality: float,
    thread_id: str | None = None,
    decision_id: UUID | str | None = None,
    axes: dict[str, float] | None = None,
    reasoning: str | None = None,
    server_url: str | None = None,
    tenant_key: str | None = None,
) -> None:
    """Submit an explicit reward to the policy server.

    Provide either `decision_id` (single decision) or `thread_id` (last N decisions
    on that thread from the sidecar). If both are None, this is a no-op.

    Fire-and-forget: no exception is raised on server failure. If you need
    guaranteed delivery for a research/production path, use the client directly.
    """
    from agensflow_langgraph.client import get_client

    ids: list[UUID]
    if decision_id is not None:
        ids = [UUID(str(decision_id))]
    else:
        ids = _side.last(thread_id)
    if not ids:
        return

    client = get_client(server_url, tenant_key)
    for did in ids:
        client.submit_reward(
            RewardSubmission(
                decision_id=did,
                quality=quality,
                quality_source="explicit",
                quality_axes=axes,
                quality_reasoning=reasoning,
            )
        )


async def arecord_reward(
    *,
    quality: float,
    thread_id: str | None = None,
    decision_id: UUID | str | None = None,
    axes: dict[str, float] | None = None,
    reasoning: str | None = None,
    server_url: str | None = None,
    tenant_key: str | None = None,
) -> None:
    """Async variant of `record_reward` — same semantics."""
    from agensflow_langgraph.client import get_client

    ids: list[UUID]
    if decision_id is not None:
        ids = [UUID(str(decision_id))]
    else:
        ids = _side.last(thread_id)
    if not ids:
        return

    client = get_client(server_url, tenant_key)
    for did in ids:
        await client.a_submit_reward(
            RewardSubmission(
                decision_id=did,
                quality=quality,
                quality_source="explicit",
                quality_axes=axes,
                quality_reasoning=reasoning,
            )
        )


# Optional helper for tests / explicit control
def _reset_sidecar() -> None:
    """Testing helper — clears the entire sidecar. NOT for production use."""
    with _side._lock:
        _side._by_thread.clear()
