"""HTTP client for the AgensFlow policy server.

Thin wrapper over httpx with tenacity retries on the `/policy/select` path
(which MUST be retry-safe — server dedupes via idempotency_key). Execute + reward
calls are fire-and-forget: a lost recording never breaks the user's graph.

Sync + async APIs both provided; the decorator picks whichever matches the
wrapped node function.
"""

from __future__ import annotations

import os
from typing import Any

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from agensflow_langgraph.contracts import (
    DecisionListResponse,
    ExecutionAck,
    ExecutionResult,
    PolicyExportResponse,
    PolicyImportRequest,
    PolicyImportResponse,
    RewardAck,
    RewardSubmission,
    RoutingRequest,
    RoutingResponse,
)
from agensflow_langgraph.errors import ServerRejected, ServerUnreachable

_DEFAULT_TIMEOUT = 5.0

_CACHE: dict[tuple[str, str], "AgensFlowClient"] = {}


def get_client(
    server_url: str | None = None, tenant_key: str | None = None
) -> "AgensFlowClient":
    """Return a process-cached client for (server_url, tenant_key).

    Falls back to env vars: AGENSFLOW_SERVER_URL, AGENSFLOW_API_KEY.
    """
    url = server_url or os.environ.get("AGENSFLOW_SERVER_URL") or "http://localhost:8000"
    key = tenant_key or os.environ.get("AGENSFLOW_API_KEY") or ""
    cache_key = (url, key)
    if cache_key not in _CACHE:
        _CACHE[cache_key] = AgensFlowClient(url, key)
    return _CACHE[cache_key]


class AgensFlowClient:
    """Sync + async transport for the four LangGraph endpoints."""

    def __init__(
        self,
        base_url: str,
        tenant_key: str,
        timeout: float = _DEFAULT_TIMEOUT,
    ) -> None:
        self._base = base_url.rstrip("/")
        self._headers = {"Authorization": f"Bearer {tenant_key}"} if tenant_key else {}
        self._timeout = timeout

    # --- Sync ------------------------------------------------------------- #

    @retry(
        retry=retry_if_exception_type(ServerUnreachable),
        stop=stop_after_attempt(3),
        wait=wait_exponential(min=0.2, max=2.0),
        reraise=True,
    )
    def select(self, req: RoutingRequest) -> RoutingResponse:
        return self._post_model(
            "/langgraph/policy/select", req.model_dump(mode="json"), RoutingResponse
        )

    def record_execution(self, result: ExecutionResult) -> ExecutionAck | None:
        """Fire-and-forget. Returns None on failure — a lost recording is never fatal."""
        try:
            return self._post_model(
                "/langgraph/decision/execute",
                result.model_dump(mode="json"),
                ExecutionAck,
            )
        except (ServerUnreachable, ServerRejected):
            return None

    def submit_reward(self, reward: RewardSubmission) -> RewardAck | None:
        try:
            return self._post_model(
                "/langgraph/reward/submit",
                reward.model_dump(mode="json"),
                RewardAck,
            )
        except (ServerUnreachable, ServerRejected):
            return None

    def list_decisions(
        self, signature: str | None = None, limit: int = 100, offset: int = 0
    ) -> DecisionListResponse:
        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if signature is not None:
            params["signature"] = signature
        try:
            with httpx.Client(timeout=self._timeout) as c:
                r = c.get(
                    f"{self._base}/langgraph/decisions",
                    params=params,
                    headers=self._headers,
                )
        except (httpx.RequestError, httpx.TimeoutException) as e:
            raise ServerUnreachable(str(e)) from e
        self._raise_for_status(r)
        return DecisionListResponse.model_validate(r.json())

    def import_policy(self, req: PolicyImportRequest) -> PolicyImportResponse:
        return self._post_model(
            "/langgraph/policy/import", req.model_dump(mode="json"), PolicyImportResponse
        )

    def export_policy(self) -> PolicyExportResponse:
        """Fetch the tenant's full policy in the shape /policy/import accepts."""
        try:
            with httpx.Client(timeout=self._timeout) as c:
                r = c.get(
                    f"{self._base}/langgraph/policy/export",
                    headers=self._headers,
                )
        except (httpx.RequestError, httpx.TimeoutException) as e:
            raise ServerUnreachable(str(e)) from e
        self._raise_for_status(r)
        return PolicyExportResponse.model_validate(r.json())

    # --- Async ------------------------------------------------------------ #

    async def a_select(self, req: RoutingRequest) -> RoutingResponse:
        # Note: tenacity async retry semantics differ; simpler to inline the retry loop.
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                return await self._a_post_model(
                    "/langgraph/policy/select",
                    req.model_dump(mode="json"),
                    RoutingResponse,
                )
            except ServerUnreachable as e:
                last_exc = e
                if attempt < 2:
                    import asyncio
                    await asyncio.sleep(min(0.2 * (2**attempt), 2.0))
        assert last_exc is not None
        raise last_exc

    async def a_record_execution(self, result: ExecutionResult) -> ExecutionAck | None:
        try:
            return await self._a_post_model(
                "/langgraph/decision/execute",
                result.model_dump(mode="json"),
                ExecutionAck,
            )
        except (ServerUnreachable, ServerRejected):
            return None

    async def a_submit_reward(self, reward: RewardSubmission) -> RewardAck | None:
        try:
            return await self._a_post_model(
                "/langgraph/reward/submit",
                reward.model_dump(mode="json"),
                RewardAck,
            )
        except (ServerUnreachable, ServerRejected):
            return None

    # --- Private helpers -------------------------------------------------- #

    def _post_model(self, path: str, payload: dict, model_cls):
        try:
            with httpx.Client(timeout=self._timeout) as c:
                r = c.post(f"{self._base}{path}", json=payload, headers=self._headers)
        except (httpx.RequestError, httpx.TimeoutException) as e:
            raise ServerUnreachable(str(e)) from e
        self._raise_for_status(r)
        return model_cls.model_validate(r.json())

    async def _a_post_model(self, path: str, payload: dict, model_cls):
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as c:
                r = await c.post(
                    f"{self._base}{path}", json=payload, headers=self._headers
                )
        except (httpx.RequestError, httpx.TimeoutException) as e:
            raise ServerUnreachable(str(e)) from e
        self._raise_for_status(r)
        return model_cls.model_validate(r.json())

    @staticmethod
    def _raise_for_status(r: httpx.Response) -> None:
        if 500 <= r.status_code < 600:
            raise ServerUnreachable(f"{r.status_code}: {r.text[:200]}")
        if 400 <= r.status_code < 500:
            raise ServerRejected(f"{r.status_code}: {r.text[:200]}")
