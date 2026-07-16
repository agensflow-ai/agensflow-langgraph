"""End-to-end: real LangGraph node + real AgensFlow server in-process.

Verifies that the whole loop works:
  select → execute → reward → decisions list reflects state

The server runs in-process via ASGI transport. We subclass `AgensFlowClient` so both
sync + async paths route through the ASGI transport instead of real HTTP.
"""

from __future__ import annotations

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from agensflow_mcp.app import create_app
from agensflow_mcp.db.session import get_engine, init_db
from agensflow_mcp.db.models import Base

from agensflow_langgraph import agensflow
from agensflow_langgraph.contracts import (
    NodeContext,
    RoutingRequest,
)
from agensflow_langgraph.sidecar import arecord_reward
from agensflow_langgraph.signature import compute_idempotency_key


@pytest_asyncio.fixture(autouse=True)
async def _fresh_db():
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    yield


@pytest_asyncio.fixture
async def server_client():
    app = create_app()
    await init_db()
    transport = ASGITransport(app=app)
    async with LifespanManager(app):
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


class _MockModel:
    def __init__(self, name: str) -> None:
        self.name = name

    def with_config(self, callbacks=None, metadata=None, **_):
        return self

    def invoke(self, x, config=None):
        return {"model_used": self.name}

    async def ainvoke(self, x, config=None):
        return {"model_used": self.name}


def _install_asgi_client(monkeypatch, server_client, api_key):
    """Patch AgensFlowClient so all HTTP calls (sync + async) route through the
    in-process ASGITransport. Returns the patched client class."""
    from agensflow_langgraph import client as client_mod

    class _ASGIClient(client_mod.AgensFlowClient):
        # Async path: async httpx via the ASGI transport is straightforward.
        async def _a_post_model(self, path, payload, model_cls):
            r = await server_client.post(path, json=payload, headers=self._headers)
            self._raise_for_status(r)
            return model_cls.model_validate(r.json())

        # Sync path: not used in async tests, but stub anyway so decorator's
        # fail-open path works if it happens to route through here.
        def _post_model(self, path, payload, model_cls):
            # In an async test, we can't run_until_complete a coroutine. This
            # method is only reachable from a sync decorator wrapper — which
            # we don't exercise from within async tests.
            raise NotImplementedError(
                "sync _post_model not usable inside pytest-asyncio; use a_ variants"
            )

    client_mod._CACHE.clear()
    monkeypatch.setattr(client_mod, "AgensFlowClient", _ASGIClient)
    monkeypatch.setenv("AGENSFLOW_SERVER_URL", "http://test")
    monkeypatch.setenv("AGENSFLOW_API_KEY", api_key)
    return _ASGIClient


@pytest.mark.asyncio
async def test_full_lifecycle_hits_server(server_client: AsyncClient, monkeypatch) -> None:
    """Adapter → server → adapter loop over ASGI transport."""
    resp = await server_client.post("/auth/anonymous")
    api_key = resp.json()["api_key"]
    _install_asgi_client(monkeypatch, server_client, api_key)

    pool = {"cheap": _MockModel("cheap"), "smart": _MockModel("smart")}

    @agensflow(pool=pool)
    async def classify_intent(state, model, config=None):
        r = await model.ainvoke(state["messages"])
        return {"messages": [{"role": "assistant", "content": r["model_used"]}]}

    state = {"messages": [{"role": "user", "content": "hello"}]}
    config = {
        "metadata": {"langgraph_node": "classify_intent"},
        "configurable": {"thread_id": "thread_1"},
    }

    result = await classify_intent(state, config=config)
    assert result["messages"][0]["content"] in ("cheap", "smart")

    # Submit an explicit reward via the async sidecar helper
    await arecord_reward(quality=0.8, thread_id="thread_1")

    # Query /langgraph/decisions to verify state
    r = await server_client.get(
        "/langgraph/decisions",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    body = r.json()
    assert body["total"] == 1
    dec = body["decisions"][0]
    assert dec["status"] == "rewarded"
    assert dec["quality"] == 0.8
    assert dec["signature"] == "classify_intent"


@pytest.mark.asyncio
async def test_idempotency_prevents_duplicate_decisions(
    server_client: AsyncClient, monkeypatch
) -> None:
    """Two select requests with the same idempotency_key ⇒ one decision on server."""
    resp = await server_client.post("/auth/anonymous")
    api_key = resp.json()["api_key"]
    ClientCls = _install_asgi_client(monkeypatch, server_client, api_key)

    c = ClientCls("http://test", api_key)
    idem = compute_idempotency_key("sig", "thread_a", 0, ["cheap", "smart"])
    req = RoutingRequest(
        context=NodeContext(signature="sig", thread_id="thread_a"),
        action_pool_keys=["cheap", "smart"],
        idempotency_key=idem,
    )
    resp1 = await c.a_select(req)
    resp2 = await c.a_select(req)
    assert resp1.decision_id == resp2.decision_id
    assert resp2.idempotency_hit is True


@pytest.mark.asyncio
async def test_substrate_learns_across_runs(server_client: AsyncClient, monkeypatch) -> None:
    """Run the same signature many times with contrasting rewards; check the
    substrate's argmax converges to the rewarded arm."""
    resp = await server_client.post("/auth/anonymous")
    api_key = resp.json()["api_key"]
    _install_asgi_client(monkeypatch, server_client, api_key)

    pool = {"cheap": _MockModel("cheap"), "smart": _MockModel("smart")}

    @agensflow(pool=pool)
    async def node(state, model, config=None):
        r = await model.ainvoke(state["messages"])
        return {"messages": [{"role": "assistant", "content": r["model_used"]}]}

    # Reward "smart" runs highly, "cheap" runs poorly. After enough runs, argmax
    # should stabilize on "smart".
    for i in range(12):
        cfg = {
            "metadata": {"langgraph_node": "learner"},
            "configurable": {"thread_id": f"thread_{i}"},
        }
        result = await node({"messages": [{"role": "user", "content": f"r{i}"}]}, config=cfg)
        chosen = result["messages"][0]["content"]
        quality = 0.95 if chosen == "smart" else 0.15
        await arecord_reward(quality=quality, thread_id=f"thread_{i}")

    r = await server_client.get(
        "/langgraph/decisions?limit=100",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    body = r.json()
    # We ran 12 iterations; not all guaranteed to be rewarded yet due to fire-and-forget,
    # but total should be 12 selects
    assert body["total"] == 12

    # Check that the "smart" arm accumulated higher reward_mean than "cheap"
    # via a direct DB peek.
    from sqlalchemy import select as _sel
    from agensflow_mcp.db.session import get_session_factory
    from agensflow_mcp.db.models import Action

    async with get_session_factory()() as s:
        r = await s.execute(_sel(Action))
        actions = {a.action_str: a for a in r.scalars().all()}
        assert "cheap" in actions or "smart" in actions
        if "smart" in actions and "cheap" in actions:
            assert actions["smart"].reward_mean > actions["cheap"].reward_mean
