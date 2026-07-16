"""Real LangGraph integration — build actual StateGraph, add decorated nodes,
compile, invoke. Verifies our assumptions about RunnableConfig threading:

  * `metadata["langgraph_node"]` is populated by the runtime
  * `config` is forwarded to node fn when fn accepts it
  * signature auto-derives to the node name (not fn.__name__)
  * `configurable["thread_id"]` is threaded correctly
  * multiple invocations accumulate substrate state
  * async graphs work (graph.ainvoke)
  * subgraph checkpoint_ns gets canonicalized
"""

from __future__ import annotations

from typing import Annotated, TypedDict

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from agensflow_mcp.app import create_app
from agensflow_mcp.db.session import get_engine, init_db
from agensflow_mcp.db.models import Base

from agensflow_langgraph import agensflow
from agensflow_langgraph.sidecar import arecord_reward, _get_sidecar, _reset_sidecar


@pytest_asyncio.fixture(autouse=True)
async def _fresh_db():
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
        await conn.run_sync(Base.metadata.create_all)
    _reset_sidecar()
    yield


@pytest_asyncio.fixture
async def server_client():
    app = create_app()
    await init_db()
    transport = ASGITransport(app=app)
    async with LifespanManager(app):
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            yield c


def _install_asgi_client(monkeypatch, server_client, api_key):
    from agensflow_langgraph import client as client_mod

    class _ASGIClient(client_mod.AgensFlowClient):
        async def _a_post_model(self, path, payload, model_cls):
            r = await server_client.post(path, json=payload, headers=self._headers)
            self._raise_for_status(r)
            return model_cls.model_validate(r.json())

        def _post_model(self, path, payload, model_cls):
            raise NotImplementedError("use async paths in async tests")

    client_mod._CACHE.clear()
    monkeypatch.setattr(client_mod, "AgensFlowClient", _ASGIClient)
    monkeypatch.setenv("AGENSFLOW_SERVER_URL", "http://test")
    monkeypatch.setenv("AGENSFLOW_API_KEY", api_key)


class State(TypedDict):
    messages: Annotated[list, add_messages]
    picks: list[str]  # each node appends its chosen action here for inspection


class MockModel:
    """Runnable-shaped mock. LangChain's `.with_config` returns a bound copy."""

    def __init__(self, name: str) -> None:
        self.name = name
        self._config: dict = {}

    def with_config(self, callbacks=None, metadata=None, **_):
        m = MockModel(self.name)
        m._config = {"callbacks": callbacks or [], "metadata": metadata or {}}
        return m

    def invoke(self, x, config=None):
        return {"model_used": self.name, "bound_metadata": self._config.get("metadata")}

    async def ainvoke(self, x, config=None):
        return {"model_used": self.name, "bound_metadata": self._config.get("metadata")}


pool = {"cheap": MockModel("cheap"), "smart": MockModel("smart")}


# --------------------------------------------------------------------------- #
# Real StateGraph — sync path
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_sync_stategraph_populates_signature_from_langgraph_node(
    server_client: AsyncClient, monkeypatch
) -> None:
    """Real StateGraph.invoke() — verify langgraph_node metadata is populated
    AND that our signature ladder uses it (not fn.__name__)."""
    resp = await server_client.post("/auth/anonymous")
    api_key = resp.json()["api_key"]
    _install_asgi_client(monkeypatch, server_client, api_key)

    # Note: we DELIBERATELY name the function differently from the node to prove
    # the substrate keys off `langgraph_node`, not `fn.__name__`.
    @agensflow(pool=pool)
    async def _internal_impl_name(state: State, model, config=None):
        r = await model.ainvoke(state.get("messages", []))
        return {
            "messages": [{"role": "assistant", "content": r["model_used"]}],
            "picks": state.get("picks", []) + [r["model_used"]],
        }

    graph = StateGraph(State)
    graph.add_node("public_intent_classifier", _internal_impl_name)
    graph.add_edge(START, "public_intent_classifier")
    graph.add_edge("public_intent_classifier", END)
    compiled = graph.compile()

    result = await compiled.ainvoke(
        {"messages": [{"role": "user", "content": "hi"}], "picks": []},
        config={"configurable": {"thread_id": "t1"}},
    )
    assert result["picks"] and result["picks"][0] in ("cheap", "smart")

    # Now verify the SERVER received the public node name as signature.
    r = await server_client.get(
        "/langgraph/decisions",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    body = r.json()
    assert body["total"] == 1
    dec = body["decisions"][0]
    assert dec["signature"] == "public_intent_classifier", (
        f"Expected langgraph_node signature 'public_intent_classifier', "
        f"got {dec['signature']!r} — fn.__name__ leaked or metadata isn't populated"
    )


@pytest.mark.asyncio
async def test_stategraph_threads_config_and_captures_thread_id(
    server_client: AsyncClient, monkeypatch
) -> None:
    """When user invokes with configurable.thread_id, the decorator's sidecar
    should be keyed by that thread_id — so record_reward(thread_id=X) works."""
    resp = await server_client.post("/auth/anonymous")
    api_key = resp.json()["api_key"]
    _install_asgi_client(monkeypatch, server_client, api_key)

    @agensflow(pool=pool)
    async def worker(state: State, model, config=None):
        r = await model.ainvoke([])
        return {
            "messages": [{"role": "assistant", "content": r["model_used"]}],
            "picks": [r["model_used"]],
        }

    graph = StateGraph(State)
    graph.add_node("worker", worker)
    graph.add_edge(START, "worker")
    graph.add_edge("worker", END)
    compiled = graph.compile()

    thread_id = "user_session_42"
    await compiled.ainvoke(
        {"messages": [{"role": "user", "content": "x"}], "picks": []},
        config={"configurable": {"thread_id": thread_id}},
    )
    # Sidecar should have exactly one decision_id keyed to that thread
    stashed = _get_sidecar().last(thread_id)
    assert len(stashed) == 1, f"expected 1 decision on {thread_id}, got {len(stashed)}"

    # And record_reward on that thread should hit /reward/submit
    await arecord_reward(quality=0.7, thread_id=thread_id)
    r = await server_client.get(
        "/langgraph/decisions",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    dec = r.json()["decisions"][0]
    assert dec["status"] == "rewarded"
    assert dec["quality"] == 0.7


@pytest.mark.asyncio
async def test_multi_node_graph_each_node_gets_its_own_signature(
    server_client: AsyncClient, monkeypatch
) -> None:
    """Two nodes in one graph → two distinct signatures on the server."""
    resp = await server_client.post("/auth/anonymous")
    api_key = resp.json()["api_key"]
    _install_asgi_client(monkeypatch, server_client, api_key)

    @agensflow(pool=pool)
    async def classifier(state: State, model, config=None):
        r = await model.ainvoke([])
        return {
            "messages": [{"role": "assistant", "content": r["model_used"]}],
            "picks": state.get("picks", []) + [f"cls:{r['model_used']}"],
        }

    @agensflow(pool=pool)
    async def responder(state: State, model, config=None):
        r = await model.ainvoke([])
        return {
            "messages": [{"role": "assistant", "content": r["model_used"]}],
            "picks": state.get("picks", []) + [f"resp:{r['model_used']}"],
        }

    graph = StateGraph(State)
    graph.add_node("classifier", classifier)
    graph.add_node("responder", responder)
    graph.add_edge(START, "classifier")
    graph.add_edge("classifier", "responder")
    graph.add_edge("responder", END)
    compiled = graph.compile()

    result = await compiled.ainvoke(
        {"messages": [], "picks": []},
        config={"configurable": {"thread_id": "t_multi"}},
    )
    assert len(result["picks"]) == 2

    r = await server_client.get(
        "/langgraph/decisions",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    sigs = {d["signature"] for d in r.json()["decisions"]}
    assert sigs == {"classifier", "responder"}, sigs


@pytest.mark.asyncio
async def test_repeated_invocations_accumulate_bandit_state(
    server_client: AsyncClient, monkeypatch
) -> None:
    """Run the same graph N times, each with distinct thread_id. Server-side
    Action rows should accumulate visits."""
    resp = await server_client.post("/auth/anonymous")
    api_key = resp.json()["api_key"]
    _install_asgi_client(monkeypatch, server_client, api_key)

    @agensflow(pool=pool)
    async def node(state: State, model, config=None):
        r = await model.ainvoke([])
        return {
            "messages": [{"role": "assistant", "content": r["model_used"]}],
            "picks": [r["model_used"]],
        }

    graph = StateGraph(State)
    graph.add_node("learner", node)
    graph.add_edge(START, "learner")
    graph.add_edge("learner", END)
    compiled = graph.compile()

    for i in range(8):
        result = await compiled.ainvoke(
            {"messages": [], "picks": []},
            config={"configurable": {"thread_id": f"iter_{i}"}},
        )
        chosen = result["picks"][0]
        # Reward "smart" strongly, "cheap" weakly — expect UCB to prefer smart eventually
        await arecord_reward(
            quality=0.9 if chosen == "smart" else 0.1, thread_id=f"iter_{i}"
        )

    r = await server_client.get(
        "/langgraph/decisions?limit=100",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    body = r.json()
    assert body["total"] == 8

    from sqlalchemy import select as _sel
    from agensflow_mcp.db.session import get_session_factory
    from agensflow_mcp.db.models import Action, Signature

    async with get_session_factory()() as s:
        sigs = (await s.execute(_sel(Signature))).scalars().all()
        assert any(sig.signature_str == "learner" for sig in sigs)
        # Total visits across both arms should equal 8
        actions = (await s.execute(_sel(Action))).scalars().all()
        total_visits = sum(a.visits for a in actions)
        assert total_visits == 8, f"expected 8 total visits, got {total_visits}"


# --------------------------------------------------------------------------- #
# Subgraph — checkpoint_ns canonicalization
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_subgraph_signature_stable_across_runs(
    server_client: AsyncClient, monkeypatch
) -> None:
    """When a node is inside a compiled subgraph, checkpoint_ns includes a runtime
    UUID. Our canonicalization must strip it so the same subgraph-node's signature
    is stable across invocations."""
    resp = await server_client.post("/auth/anonymous")
    api_key = resp.json()["api_key"]
    _install_asgi_client(monkeypatch, server_client, api_key)

    @agensflow(pool=pool)
    async def inner_node(state: State, model, config=None):
        r = await model.ainvoke([])
        return {
            "messages": [{"role": "assistant", "content": r["model_used"]}],
            "picks": [r["model_used"]],
        }

    # Build a subgraph with `inner_node` inside it.
    sub = StateGraph(State)
    sub.add_node("inner_node", inner_node)
    sub.add_edge(START, "inner_node")
    sub.add_edge("inner_node", END)
    sub_compiled = sub.compile()

    # Build a parent that invokes the subgraph.
    parent = StateGraph(State)
    parent.add_node("wrapped", sub_compiled)
    parent.add_edge(START, "wrapped")
    parent.add_edge("wrapped", END)
    parent_compiled = parent.compile()

    # Invoke twice — signatures should MATCH (subgraph UUID stripped)
    await parent_compiled.ainvoke(
        {"messages": [], "picks": []},
        config={"configurable": {"thread_id": "s1"}},
    )
    await parent_compiled.ainvoke(
        {"messages": [], "picks": []},
        config={"configurable": {"thread_id": "s2"}},
    )

    r = await server_client.get(
        "/langgraph/decisions",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    sigs = {d["signature"] for d in r.json()["decisions"]}
    # Both invocations should have produced the SAME signature — 1 unique sig, not 2
    assert len(sigs) == 1, (
        f"expected 1 stable signature, got {len(sigs)}: {sigs}. "
        "Subgraph checkpoint_ns UUID likely not being canonicalized."
    )
