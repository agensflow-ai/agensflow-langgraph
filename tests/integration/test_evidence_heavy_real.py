"""Real-LLM integration test — opt-in via AGF_ENABLE_REAL_LLM_TESTS=1.

Runs the evidence_heavy_mas graph against real OpenRouter for 3 iterations of
one easy task. Asserts that CostCapture actually recorded tokens (proving the
whole callback dispatch chain works with real ChatOpenAI), that the graph
routed each node, and that the substrate accumulated bandit stats server-side.

Cost: ~$0.30-$1.00 per test run. Skipped by default in CI.

To run manually:

    export AGF_ENABLE_REAL_LLM_TESTS=1
    export OPENROUTER_API_KEY=sk-or-...
    pytest tests/integration/test_evidence_heavy_real.py -v -s
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio
from asgi_lifespan import LifespanManager
from httpx import ASGITransport, AsyncClient

from agensflow_mcp.app import create_app
from agensflow_mcp.db.session import get_engine, init_db
from agensflow_mcp.db.models import Base

from agensflow_langgraph import arecord_reward


pytestmark = pytest.mark.skipif(
    os.getenv("AGF_ENABLE_REAL_LLM_TESTS") != "1",
    reason="opt-in real-LLM test — set AGF_ENABLE_REAL_LLM_TESTS=1 to run",
)


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


def _install_asgi_client(monkeypatch, server_client, api_key):
    from agensflow_langgraph import client as client_mod

    class _ASGIClient(client_mod.AgensFlowClient):
        async def _a_post_model(self, path, payload, model_cls):
            r = await server_client.post(path, json=payload, headers=self._headers)
            self._raise_for_status(r)
            return model_cls.model_validate(r.json())

        def _post_model(self, path, payload, model_cls):
            raise NotImplementedError("use async in async tests")

    client_mod._CACHE.clear()
    monkeypatch.setattr(client_mod, "AgensFlowClient", _ASGIClient)
    monkeypatch.setenv("AGENSFLOW_SERVER_URL", "http://test")
    monkeypatch.setenv("AGENSFLOW_API_KEY", api_key)


@pytest.mark.asyncio
async def test_evidence_heavy_mas_real_openrouter(
    server_client: AsyncClient, monkeypatch
) -> None:
    """One task × 3 iterations against real OpenRouter. Asserts real tokens
    were recorded server-side and the substrate accumulated stats."""
    if not os.environ.get("OPENROUTER_API_KEY"):
        pytest.skip("OPENROUTER_API_KEY not set")

    resp = await server_client.post("/auth/anonymous")
    api_key = resp.json()["api_key"]
    _install_asgi_client(monkeypatch, server_client, api_key)

    # Deferred imports so pytest discovery works without OPENROUTER_API_KEY.
    from examples.evidence_heavy_mas.graph import build_graph, build_pools
    from examples.evidence_heavy_mas.tasks import TASKS, score_answer

    pools = build_pools()
    compiled = build_graph(pools)
    task = next(t for t in TASKS if t.difficulty == "easy")

    for i in range(3):
        thread_id = f"real_test_{i}"
        result = await compiled.ainvoke(
            {"user_task": task.question, "revision_count": 0, "trace": []},
            config={"configurable": {"thread_id": thread_id}},
        )
        assert result.get("final_answer"), "graph didn't reach evaluator"
        quality, axes, _ = score_answer(task, result["final_answer"])
        await arecord_reward(quality=quality, thread_id=thread_id, axes=axes)

    # Query the server: 3 iterations × 5 nodes = 15 decisions
    r = await server_client.get(
        "/langgraph/decisions?limit=100",
        headers={"Authorization": f"Bearer {api_key}"},
    )
    body = r.json()
    assert body["total"] >= 15, f"expected >=15 decisions, got {body['total']}"

    # Every node's decisions should have >0 tokens recorded (proves CostCapture
    # actually fired for the real ChatOpenAI calls).
    zero_token_decisions = [
        d for d in body["decisions"] if (d.get("cost_usd") or 0) == 0
        and d["status"] == "rewarded"
    ]
    # Some providers may not surface cost — but tokens should be recorded.
    # Fetch via DB directly to check tokens_input.
    from sqlalchemy import select as _sel
    from agensflow_mcp.db.session import get_session_factory
    from agensflow_mcp.db.models import DecisionRecord

    async with get_session_factory()() as s:
        rows = (
            await s.execute(_sel(DecisionRecord).where(DecisionRecord.status == "rewarded"))
        ).scalars().all()
        assert rows, "no rewarded decisions found in DB"
        zero_token = [d for d in rows if (d.tokens_input or 0) == 0]
        assert not zero_token, (
            f"{len(zero_token)}/{len(rows)} rewarded decisions had 0 input_tokens — "
            f"CostCapture didn't fire for real ChatOpenAI calls. "
            f"Nodes affected: {[d.action for d in zero_token[:3]]}"
        )

    # Signatures should show all 5 node names
    signatures = {d["signature"] for d in body["decisions"]}
    assert signatures == {"planner", "memory", "solver", "verifier", "evaluator"}, signatures
