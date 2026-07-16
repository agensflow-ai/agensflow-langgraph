"""Decorator behavior: sync/async, fail-open, model injection, sidecar recording."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest

from agensflow_langgraph import agensflow
from agensflow_langgraph.contracts import (
    ExecutionAck,
    RewardAck,
    RoutingResponse,
)
from agensflow_langgraph.errors import InvalidPool
from agensflow_langgraph.sidecar import _get_sidecar, _reset_sidecar


class _MockModel:
    """Stand-in for a LangChain Runnable — supports .with_config().invoke(...)."""

    def __init__(self, name: str, response: str = "ok") -> None:
        self.name = name
        self.response = response
        self._config: dict = {}

    def with_config(self, callbacks=None, metadata=None, **_):
        m = _MockModel(self.name, self.response)
        m._config = {"callbacks": callbacks or [], "metadata": metadata or {}}
        return m

    def invoke(self, messages, config=None):
        return {"model_used": self.name, "response": self.response}

    async def ainvoke(self, messages, config=None):
        return {"model_used": self.name, "response": self.response}


@pytest.fixture(autouse=True)
def reset_sidecar():
    _reset_sidecar()
    yield
    _reset_sidecar()


def _mock_routing_response(action: str = "cheap") -> RoutingResponse:
    return RoutingResponse(
        decision_id=uuid4(),
        action=action,
        policy_snapshot={"n_visits": 0.0, "reward_mean": 0.0},
        ucb_scores={"cheap": "inf", "smart": "inf"},
        reasoning="test",
        idempotency_hit=False,
    )


def test_empty_pool_raises():
    with pytest.raises(InvalidPool):
        agensflow(pool={})


def test_sync_decorator_injects_model():
    pool = {"cheap": _MockModel("cheap"), "smart": _MockModel("smart")}

    with patch("agensflow_langgraph.decorator.get_client") as mock_client:
        client = MagicMock()
        client.select.return_value = _mock_routing_response("cheap")
        client.record_execution.return_value = ExecutionAck(
            decision_id=uuid4(), status="executed"
        )
        mock_client.return_value = client

        @agensflow(pool=pool)
        def node(state, model):
            # `model` was injected — verify it's a wrapped version of "cheap"
            result = model.invoke(state["messages"])
            return {"messages": [{"role": "assistant", "content": result["response"]}]}

        out = node({"messages": [{"role": "user", "content": "hi"}]})
        assert out["messages"][0]["content"] == "ok"
        client.select.assert_called_once()
        client.record_execution.assert_called_once()


@pytest.mark.asyncio
async def test_async_decorator_awaits_model():
    pool = {"cheap": _MockModel("cheap"), "smart": _MockModel("smart")}

    with patch("agensflow_langgraph.decorator.get_client") as mock_client:
        client = MagicMock()
        client.a_select = AsyncMock(return_value=_mock_routing_response("smart"))
        client.a_record_execution = AsyncMock(
            return_value=ExecutionAck(decision_id=uuid4(), status="executed")
        )
        mock_client.return_value = client

        @agensflow(pool=pool)
        async def node(state, model):
            result = await model.ainvoke(state["messages"])
            return {"messages": [{"role": "assistant", "content": result["response"]}]}

        out = await node({"messages": [{"role": "user", "content": "hi"}]})
        assert out["messages"][0]["content"] == "ok"
        client.a_select.assert_awaited_once()
        client.a_record_execution.assert_awaited_once()


def test_fail_open_uses_fallback_when_server_down():
    from agensflow_langgraph.errors import ServerUnreachable

    pool = {"cheap": _MockModel("cheap"), "smart": _MockModel("smart")}

    with patch("agensflow_langgraph.decorator.get_client") as mock_client:
        client = MagicMock()
        client.select.side_effect = ServerUnreachable("boom")
        mock_client.return_value = client

        @agensflow(pool=pool, fallback_action="cheap")
        def node(state, model):
            result = model.invoke(state["messages"])
            return {"messages": [{"role": "assistant", "content": result["response"]}]}

        # Doesn't raise — proceeds with fallback
        out = node({"messages": [{"role": "user", "content": "hi"}]})
        assert out["messages"][0]["content"] == "ok"
        # No record_execution when fail-open (decision_id is None)
        client.record_execution.assert_not_called()


def test_fail_closed_raises():
    from agensflow_langgraph.errors import ServerUnreachable

    pool = {"cheap": _MockModel("cheap")}

    with patch("agensflow_langgraph.decorator.get_client") as mock_client:
        client = MagicMock()
        client.select.side_effect = ServerUnreachable("boom")
        mock_client.return_value = client

        @agensflow(pool=pool, fail_closed=True)
        def node(state, model):
            return {"messages": []}

        with pytest.raises(ServerUnreachable):
            node({"messages": []})


def test_decision_id_recorded_to_sidecar():
    pool = {"cheap": _MockModel("cheap")}
    routing = _mock_routing_response("cheap")

    with patch("agensflow_langgraph.decorator.get_client") as mock_client:
        client = MagicMock()
        client.select.return_value = routing
        client.record_execution.return_value = ExecutionAck(
            decision_id=routing.decision_id, status="executed"
        )
        mock_client.return_value = client

        @agensflow(pool=pool)
        def node(state, model):
            return {"messages": []}

        node(
            {"messages": []},
            config={"configurable": {"thread_id": "thread_A"}, "metadata": {}},
        )
        assert routing.decision_id in _get_sidecar().last("thread_A")


def test_config_forwarded_when_fn_accepts_it():
    pool = {"cheap": _MockModel("cheap")}
    received = {}

    with patch("agensflow_langgraph.decorator.get_client") as mock_client:
        client = MagicMock()
        client.select.return_value = _mock_routing_response("cheap")
        client.record_execution.return_value = ExecutionAck(
            decision_id=uuid4(), status="executed"
        )
        mock_client.return_value = client

        @agensflow(pool=pool)
        def node(state, model, config=None):
            received["config"] = config
            return {"messages": []}

        cfg = {"metadata": {"langgraph_node": "n"}, "configurable": {}}
        node({"messages": []}, config=cfg)
        assert received["config"] is cfg


def test_capture_messages_disabled_by_default():
    pool = {"cheap": _MockModel("cheap")}
    captured_exec = {}

    with patch("agensflow_langgraph.decorator.get_client") as mock_client:
        client = MagicMock()
        client.select.return_value = _mock_routing_response("cheap")

        def _cap(exec_result):
            captured_exec["result"] = exec_result
            return ExecutionAck(decision_id=exec_result.decision_id, status="executed")

        client.record_execution.side_effect = _cap
        mock_client.return_value = client

        @agensflow(pool=pool)
        def node(state, model):
            return {"messages": [{"role": "assistant", "content": "hi"}]}

        node({"messages": [{"role": "user", "content": "hello"}]})
        # capture_messages default is False
        assert captured_exec["result"].input_messages is None
        assert captured_exec["result"].output_message is None


def test_capture_messages_enabled_sends_messages():
    pool = {"cheap": _MockModel("cheap")}
    captured_exec = {}

    with patch("agensflow_langgraph.decorator.get_client") as mock_client:
        client = MagicMock()
        client.select.return_value = _mock_routing_response("cheap")

        def _cap(exec_result):
            captured_exec["result"] = exec_result
            return ExecutionAck(decision_id=exec_result.decision_id, status="executed")

        client.record_execution.side_effect = _cap
        mock_client.return_value = client

        @agensflow(pool=pool, capture_messages=True)
        def node(state, model):
            return {"messages": [{"role": "assistant", "content": "hi"}]}

        node({"messages": [{"role": "user", "content": "hello"}]})
        assert captured_exec["result"].input_messages is not None
        assert captured_exec["result"].output_message is not None


def test_redact_hook_applied_to_messages():
    pool = {"cheap": _MockModel("cheap")}
    captured_exec = {}

    def redact_all(payload):
        if isinstance(payload, list):
            return [{"role": "?", "content": "[REDACTED]"} for _ in payload]
        return {"role": "?", "content": "[REDACTED]"}

    with patch("agensflow_langgraph.decorator.get_client") as mock_client:
        client = MagicMock()
        client.select.return_value = _mock_routing_response("cheap")

        def _cap(exec_result):
            captured_exec["result"] = exec_result
            return ExecutionAck(decision_id=exec_result.decision_id, status="executed")

        client.record_execution.side_effect = _cap
        mock_client.return_value = client

        @agensflow(pool=pool, capture_messages=True, redact=redact_all)
        def node(state, model):
            return {"messages": [{"role": "assistant", "content": "secret"}]}

        node({"messages": [{"role": "user", "content": "PII"}]})
        assert captured_exec["result"].input_messages[0]["content"] == "[REDACTED]"
