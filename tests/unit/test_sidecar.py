"""Sidecar semantics: keyed by thread_id, isolated across threads."""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from uuid import uuid4

from agensflow_langgraph.contracts import RewardAck
from agensflow_langgraph.sidecar import (
    _get_sidecar,
    _reset_sidecar,
    record_reward,
)


def setup_function():
    _reset_sidecar()


def test_record_and_last():
    side = _get_sidecar()
    d1 = uuid4()
    d2 = uuid4()
    side.record("thread_A", d1)
    side.record("thread_A", d2)
    assert side.last("thread_A") == [d1, d2]


def test_isolation_across_threads():
    side = _get_sidecar()
    d1 = uuid4()
    d2 = uuid4()
    side.record("thread_A", d1)
    side.record("thread_B", d2)
    assert side.last("thread_A") == [d1]
    assert side.last("thread_B") == [d2]


def test_clear_removes_only_that_thread():
    side = _get_sidecar()
    d1 = uuid4()
    d2 = uuid4()
    side.record("A", d1)
    side.record("B", d2)
    side.clear("A")
    assert side.last("A") == []
    assert side.last("B") == [d2]


def test_record_reward_by_thread_id():
    side = _get_sidecar()
    d1 = uuid4()
    d2 = uuid4()
    side.record("thread_X", d1)
    side.record("thread_X", d2)

    with patch("agensflow_langgraph.client.get_client") as mock_client:
        client = MagicMock()
        client.submit_reward.return_value = RewardAck(
            decision_id=d1, status="rewarded", reward_value=0.5
        )
        mock_client.return_value = client

        record_reward(thread_id="thread_X", quality=0.8)
        assert client.submit_reward.call_count == 2


def test_record_reward_by_decision_id_bypasses_sidecar():
    d = uuid4()
    with patch("agensflow_langgraph.client.get_client") as mock_client:
        client = MagicMock()
        client.submit_reward.return_value = RewardAck(
            decision_id=d, status="rewarded", reward_value=0.5
        )
        mock_client.return_value = client

        record_reward(decision_id=d, quality=0.9)
        assert client.submit_reward.call_count == 1
        call_kwargs = client.submit_reward.call_args.args[0]
        assert call_kwargs.decision_id == d


def test_record_reward_noop_when_nothing_stashed():
    """Sidecar is empty for this thread — should silently do nothing."""
    with patch("agensflow_langgraph.client.get_client") as mock_client:
        client = MagicMock()
        mock_client.return_value = client
        record_reward(thread_id="never_used", quality=0.5)
        client.submit_reward.assert_not_called()
