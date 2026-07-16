"""CostCapture verification against a real LangChain BaseChatModel dispatch path.

This is the unit-ish test that closes the "MockModel isn't proof" gap for cost
capture. It uses a stub subclass of `BaseChatModel` (LangChain's canonical base
class for chat models like `ChatOpenAI`), which returns an `AIMessage` with the
canonical `usage_metadata` shape. Both `.invoke()` and `.ainvoke()` are exercised,
including through a compiled LangGraph node (where `.with_config(callbacks=[...])`
must propagate).

No OpenRouter / OpenAI keys needed — this validates the same dispatch path that
`ChatOpenAI(base_url=openrouter, ...)` uses.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langgraph.graph import END, START, StateGraph

from agensflow_langgraph import agensflow
from agensflow_langgraph.callbacks import CostCapture
from agensflow_langgraph.contracts import ExecutionAck, RoutingResponse
from agensflow_langgraph.sidecar import _reset_sidecar


class StubChatModel(BaseChatModel):
    """A `BaseChatModel` that returns an `AIMessage` with canonical `usage_metadata`.

    Exercises the same call path as `ChatOpenAI(base_url=...)` without needing
    real HTTP. The critical property: `AIMessage.usage_metadata` is populated
    with the LangChain-0.3+ shape, including `output_token_details.reasoning`
    which our CostCapture must read for OpenRouter reasoning-model support.
    """

    reasoning_tokens: int = 0

    @property
    def _llm_type(self) -> str:
        return "stub_chat"

    def _generate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._make_result()

    async def _agenerate(
        self,
        messages: list[BaseMessage],
        stop: list[str] | None = None,
        run_manager: CallbackManagerForLLMRun | None = None,
        **kwargs: Any,
    ) -> ChatResult:
        return self._make_result()

    def _make_result(self) -> ChatResult:
        # This is the shape ChatOpenAI (LangChain 0.3+) surfaces on every response.
        usage_metadata = {
            "input_tokens": 123,
            "output_tokens": 45,
            "total_tokens": 168,
            "input_token_details": {"cache_read": 10},
            "output_token_details": {"reasoning": self.reasoning_tokens},
        }
        message = AIMessage(
            content="stub-response",
            usage_metadata=usage_metadata,
            response_metadata={"model_name": "stub/model-v1"},
        )
        return ChatResult(generations=[ChatGeneration(message=message)])


# --------------------------------------------------------------------------- #
# 1. Direct: CostCapture reads AIMessage.usage_metadata via with_config
# --------------------------------------------------------------------------- #


def test_costcapture_reads_usage_metadata_via_with_config():
    """The most basic path: bind CostCapture via .with_config, call the model,
    verify capture is populated from AIMessage.usage_metadata."""
    capture = CostCapture()
    model = StubChatModel().with_config(callbacks=[capture])
    result = model.invoke([("human", "hi")])
    assert isinstance(result, AIMessage)
    assert capture.input_tokens == 123
    assert capture.output_tokens == 45
    assert capture.thinking_tokens == 0
    assert "stub/model-v1" in capture._model_names


def test_costcapture_reads_reasoning_from_output_token_details():
    """OpenAI o1/o3 + OpenRouter reasoning wrappers put reasoning tokens
    under `output_token_details.reasoning` — this must be picked up."""
    capture = CostCapture()
    model = StubChatModel(reasoning_tokens=200).with_config(callbacks=[capture])
    model.invoke([("human", "hi")])
    assert capture.thinking_tokens == 200


@pytest.mark.asyncio
async def test_costcapture_async_path():
    capture = CostCapture()
    model = StubChatModel().with_config(callbacks=[capture])
    await model.ainvoke([("human", "hi")])
    assert capture.input_tokens == 123
    assert capture.output_tokens == 45


def test_costcapture_dedup_across_hooks_by_run_id():
    """If both on_chat_model_end and on_llm_end fire for the same run_id,
    tokens must be counted once (not doubled)."""
    from langchain_core.outputs import LLMResult

    capture = CostCapture()
    run_id = uuid4()
    fake = LLMResult(
        generations=[
            [
                ChatGeneration(
                    message=AIMessage(
                        content="x",
                        usage_metadata={
                            "input_tokens": 10,
                            "output_tokens": 5,
                            "total_tokens": 15,
                        },
                    )
                )
            ]
        ]
    )
    capture.on_chat_model_end(fake, run_id=run_id)
    capture.on_llm_end(fake, run_id=run_id)  # same run_id — should be ignored
    assert capture.input_tokens == 10
    assert capture.output_tokens == 5


# --------------------------------------------------------------------------- #
# 2. LangGraph: CostCapture inside a compiled StateGraph node
# --------------------------------------------------------------------------- #


class _State(dict):
    """Loose state — a plain dict subclass so LangGraph accepts any keys."""


@pytest.mark.asyncio
async def test_costcapture_propagates_through_compiled_langgraph_node():
    """The critical LangGraph integration: build a compiled StateGraph with a
    decorated node whose pool contains a real BaseChatModel. When the graph is
    invoked, CostCapture must fire — proving that .with_config(callbacks=[...])
    propagates into the LangChain call dispatch under LangGraph's execution."""
    _reset_sidecar()
    captured_exec = {}

    # Patch the client so we don't need a real server here — we only care that
    # CostCapture runs and its numbers show up on the ExecutionResult sent to
    # the server.
    with patch("agensflow_langgraph.decorator.get_client") as mock_client:
        client = MagicMock()
        decision_id = uuid4()
        client.a_select = AsyncMock(
            return_value=RoutingResponse(
                decision_id=decision_id,
                action="only",
                policy_snapshot={"n_visits": 0.0, "reward_mean": 0.0},
                ucb_scores={"only": "inf"},
                reasoning="test",
                idempotency_hit=False,
            )
        )

        def _cap_exec(exec_result):
            captured_exec["exec"] = exec_result
            return ExecutionAck(decision_id=exec_result.decision_id, status="executed")

        client.a_record_execution = AsyncMock(side_effect=_cap_exec)
        mock_client.return_value = client

        pool = {"only": StubChatModel()}

        @agensflow(pool=pool)
        async def worker(state, model, config=None):
            r = await model.ainvoke([("human", "hi")])
            return {"messages": [{"role": "assistant", "content": r.content}]}

        graph = StateGraph(dict)
        graph.add_node("worker", worker)
        graph.add_edge(START, "worker")
        graph.add_edge("worker", END)
        compiled = graph.compile()

        await compiled.ainvoke(
            {"messages": []},
            config={"configurable": {"thread_id": "t"}},
        )

    exec_result = captured_exec["exec"]
    assert exec_result.input_tokens == 123, (
        f"CostCapture didn't fire under LangGraph's execution "
        f"(got input_tokens={exec_result.input_tokens}). "
        "The .with_config(callbacks=[capture]) chain likely doesn't propagate."
    )
    assert exec_result.output_tokens == 45
