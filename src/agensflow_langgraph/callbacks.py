"""LangChain callback for provider-agnostic cost + token capture.

Installed via `pool[action].with_config(callbacks=[capture], metadata={...})` so
capture is triggered by every LLM call inside the decorated node — regardless of
provider (OpenAI, Anthropic, Bedrock, Vertex, etc.). Capture happens on the
`on_chat_model_end` hook, which is what modern chat models actually fire; the
older `on_llm_end` hook is also handled for backwards compatibility with
completion-style LLMs.

Cost is read from `AIMessage.usage_metadata` — the canonical LangChain 0.3+
shape — with `output_token_details.reasoning` picked up for OpenAI o1/o3 and
OpenRouter reasoning-model wrappers. Provider-specific pricing is not computed
here; if the model surfaces a `cost` field in llm_output we take it, else the
ledger records tokens only and the substrate's cost weight simply drops out for
that run.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

try:
    from langchain_core.callbacks import BaseCallbackHandler
    from langchain_core.outputs import LLMResult
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "agensflow-langgraph requires langchain-core >= 0.3. "
        "Install via `pip install agensflow-langgraph[dev]`."
    ) from e


class CostCapture(BaseCallbackHandler):
    """One instance per decorated-node invocation. Accumulates across all sub-LLM
    calls within that node (a node may issue multiple model calls, e.g. tool-use
    loops).

    Reads token counts from two places, preferring the newer:
      1. `AIMessage.usage_metadata` (canonical LangChain 0.3+ shape)
      2. `LLMResult.llm_output["token_usage"]` (older completion-style shape)

    Because a single LLM call may fire BOTH `on_chat_model_end` AND `on_llm_end`
    on some backends, we dedupe by `run_id` — the first hook to fire wins for
    that run.
    """

    def __init__(self) -> None:
        super().__init__()
        self.input_tokens: int = 0
        self.output_tokens: int = 0
        self.thinking_tokens: int = 0
        self.cost_usd: float = 0.0
        self._model_names: list[str] = []
        # Dedup so both hooks can't double-count the same run.
        self._seen_run_ids: set[UUID] = set()

    # ------------------------------------------------------------------ #
    # Modern chat-model hook — the load-bearing path for ChatOpenAI etc.
    # ------------------------------------------------------------------ #

    def on_chat_model_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if run_id in self._seen_run_ids:
            return
        self._seen_run_ids.add(run_id)
        self._accumulate(response)

    # ------------------------------------------------------------------ #
    # Legacy completion-LLM hook — kept for backwards compat.
    # ------------------------------------------------------------------ #

    def on_llm_end(
        self,
        response: LLMResult,
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        if run_id in self._seen_run_ids:
            return
        self._seen_run_ids.add(run_id)
        self._accumulate(response)

    # ------------------------------------------------------------------ #
    # Extraction logic — shared between the two hooks.
    # ------------------------------------------------------------------ #

    def _accumulate(self, response: LLMResult) -> None:
        usage = self._extract_usage(response)
        self.input_tokens += int(usage.get("input_tokens", 0))
        self.output_tokens += int(usage.get("output_tokens", 0))
        self.thinking_tokens += int(usage.get("thinking_tokens", 0))
        self.cost_usd += self._extract_cost(response)
        model = self._extract_model_name(response)
        if model and model not in self._model_names:
            self._model_names.append(model)

    @staticmethod
    def _extract_usage(response: LLMResult) -> dict[str, int]:
        # 1) The canonical modern shape: AIMessage.usage_metadata on the message
        #    inside response.generations[0][0].message. This is what ChatOpenAI
        #    (including OpenRouter via base_url override) surfaces.
        msg = _first_ai_message(response)
        if msg is not None:
            um = getattr(msg, "usage_metadata", None) or {}
            if um:
                out_details = um.get("output_token_details") or {}
                return {
                    "input_tokens": int(um.get("input_tokens", 0) or 0),
                    "output_tokens": int(um.get("output_tokens", 0) or 0),
                    "thinking_tokens": int(
                        # OpenAI o1/o3 + OpenRouter reasoning wrappers use
                        # output_token_details.reasoning; some older paths
                        # surface a flat "thinking_tokens" key.
                        out_details.get("reasoning", 0)
                        or um.get("thinking_tokens", 0)
                        or 0
                    ),
                }

        # 2) Legacy completion-style shape: LLMResult.llm_output["token_usage"].
        llm_out = response.llm_output or {}
        token_usage = llm_out.get("token_usage") or llm_out.get("usage") or {}
        return {
            "input_tokens": int(
                token_usage.get("input_tokens")
                or token_usage.get("prompt_tokens")
                or 0
            ),
            "output_tokens": int(
                token_usage.get("output_tokens")
                or token_usage.get("completion_tokens")
                or 0
            ),
            "thinking_tokens": int(
                token_usage.get("thinking_tokens")
                or token_usage.get("reasoning_tokens")
                or 0
            ),
        }

    @staticmethod
    def _extract_cost(response: LLMResult) -> float:
        # Some providers / proxies report cost directly on llm_output.
        llm_out = response.llm_output or {}
        if "cost" in llm_out:
            try:
                return float(llm_out["cost"])
            except (TypeError, ValueError):
                pass
        # Try AIMessage.response_metadata as a secondary source (some LangChain
        # integrations surface cost there instead of llm_output).
        msg = _first_ai_message(response)
        if msg is not None:
            rm = getattr(msg, "response_metadata", None) or {}
            if "cost" in rm:
                try:
                    return float(rm["cost"])
                except (TypeError, ValueError):
                    pass
        return 0.0

    @staticmethod
    def _extract_model_name(response: LLMResult) -> str | None:
        llm_out = response.llm_output or {}
        if name := (llm_out.get("model_name") or llm_out.get("model")):
            return str(name)
        msg = _first_ai_message(response)
        if msg is not None:
            rm = getattr(msg, "response_metadata", None) or {}
            if name := (rm.get("model_name") or rm.get("model")):
                return str(name)
        return None


def _first_ai_message(response: LLMResult) -> Any | None:
    """Pull the first AIMessage out of an LLMResult, defensively.

    LangChain's `LLMResult.generations` is `list[list[Generation]]`; for chat
    models the inner Generation is a `ChatGeneration` whose `.message` is an
    `AIMessage` carrying `usage_metadata`. We poke through defensively so the
    callback stays robust against schema quirks.
    """
    generations = getattr(response, "generations", None) or []
    for prompt_gens in generations:
        for gen in prompt_gens or []:
            if msg := getattr(gen, "message", None):
                return msg
    return None
