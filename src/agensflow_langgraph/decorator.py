"""The `@agensflow` decorator — the entire public shape of the library.

Wraps any LangGraph node function `(state, config?) -> dict` into a substrate-routed
node. Detects sync vs async and preserves the shape. Forwards `config` to user code
if the wrapped function accepts it.

Fail-open by default: if the policy server is unreachable, the graph proceeds with
`fallback_action` (or the first pool key) and skips recording. Set `fail_closed=True`
to raise instead.

Judge is OFF by default. When enabled (via `judge=...` kwarg), quality is submitted
asynchronously after the node returns; the graph never blocks on judge latency.
"""

from __future__ import annotations

import asyncio
import functools
import inspect
import time
from collections.abc import Callable
from typing import Any

from agensflow_langgraph.callbacks import CostCapture
from agensflow_langgraph.client import get_client
from agensflow_langgraph.contracts import (
    ExecutionResult,
    NodeContext,
    RoutingRequest,
)
from agensflow_langgraph.errors import InvalidPool, ServerUnreachable
from agensflow_langgraph.judge import resolve_judge, submit_judge_reward
from agensflow_langgraph.sidecar import _get_sidecar
from agensflow_langgraph.signature import (
    _maybe_warn_sentinel,
    canonicalize_ns,
    compute_idempotency_key,
    derive_signature,
    resolve_discriminator,
)


def agensflow(
    pool: dict[str, Any],
    *,
    signature: str | None = None,
    node_name: str | None = None,
    fallback_action: str | None = None,
    fail_closed: bool = False,
    judge: str | Callable[..., Any] | bool | None = False,
    capture_messages: bool = False,
    redact: Callable[[Any], Any] | None = None,
    server_url: str | None = None,
    tenant_key: str | None = None,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Wrap a LangGraph node function so its LLM calls route through the AgensFlow substrate.

    Parameters:
        pool: {opaque_key: Runnable} — the models to route among. Every value must be
            a LangChain Runnable (or anything with `.with_config(...)` + `.invoke()`).
        signature: Explicit signature override. Bypasses the auto-derivation ladder.
        node_name: Explicit node name — takes priority over the metadata-derived name.
        fallback_action: Which pool key to use if the policy server is unreachable
            (fail-open). Defaults to `next(iter(pool))`. Set `fail_closed=True` to raise
            instead.
        fail_closed: If True, raise ServerUnreachable when the server is down. Default False.
        judge: str model spec ("openrouter:openai/gpt-4o-mini"), callable, True (use env
            default), False or None (disabled). Default is False.
        capture_messages: If True, input_messages + output_message are sent to the server
            for optional replay (server writes them only if tenant has opted in).
        redact: Optional callable applied to messages before they leave the process (for
            judge scoring or replay). Signature: `(messages) -> messages`.
        server_url / tenant_key: Overrides for env vars AGENSFLOW_SERVER_URL / AGENSFLOW_API_KEY.

    The decorated function's signature:
        def my_node(state, model, config=None): ...     # sync
        async def my_node(state, model, config=None): ...  # async
    The `model` parameter is injected — user code just calls `.invoke()` or `.ainvoke()` on it.
    """
    if not pool:
        raise InvalidPool("pool must have at least one key")

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        is_async = inspect.iscoroutinefunction(fn)
        sig_inspect = inspect.signature(fn)
        forwards_config = "config" in sig_inspect.parameters

        def _build_context(state: Any, config: dict | None):
            sig = derive_signature(
                config,
                fn=fn,
                explicit_node_name=node_name,
                explicit_signature=signature,
            )
            meta = (config or {}).get("metadata") or {}
            conf = (config or {}).get("configurable") or {}
            thread_id = conf.get("thread_id")
            step = meta.get("langgraph_step")
            pool_keys = list(pool)
            # Idempotency discriminator: prefer thread_id → run_id → sentinel.
            # Falling back to run_id keeps invocations distinct when the user
            # didn't set thread_id (common for one-shot graph.ainvoke calls).
            discriminator, source = resolve_discriminator(config)
            _maybe_warn_sentinel(source)
            idem = compute_idempotency_key(sig, discriminator, step, pool_keys)
            # The recorded thread_id we send to the server is the actual
            # configurable.thread_id when present (so dashboards can join on it);
            # if we're using run_id as discriminator, we still surface it in the
            # thread_id field so the audit trail shows what we used.
            recorded_thread_id = thread_id or (
                discriminator if source == "run_id" else None
            )
            request = RoutingRequest(
                context=NodeContext(
                    signature=sig,
                    langgraph_node=meta.get("langgraph_node"),
                    checkpoint_ns_canonical=canonicalize_ns(meta.get("checkpoint_ns")),
                    thread_id=recorded_thread_id,
                    step=step,
                ),
                action_pool_keys=pool_keys,
                idempotency_key=idem,
            )
            return sig, thread_id, pool_keys, request

        def _bind(action: str, decision_id, sig: str):
            capture = CostCapture()
            bound_model = pool[action].with_config(
                callbacks=[capture],
                metadata={
                    "agensflow_decision_id": str(decision_id) if decision_id else None,
                    "agensflow_action": action,
                    "agensflow_signature": sig,
                },
            )
            return capture, bound_model

        def _prepare(state: Any, config: dict | None):
            sig, thread_id, pool_keys, request = _build_context(state, config)
            client = get_client(server_url, tenant_key)
            action = None
            decision_id = None
            try:
                resp = client.select(request)
                action = resp.action
                decision_id = resp.decision_id
            except ServerUnreachable:
                if fail_closed:
                    raise
                action = fallback_action or pool_keys[0]
            capture, bound_model = _bind(action, decision_id, sig)
            return bound_model, capture, decision_id, action, sig, thread_id

        async def _a_prepare(state: Any, config: dict | None):
            sig, thread_id, pool_keys, request = _build_context(state, config)
            client = get_client(server_url, tenant_key)
            action = None
            decision_id = None
            try:
                resp = await client.a_select(request)
                action = resp.action
                decision_id = resp.decision_id
            except ServerUnreachable:
                if fail_closed:
                    raise
                action = fallback_action or pool_keys[0]
            capture, bound_model = _bind(action, decision_id, sig)
            return bound_model, capture, decision_id, action, sig, thread_id

        def _make_kwargs(config: dict | None) -> dict:
            return {"config": config} if forwards_config else {}

        def _build_exec_result(
            state: Any,
            result: Any,
            capture: CostCapture,
            decision_id,
            action: str,
            config: dict | None,
            started: float,
        ):
            latency = time.monotonic() - started
            input_msgs = None
            output_msg = None
            if capture_messages:
                raw_in = _extract_messages(state)
                raw_out = _extract_output_message(result)
                input_msgs = _apply_redact(raw_in, redact)
                output_msg = _apply_redact(raw_out, redact)
            output_msg_flat = (
                output_msg[0] if isinstance(output_msg, list) and output_msg else output_msg
            )
            run_id = (config or {}).get("run_id")
            exec_result = ExecutionResult(
                decision_id=decision_id,
                action=action,
                cost_usd=capture.cost_usd,
                latency_s=latency,
                input_tokens=capture.input_tokens,
                output_tokens=capture.output_tokens,
                thinking_tokens=capture.thinking_tokens or None,
                input_messages=input_msgs,
                output_message=output_msg_flat,
                langsmith_run_id=str(run_id) if run_id else None,
            )
            return exec_result, input_msgs, output_msg

        def _maybe_schedule_judge(
            decision_id, state: Any, result: Any, input_msgs, output_msg, action: str
        ) -> None:
            judge_cfg = resolve_judge(judge)
            if judge_cfg is None:
                return
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                return
            if not loop.is_running():
                return
            asyncio.ensure_future(
                submit_judge_reward(
                    decision_id=decision_id,
                    input_messages=input_msgs
                    or _apply_redact(_extract_messages(state), redact),
                    output_message=output_msg
                    or _apply_redact(_extract_output_message(result), redact),
                    action=action,
                    judge_cfg=judge_cfg,
                    server_url=server_url,
                    tenant_key=tenant_key,
                )
            )

        def _finish(
            state: Any,
            result: Any,
            capture: CostCapture,
            decision_id,
            action: str,
            thread_id: str | None,
            config: dict | None,
            started: float,
        ) -> None:
            if decision_id is None:
                return
            _get_sidecar().record(thread_id, decision_id)
            exec_result, input_msgs, output_msg = _build_exec_result(
                state, result, capture, decision_id, action, config, started
            )
            get_client(server_url, tenant_key).record_execution(exec_result)
            _maybe_schedule_judge(decision_id, state, result, input_msgs, output_msg, action)

        async def _a_finish(
            state: Any,
            result: Any,
            capture: CostCapture,
            decision_id,
            action: str,
            thread_id: str | None,
            config: dict | None,
            started: float,
        ) -> None:
            if decision_id is None:
                return
            _get_sidecar().record(thread_id, decision_id)
            exec_result, input_msgs, output_msg = _build_exec_result(
                state, result, capture, decision_id, action, config, started
            )
            await get_client(server_url, tenant_key).a_record_execution(exec_result)
            _maybe_schedule_judge(decision_id, state, result, input_msgs, output_msg, action)

        if is_async:

            @functools.wraps(fn)
            async def awrapper(state: Any, config: dict | None = None) -> Any:
                bound_model, capture, decision_id, action, _sig, thread_id = (
                    await _a_prepare(state, config)
                )
                started = time.monotonic()
                kwargs = _make_kwargs(config)
                result = await fn(state, model=bound_model, **kwargs)
                await _a_finish(
                    state, result, capture, decision_id, action, thread_id, config, started
                )
                return result

            return awrapper

        @functools.wraps(fn)
        def wrapper(state: Any, config: dict | None = None) -> Any:
            bound_model, capture, decision_id, action, _sig, thread_id = _prepare(
                state, config
            )
            started = time.monotonic()
            kwargs = _make_kwargs(config)
            result = fn(state, model=bound_model, **kwargs)
            _finish(state, result, capture, decision_id, action, thread_id, config, started)
            return result

        return wrapper

    return decorator


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _extract_messages(state: Any) -> list[dict] | None:
    """Best-effort message extraction from a LangGraph state.

    Handles the common shapes: `state["messages"]`, dict-like state with a
    "messages" key, or an object with a `.messages` attribute. Returns None if
    nothing matches — that's OK; capture_messages just yields None."""
    if isinstance(state, dict):
        msgs = state.get("messages")
        if msgs is None:
            return None
        return [_msg_to_dict(m) for m in msgs] if isinstance(msgs, list) else None
    msgs = getattr(state, "messages", None)
    if isinstance(msgs, list):
        return [_msg_to_dict(m) for m in msgs]
    return None


def _extract_output_message(result: Any) -> Any:
    """Pull the last message out of the node's return value, if any."""
    if isinstance(result, dict):
        msgs = result.get("messages")
        if isinstance(msgs, list) and msgs:
            return _msg_to_dict(msgs[-1])
    return None


def _msg_to_dict(msg: Any) -> dict:
    if isinstance(msg, dict):
        return msg
    # LangChain BaseMessage — best-effort dump
    dump_method = getattr(msg, "model_dump", None) or getattr(msg, "dict", None)
    if callable(dump_method):
        try:
            return dump_method()
        except Exception:
            pass
    return {
        "role": getattr(msg, "type", getattr(msg, "role", "unknown")),
        "content": getattr(msg, "content", str(msg)),
    }


def _apply_redact(payload: Any, redact: Callable[[Any], Any] | None) -> Any:
    if redact is None or payload is None:
        return payload
    try:
        return redact(payload)
    except Exception:
        # If the redaction hook errors, drop the payload rather than leak.
        return None
