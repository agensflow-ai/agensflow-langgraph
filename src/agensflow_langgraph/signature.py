"""Signature derivation — from a LangGraph node's RunnableConfig to a stable string.

Canonicalization is critical: `checkpoint_ns` includes runtime UUIDs for subgraph
invocations, so a naive `f"{ns}:{node}"` would create ONE signature per run instead
of one per NODE — the substrate would learn nothing. We strip UUID/32-hex segments
before use.

Fallback ladder (first match wins):
    1. Explicit `signature=` decorator kwarg
    2. `configurable["agensflow_signature"]` on the config
    3. Explicit `node_name=` decorator kwarg (canonical NS prepended if present)
    4. `metadata["langgraph_node"]` (canonical NS prepended if present)
    5. `fn.__name__` (the decorated function's name — always stable)
    6. `run_name` from the config (non-LangGraph LangChain use)
    7. "default"
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

_UUID_RE = re.compile(
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}", re.I
)
_HEX32_RE = re.compile(r"\b[0-9a-f]{32}\b", re.I)


def canonicalize_ns(ns: str | None) -> str | None:
    """Strip UUID / 32-hex segments so subgraph signatures are stable across runs.

    LangGraph's `checkpoint_ns` for a subgraph looks like `parent:child:<uuid>`,
    where the trailing UUID identifies THAT invocation. We drop it so per-node
    learning aggregates across runs.

    >>> canonicalize_ns("supervisor:worker:12345678-1234-1234-1234-123456789012")
    'supervisor:worker'
    >>> canonicalize_ns("plain")
    'plain'
    >>> canonicalize_ns(None) is None
    True
    >>> canonicalize_ns("")  # empty stays empty (returns None)
    """
    if not ns:
        return None
    ns = _UUID_RE.sub("", ns)
    ns = _HEX32_RE.sub("", ns)
    parts = [p for p in ns.split(":") if p]  # drop empties from stripped segments
    return ":".join(parts) if parts else None


def derive_signature(
    config: dict[str, Any] | None,
    fn: Callable[..., Any] | None = None,
    explicit_node_name: str | None = None,
    explicit_signature: str | None = None,
) -> str:
    """Resolve the signature string for a given node invocation.

    See module docstring for the fallback ladder. Result is a stable string that
    the substrate uses as its bandit key.
    """
    if explicit_signature:
        return explicit_signature

    conf = (config or {}).get("configurable") or {}
    if sig := conf.get("agensflow_signature"):
        return str(sig)

    meta = (config or {}).get("metadata") or {}
    ns = canonicalize_ns(meta.get("checkpoint_ns"))

    if explicit_node_name:
        base = explicit_node_name
    elif node := meta.get("langgraph_node"):
        base = str(node)
    elif fn is not None and getattr(fn, "__name__", None):
        base = fn.__name__
    elif run_name := (config or {}).get("run_name"):
        base = str(run_name)
    else:
        return "default"

    return f"{ns}:{base}" if ns else base


def resolve_discriminator(config: dict | None) -> tuple[str, str]:
    """Choose the best discriminator for the idempotency key.

    Returns (discriminator, source) where source is one of:
      * "thread_id"   — configurable.thread_id was set (best)
      * "run_id"      — LangGraph's per-invocation run_id (good fallback)
      * "sentinel"    — neither was available, use "-" (worst; retries WILL
                        collide, so we warn once per process)
    """
    conf = (config or {}).get("configurable") or {}
    if tid := conf.get("thread_id"):
        return str(tid), "thread_id"
    if rid := (config or {}).get("run_id"):
        # LangGraph populates run_id per node invocation. It's unique enough for
        # idempotency across the natural graph invocations, though it doesn't
        # provide the same "retry hits the same decision" guarantee as thread_id.
        return str(rid), "run_id"
    return "-", "sentinel"


_WARNED_SENTINEL = False


def _maybe_warn_sentinel(source: str) -> None:
    """Emit a one-shot warning if the caller landed on the sentinel."""
    global _WARNED_SENTINEL
    if source == "sentinel" and not _WARNED_SENTINEL:
        import warnings

        warnings.warn(
            "agensflow-langgraph: no thread_id or run_id available on the "
            "RunnableConfig; idempotency-key discriminator falling back to "
            "'-'. Repeated invocations of the same signature/pool will "
            "return the SAME decision_id from the server. Set "
            "configurable.thread_id to enable proper idempotency.",
            stacklevel=3,
        )
        _WARNED_SENTINEL = True


def compute_idempotency_key(
    signature: str,
    thread_id: str | None,
    step: int | None,
    pool_keys: list[str],
) -> str:
    """Stable hash of the invocation coordinates.

    Same (signature, thread_id, step, pool_keys) ⇒ same key ⇒ server-side idempotency
    returns the same decision_id even if the client retries after a timeout.

    Callers should first resolve the discriminator via `resolve_discriminator(config)`
    so that missing `thread_id` falls back to `run_id` instead of colliding on `-`.
    """
    import hashlib

    parts = [
        signature,
        thread_id or "-",
        str(step) if step is not None else "-",
        "|".join(sorted(pool_keys)),
    ]
    payload = "\x1f".join(parts).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:32]
