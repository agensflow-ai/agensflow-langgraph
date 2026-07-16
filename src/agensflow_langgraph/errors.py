"""Adapter-specific exceptions."""

from __future__ import annotations


class AgensFlowError(Exception):
    """Base class for all agensflow-langgraph errors."""


class ServerUnreachable(AgensFlowError):
    """The policy server didn't respond in time or refused the connection.

    Raised only when `fail_closed=True` on the decorator. Under the default
    fail-open behavior, this exception is caught internally and the graph
    proceeds with `fallback_action` (or the first pool key)."""


class ServerRejected(AgensFlowError):
    """The policy server returned a 4xx (bad request, auth failure, etc.).

    Not retried; propagates immediately."""


class InvalidPool(AgensFlowError):
    """The pool passed to `@agensflow(pool=...)` is empty or malformed."""
