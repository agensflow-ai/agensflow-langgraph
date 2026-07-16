"""Signature derivation: canonicalization + fallback ladder."""

from __future__ import annotations

from agensflow_langgraph.signature import (
    canonicalize_ns,
    compute_idempotency_key,
    derive_signature,
)


def test_canonicalize_strips_uuid_segments():
    assert (
        canonicalize_ns("parent:child:12345678-1234-1234-1234-123456789012")
        == "parent:child"
    )


def test_canonicalize_strips_hex32():
    assert canonicalize_ns("worker:a1b2c3d4e5f6a1b2c3d4e5f6a1b2c3d4") == "worker"


def test_canonicalize_none_stays_none():
    assert canonicalize_ns(None) is None
    assert canonicalize_ns("") is None


def test_canonicalize_leaves_plain_names():
    assert canonicalize_ns("supervisor:worker") == "supervisor:worker"


def test_derive_uses_explicit_signature_first():
    sig = derive_signature(config={"metadata": {"langgraph_node": "foo"}},
                           explicit_signature="my_sig")
    assert sig == "my_sig"


def test_derive_uses_configurable_over_metadata():
    sig = derive_signature(
        config={
            "metadata": {"langgraph_node": "foo"},
            "configurable": {"agensflow_signature": "from_configurable"},
        }
    )
    assert sig == "from_configurable"


def test_derive_uses_node_name_arg_over_metadata():
    sig = derive_signature(
        config={"metadata": {"langgraph_node": "meta_node"}},
        explicit_node_name="explicit_node",
    )
    assert sig == "explicit_node"


def test_derive_uses_langgraph_node_metadata():
    sig = derive_signature(config={"metadata": {"langgraph_node": "my_node"}})
    assert sig == "my_node"


def test_derive_falls_back_to_fn_name():
    def my_node_fn(state):
        return {}

    sig = derive_signature(config=None, fn=my_node_fn)
    assert sig == "my_node_fn"


def test_derive_prepends_canonical_ns():
    sig = derive_signature(
        config={
            "metadata": {
                "langgraph_node": "worker",
                "checkpoint_ns": "supervisor:worker:12345678-1234-1234-1234-123456789012",
            }
        }
    )
    assert sig == "supervisor:worker:worker"


def test_derive_default_when_nothing_available():
    assert derive_signature(config=None) == "default"


def test_idempotency_key_stable():
    k1 = compute_idempotency_key("sig", "thread_x", 3, ["a", "b"])
    k2 = compute_idempotency_key("sig", "thread_x", 3, ["b", "a"])  # order-independent
    assert k1 == k2
    assert len(k1) == 32


def test_idempotency_key_differs_by_signature():
    k1 = compute_idempotency_key("sig_a", "t", 0, ["a"])
    k2 = compute_idempotency_key("sig_b", "t", 0, ["a"])
    assert k1 != k2
