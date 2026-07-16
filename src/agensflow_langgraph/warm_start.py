"""Policy warm-start — import an exported policy JSON into the current tenant.

Loads a policy from a local path or URL and POSTs it to `/langgraph/policy/import`.
The server Welford-merges the imported stats into the tenant's existing bandit state.

Curated policy files (e.g. from AgensFlow MAS work on security + distributed-systems
tasks) ship SEPARATELY from this package — download and import as opt-in.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from urllib.parse import urlparse

from agensflow_langgraph.client import get_client
from agensflow_langgraph.contracts import PolicyImportRequest


def import_policy(
    path_or_url: str | Path,
    *,
    server_url: str | None = None,
    tenant_key: str | None = None,
) -> dict:
    """Read a policy JSON from disk or URL and POST it to the policy server.

    Accepts EITHER:
      * raw `{signature_str: {action_key: stats}}` (older format), or
      * the wrapped shape `{contract_version, policy, ...}` that `export_policy`
        writes.

    Returns a dict with `signatures_merged` + `actions_merged` counts.
    """
    raw = _read_source(str(path_or_url))
    parsed = json.loads(raw)
    if not isinstance(parsed, dict):
        raise ValueError("policy JSON must be a top-level object")

    # Unwrap the export payload if that's what we were given.
    if "policy" in parsed and isinstance(parsed.get("policy"), dict):
        policy = parsed["policy"]
    else:
        policy = parsed
    source_hash = hashlib.sha256(raw.encode()).hexdigest()[:32]

    client = get_client(server_url, tenant_key)
    resp = client.import_policy(
        PolicyImportRequest(policy=policy, source_hash=source_hash)
    )
    return {
        "signatures_merged": resp.signatures_merged,
        "actions_merged": resp.actions_merged,
        "source_hash": source_hash,
    }


def export_policy(
    out_path: str | Path,
    *,
    server_url: str | None = None,
    tenant_key: str | None = None,
) -> dict:
    """Fetch the tenant's current policy from the server and save it as JSON.

    Returns a summary dict {n_signatures, n_actions, path}. The JSON file
    written is directly re-importable via `import_policy(out_path)`.
    """
    client = get_client(server_url, tenant_key)
    resp = client.export_policy()
    payload = {
        "contract_version": "v1",
        "policy": resp.policy,
        "n_signatures": resp.n_signatures,
        "n_actions": resp.n_actions,
    }
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    # Store the WRAPPED payload so downstream tooling can also see the counts;
    # `import_policy` accepts either the wrapped shape or the raw policy dict.
    out.write_text(json.dumps(payload, indent=2, default=str))
    return {
        "n_signatures": resp.n_signatures,
        "n_actions": resp.n_actions,
        "path": str(out),
    }


def _read_source(loc: str) -> str:
    parsed = urlparse(loc)
    if parsed.scheme in ("http", "https"):
        import httpx

        with httpx.Client(timeout=30.0) as c:
            r = c.get(loc)
            r.raise_for_status()
            return r.text
    return Path(loc).read_text()
