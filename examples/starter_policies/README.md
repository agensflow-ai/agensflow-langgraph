# Starter policies

Pre-converged bandit-state JSONs you can import into a fresh AgensFlow tenant
to skip the cold-start exploration phase. Downloadable warm-starts, one per
domain × MAS-shape combination.

## How to use

```python
from agensflow_langgraph import aimport_policy

result = await aimport_policy(
    "examples/starter_policies/security_v1.json",
    server_url="http://localhost:8000",   # or your hosted server
    tenant_key="agf_...",
)
print(f"Merged {result['signatures_merged']} sigs, {result['actions_merged']} arms")
```

After import, your bandit's UCB1 sees pre-populated `(visits, reward_mean,
reward_m2)` on the matched signatures and skips forced-exploration on those
arms. New signatures + new arms still cold-start normally.

## Included policies

### `security_v1.json`

Converged from running the paper's 60-task security-advisory suite (from
[`examples/security_domain/tasks.py`](../security_domain/tasks.py)) through
the [`security_domain`](../security_domain/) MAS graph — same 6-node
`parallel_critic` topology as [`parallel_critic_mas`](../parallel_critic_mas/),
scored per-run by the free-tier 3-panel judge (Google + Qwen + xAI).

6 signatures × 12 total arms. Regenerate with:

```bash
python -m examples.security_domain.converge --epochs 6 --output security_v1.json
```

**Use when:** your MAS follows the same shape — planner → memory → solver →
[critic || verifier] → evaluator with per-node pools named
`fast`/`deep` (planner), `cheap`/`solid` (memory, critic, verifier),
`cheap`/`balanced`/`deep` (solver), `default` (evaluator). Signatures
auto-derive from LangGraph node names, so as long as your graph uses these
five node names + these pool keys, the import will match.

The notebook `notebooks/quickstart.ipynb` imports this file directly.

## Also see

- **`research/`** — the paper's original bandit checkpoints (e07, e09) as
  pickle files. Those are for reproducing the paper — they use a different
  substrate (trajectory-aware, richer signatures) and CAN'T be imported into
  this OSS adapter's flat-UCB. See `research/README.md`.

## Coming later

- `distributed_v1.json` — a converged policy for the paper's distributed-
  systems task suite (`e07_skill_variants`, ported through the OSS adapter).

## Format

Policy JSONs are the wrapped shape produced by `agensflow_langgraph.aexport_policy`:

```json
{
  "contract_version": "v1",
  "policy": {
    "<signature_str>": {
      "<action_key>": {
        "visits": 21,
        "reward_mean": 0.998,
        "reward_m2": 0.001,
        "failure_count": 0
      }
    }
  },
  "n_signatures": 6,
  "n_actions": 12
}
```

`aimport_policy` accepts either the wrapped shape or the bare `policy` dict —
so you can also hand-craft a starter without going through the export helper.
