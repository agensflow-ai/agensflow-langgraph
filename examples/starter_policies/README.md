# Starter policies

Pre-converged bandit-state JSONs you can import into a fresh AgensFlow tenant
to skip the cold-start exploration phase. Downloadable warm-starts, one per
MAS shape.

## How to use

```python
from agensflow_langgraph import import_policy

result = import_policy(
    "examples/starter_policies/parallel_critic_v1.json",
    server_url="http://localhost:8000",   # or your hosted server
    tenant_key="agf_...",
)
print(f"Merged {result['signatures_merged']} sigs, {result['actions_merged']} arms")
```

After import, your bandit's UCB1 sees pre-populated `(visits, reward_mean,
reward_m2)` on the matched signatures and skips forced-exploration on those
arms. New signatures + new arms still cold-start normally.

## Included policies

### `parallel_critic_v1.json`

Converged from 40 iterations of the
[`parallel_critic_mas`](../parallel_critic_mas/) example against real OpenRouter.
6 signatures × 12 total arms.

| Signature | Arms | Notes |
|---|---|---|
| planner | fast, deep | ~21 visits each |
| memory | cheap, solid | ~21 visits each |
| solver | cheap, balanced, deep | ~14 visits each |
| critic | cheap, solid | ~21 visits each |
| verifier | cheap, solid | ~21 visits each |
| evaluator | default | 42 visits |

**Use when:** your MAS follows a similar shape — planner → memory → solver →
[critic || verifier] → evaluator with per-node pools named `fast`/`deep` (or
`cheap`/`balanced`/`deep` for the solver). Signatures auto-derive from the
LangGraph node names, so as long as your graph uses these five node names, the
import will match.

**Honesty caveat:** the taskflow used to converge this policy is easy enough
that all arms scored ~1.0 quality — so the substrate learned the arms are
*equivalent* for this task class rather than that one arm is *best*. That's a
legitimate outcome ("no reward gradient exists here"), and it still saves the
cold-start exploration cost when you plug in your own harder tasks. Once real
task-difficulty variance appears in your workload, the substrate will begin
differentiating from these warm priors.

## Coming later

- `evidence_heavy_v1.json` — a converged policy for the linear
  planner → memory → solver → verifier → evaluator MAS with revision loop.
- Translated legacy AgensFlow policies (e09 cross-domain security, e05
  topology skip) after we solve the schema-projection question — the old
  substrate used belief-vector signatures and skill-variant actions that
  don't map 1:1 to our simpler `(signature_str, action_str)` shape.

## Format

Policy JSONs are the wrapped shape produced by `agensflow_langgraph.export_policy`:

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

`import_policy` accepts either the wrapped shape or the bare `policy` dict — so
you can also hand-craft a starter without going through the export helper.
