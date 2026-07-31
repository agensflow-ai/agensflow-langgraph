# Starter policies

Pre-baked bandit-state JSONs for the security-domain MAS. Two variants — pick
one based on whether you're using the same model tiers the substrate was
adapted on.

## When to use which

| File | Priors from | Use when |
|---|---|---|
| **`security_v1.json`** | Paper's converged policy (translated, capped at 20 visits/arm) | You're bringing your own models, or you want to see the substrate adapt from a lightweight nudge |
| **`security_v1_adapted.json`** | Paper priors + 120 real runs on `claude-haiku-4.5` + `claude-sonnet-5` + `thinkingmachines/inkling` (uncapped) | You plan to use those exact three model tiers — reward means are calibrated to what they actually produce |

Both files match the OSS `/langgraph/policy/import` shape (5 signatures × 16
arms). `aimport_policy` accepts either wrapped or bare policy dicts.

## Which notebook demonstrates each

- [`notebooks/quickstart.ipynb`](../../notebooks/quickstart.ipynb) — model-agnostic path: imports `security_v1.json`, shows the substrate adapting from paper priors on your workload.
- [`notebooks/quickstart_adapted.ipynb`](../../notebooks/quickstart_adapted.ipynb) — modern-model path: imports `security_v1_adapted.json`, shows a stronger starting policy tuned for the specific tier bindings the graph uses.

Both notebooks otherwise run the same 6-node parallel_critic security-domain
MAS on the same paper task C2.1 in Section 6.

## How to use

```python
from agensflow_langgraph import aimport_policy

result = await aimport_policy(
    "examples/starter_policies/security_v1.json",   # or _adapted
    server_url="http://localhost:8000",              # or your hosted server
    tenant_key="agf_...",
)
print(f"Merged {result['signatures_merged']} sigs, {result['actions_merged']} arms")
```

After import, your bandit's UCB1 sees pre-populated `(visits, reward_mean,
reward_m2)` on the matched signatures and skips forced-exploration on those
arms. New signatures + new arms still cold-start normally.

## Regenerating

```bash
# security_v1.json — from the paper's pkl (zero LLM cost)
python -m examples.security_domain.translate_paper_pkl \
    --max-visits 20 \
    --output examples/starter_policies/security_v1.json

# security_v1_adapted.json — run 60 tasks × 2 epochs on real models
# (~$8-12 in OpenRouter fees, ~4-5 hrs), then extract policy_after
python -m examples.security_domain.verify_warmstart --tasks all --epochs 2
# Extractor script pulls policy_after from the resulting report JSON;
# see examples/security_domain/reports/README.md for the recipe.
```

## Also see

- **`research/`** — the paper's original bandit checkpoints (e07, e09) as
  pickle files. Different substrate shape (12-tuple trajectory-aware
  signatures), NOT importable via `/langgraph/policy/import`. Ship for
  paper reproducibility only. See `research/README.md`.

## Format

Policy JSONs are the wrapped shape produced by `agensflow_langgraph.aexport_policy`:

```json
{
  "contract_version": "v1",
  "policy": {
    "<signature_str>": {
      "<action_key>": {
        "visits": 20,
        "reward_mean": 0.423,
        "reward_m2": 0.4,
        "failure_count": 0
      }
    }
  },
  "n_signatures": 5,
  "n_actions": 16
}
```
