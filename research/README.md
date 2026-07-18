# research/ — paper artifacts (reproducibility only)

This folder ships the converged bandit-policy checkpoints from the AgensFlow
paper's experiments. **They are here for citation and reproducibility of the
paper's numbers, NOT as warm-start files for this OSS adapter.**

## What's in each subfolder

- **`e07_skill_variants/policy_graph.pkl`** — converged policy from the
  paper's skill-variant experiments over the distributed-systems papers
  corpus.
- **`e09_cross_domain_security/policy_graph.pkl`** — converged policy from
  the paper's cross-domain-transfer experiments over the security-advisory
  corpus (the same corpus you'll find ported in
  `examples/security_domain/corpus.py`).
- Each also carries the original `RESULTS.md` from that experiment.

## Why these DON'T warm-start the OSS adapter

The paper's substrate and the OSS `agensflow-langgraph` adapter use
**different signature schemas**:

|  | Paper substrate | OSS adapter |
|---|---|---|
| Signature | 12-tuple: `(regime, activation_flags[7], calibration_params[4])` | Flat string: `'planner'` (just the node name) |
| Actions | Stage names per node | Pool keys per node (`cheap`/`balanced`/`deep`) |
| Policy shape | Trajectory-aware (outgoing edges) | Flat per-node UCB1 |
| Regime detection | Built-in (evidence_heavy/straightforward/ambiguous) | Not modeled |

The OSS adapter traded paper-fidelity for one-decorator-per-node integration.
As a result, the paper pkls above **cannot** be `pip install`ed and imported
via `POST /langgraph/policy/import` — the schemas don't match. Trying to
convert would silently lose the regime and trajectory information.

## What DOES warm-start the OSS adapter

See `examples/starter_policies/security_v1.json` — the converged policy
produced by running the paper's 60-task security suite (from
`examples/security_domain/tasks.py`) through the OSS adapter's flat-UCB
substrate. That's the file `notebooks/quickstart.ipynb` imports.

## Loading the paper pkls

The pkls are plain Python `dict[tuple, GraphNode]` — they can be inspected
with the paper's `agensflow` package on PyPI (release TBD) or from source:

```python
import pickle
with open("research/e09_cross_domain_security/policy_graph.pkl", "rb") as f:
    policy = pickle.load(f)
# policy is dict[signature_tuple → GraphNode]
```

The `GraphNode` class definition lives in the paper's `agensflow.learning.
policy_graph.core` module.

## Citation

If you use these checkpoints in academic work, please cite the AgensFlow paper
(link forthcoming).
