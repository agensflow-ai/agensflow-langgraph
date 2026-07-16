# parallel_critic_mas

MAS with **critic + verifier running in parallel** after the solver, then an
evaluator that merges both signals. Topologically distinct from
[`evidence_heavy_mas`](../evidence_heavy_mas/) (which is linear with a
revision loop).

```
     START
       │
       ▼
   planner → memory → solver
                        │
             ┌──────────┴──────────┐   ← fan-out (LangGraph runs concurrently)
             ▼                     ▼
         critic                verifier
             │                     │
             └──────────┬──────────┘   ← fan-in
                        ▼
                    evaluator
                        │
                        ▼
                       END
```

## What this example proves (that evidence_heavy doesn't)

- **Parallel node execution.** `solver` fans out to `critic` and `verifier`;
  LangGraph runs both concurrently. `@agensflow` fires exactly once per node
  even when siblings are running.
- **Fan-in with reducer-based state merging.** The `trace` state field uses
  `Annotated[list, operator.add]` because critic + verifier both append to it
  in the same graph step — without a reducer, LangGraph raises
  `InvalidUpdateError`.
- **Two orthogonal quality signals.** Critic judges reasoning *soundness*
  (independent of evidence); verifier judges *evidence grounding*. Evaluator
  merges both.

## Run it

```bash
export OPENROUTER_API_KEY=sk-or-...
export AGENSFLOW_SERVER_URL=http://localhost:8000
export AGENSFLOW_API_KEY=agf_...

python -m examples.parallel_critic_mas.run --tasks 5 --runs 4
```

## Export a policy for the starter bundle

```bash
python -m examples.parallel_critic_mas.run --tasks 5 --runs 8 \
    --export-policy examples/starter_policies/parallel_critic_v1.json
```

The exported JSON is directly importable via
`agensflow_langgraph.import_policy` — see
[`examples/starter_policies/README.md`](../starter_policies/README.md).

## Cost

- One graph invocation: ~$0.15–$0.40 (six model calls: planner, memory,
  solver, critic, verifier, evaluator)
- `--tasks 5 --runs 8` = 40 invocations ≈ **$6–$16**

## Layout

- `graph.py` — StateGraph + 6 decorated nodes
- `prompts.py` — Pydantic schemas + per-agent system prompts
- `documents.py` — small in-code document corpus (TCP/UDP/DNS/TLS/HTTP2)
- `tasks.py` — 5-task benchmark with rubric-based scoring
- `run.py` — CLI entrypoint with per-run trace + summary + policy export
