# evidence_heavy_mas — real MAS example

A five-node LangGraph MAS with structured Pydantic handoffs, conditional
verifier-gate routing, InMemorySaver checkpointing, and real LLM calls via
OpenRouter. Every node is decorated with `@agensflow(pool=...)` — the substrate
learns which pool member to route each node to as the demo runs across a
five-task benchmark set.

## What this example proves

The v1 tests used a `MockModel` I wrote — nothing that required real LangChain
callback dispatch, real `AIMessage.usage_metadata`, streaming events, conditional
routing, or checkpointer paths was exercised. This example closes those gaps:

- **Real `ChatOpenAI(base_url="https://openrouter.ai/api/v1", ...)`** for every
  model call. One env var (`OPENROUTER_API_KEY`) unlocks Anthropic / OpenAI /
  Qwen through the same client class.
- **Structured output** via `.with_structured_output(PydanticClass)` before
  `@agensflow` wraps the model. The decorator handles a bound `Runnable`, not
  just a raw ChatModel.
- **Real `usage_metadata` capture** — `CostCapture` reads
  `AIMessage.usage_metadata` (LangChain 0.3+ canonical shape), including
  `output_token_details.reasoning` for OpenRouter reasoning-model wrappers.
- **Conditional edges** with a verifier gate — verdict `unsupported` sends the
  graph back to `solver` (up to `MAX_REVISIONS=2` times), else it flows to
  `evaluator`. Exercises `add_conditional_edges` around a decorated node.
- **`InMemorySaver` checkpointer** — verifies `configurable.thread_id` threads
  through with a checkpointer installed and that our sidecar keys correctly.
- **Streaming** via `graph.astream()` — `--stream` flag switches to the
  streaming path, verifying our decorator fires once per node execution (not
  once per token).

## Prerequisites

1. `OPENROUTER_API_KEY` — for the model calls
2. A running AgensFlow policy server (`agensflow-mcp` — see its README)
3. An `AGENSFLOW_API_KEY` (issued by the server via `POST /auth/anonymous`)

```bash
# Terminal 1 — start the policy server
cd ../../../agensflow-mcp
source .venv/bin/activate
uvicorn agensflow_mcp.app:app --port 8000

# Terminal 2 — issue a key + run the demo
cd ../agensflow-langgraph
source .venv/bin/activate

export OPENROUTER_API_KEY=sk-or-...
export AGENSFLOW_SERVER_URL=http://localhost:8000
export AGENSFLOW_API_KEY=$(curl -sX POST http://localhost:8000/auth/anonymous \
    | python -c "import sys,json;print(json.load(sys.stdin)['api_key'])")

python -m examples.evidence_heavy_mas.run --tasks 5 --runs 3
```

## Cost caveats

At OpenRouter's rates (as of 2026-07):
- One graph invocation ≈ $0.05–$0.30 (varies by which arms UCB1 picks; deep
  runs of Claude are the priciest).
- Full `--tasks 5 --runs 3` = 15 graph invocations ≈ **$2–$5**.
- Add `--runs 6` to see clearer convergence: ~30 invocations ≈ **$5–$10**.

## What the substrate should learn

The task set spans three difficulty tiers with hand-tuned rubrics:

| Task | Difficulty | Expected argmax on `solver` |
|---|---|---|
| easy-tcp-header-size | easy | `cheap` |
| easy-udp-use-cases | easy | `cheap` |
| medium-dns-fallback | medium | `balanced` |
| medium-tls-layers | medium | `balanced` |
| hard-http2-tls-alpn | hard | `deep` |

Because our v1 substrate learns per-signature (not per-task-instance), the
`solver` node has ONE signature that averages across all 5 task types. So the
substrate's argmax on `solver` will reflect the *average* difficulty of the
benchmark set — not necessarily `deep` unless the harder tasks dominate the
score gradient. Read the "convergence" numbers accordingly.

To route per-task-difficulty, pass an explicit `signature=` per invocation via
`configurable.agensflow_signature` — but that's out of scope for this demo.

## Interpreting the output

Per-run print:
```
[  1/15] easy-tcp-header-size         diff=easy   q=1.00  lat=8.4s  revs=0
        planner    → fast
        memory     → cheap
        solver     → balanced
        verifier   → cheap
        evaluator  → default
```

Summary:
```
  ══════ Summary ══════
    planner    argmax=fast    (67%, n=15)   deep=5
    memory     argmax=cheap   (73%, n=15)   solid=4
    solver     argmax=cheap   (60%, n=15)   balanced=4, deep=2
    verifier   argmax=cheap   (80%, n=15)   solid=3
    evaluator  argmax=default (100%, n=15)

    quality[easy]   mean=0.85  (n=6)
    quality[medium] mean=0.70  (n=6)
    quality[hard]   mean=0.55  (n=3)
```

At n=15 the substrate is still exploring — meaningful convergence typically
needs 30–60 total runs (`--runs 8` or so).

## Export the learned policy

```bash
python -m examples.evidence_heavy_mas.run --tasks 5 --runs 6 \
    --export-policy examples/evidence_heavy_mas/policy_export/mas_v1.json
```

The exported JSON can be re-imported into another tenant via
`agensflow_langgraph.import_policy(...)` to seed a warm-start.
