# Deterministic-mock example

Shows the full AgensFlow substrate loop — signature derivation, action selection,
execution recording, explicit reward — without any LLM calls, API keys, or judge
setup. Mock models return canned responses at known cost; a synthetic quality
landscape defines each `(node, action)` pair's ground-truth reward.

## What this demo proves

Three nodes with a **different** optimal action per node:

| Node                  | Optimal action | Why                                    |
|-----------------------|----------------|-----------------------------------------|
| `classify_intent`     | `cheap`        | simple pattern-match — cheapest suffices |
| `retrieve_context`    | `balanced`     | needs some reasoning; deep is overkill  |
| `synthesize_response` | `deep`         | genuine reasoning task — deep pays off  |

The substrate should learn to prefer the optimal action per node without ever
seeing the ground-truth table above — it learns purely from the synthetic quality
scores submitted after each run.

**Convergence timing:** `synthesize_response` typically converges within 15-20
iterations because its quality gap (0.95 vs 0.20) is wide. `classify_intent` and
`retrieve_context` have narrower gaps (0.10 spread) and take 40-60+ iterations —
UCB1 keeps exploring while it's uncertain. This is correct bandit behavior, not
a bug: the substrate is asking "how much does this cheaper arm really cost me?"

## Run it

**Terminal 1** — start the AgensFlow policy server:

```bash
cd ../../agensflow-mcp
source .venv/bin/activate
uvicorn agensflow_mcp.app:app --port 8000
```

**Terminal 2** — issue a key and run the demo:

```bash
cd agensflow-langgraph
source .venv/bin/activate

export AGENSFLOW_SERVER_URL=http://localhost:8000
export AGENSFLOW_API_KEY=$(curl -sX POST http://localhost:8000/auth/anonymous \
    | python -c "import sys,json;print(json.load(sys.stdin)['api_key'])")

python examples/deterministic_mock/graph.py --runs 20
```

## Expected output

```
  Running 20 iterations against http://localhost:8000
  Expected converged routing:
    classify_intent        → cheap (quality 0.95)
    retrieve_context       → balanced (quality 0.9)
    synthesize_response    → deep (quality 0.95)

  [ 1/20]  classify_intent=cheap  retrieve_context=cheap  synthesize_response=cheap
  [ 2/20]  classify_intent=balanced  ...
  ...
  [20/20]  classify_intent=cheap  retrieve_context=balanced  synthesize_response=deep

  Last 5 iterations (convergence check):
    ✓ classify_intent        ['cheap', 'cheap', 'cheap', 'cheap', 'cheap']  (expected cheap)
    ✓ retrieve_context       ['balanced', 'balanced', ...]                  (expected balanced)
    ✓ synthesize_response    ['deep', 'deep', 'deep', 'deep', 'deep']       (expected deep)
```

## What this demo does NOT show

- Real LLM cost/token capture (mock models don't emit `usage_metadata`)
- OpenRouter judge fallback (uses explicit `record_reward` instead)
- Long-tail behavior at N > 100 iterations
- Cross-provider routing (all pool entries are the same `MockModel` class)

For a fuller demo, see the (v1.1) `examples/mixed_complexity_graph/` — real
LangChain models, OpenRouter judge, cost capture.
