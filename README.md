# agensflow-langgraph

Decorator-based routing for LangGraph agents. Learn which model in your declared
pool each node of your graph should use — automatically, per node, from feedback.

<a target="_blank" href="https://colab.research.google.com/github/agensflow-ai/agensflow-langgraph/blob/main/notebooks/quickstart.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

**Status:** v0.1.2 (alpha).

## The free-tier bundle

Everything needed to run an adaptive MAS end-to-end, no hosted service, no
external dependencies:

| Piece | What it is |
|---|---|
| **`agensflow-mcp`** | The policy server. UCB1 bandits, tenant isolation, HTTP + MCP surfaces. Self-hostable (SQLite or Postgres) or hosted. |
| **`agensflow-langgraph`** | This package. The `@agensflow` decorator wraps any LangGraph node and routes its LLM calls through the server. |
| **Starter policy** | [`examples/starter_policies/parallel_critic_v1.json`](examples/starter_policies/README.md) — a converged 40-run bandit state you import to skip cold-start exploration. |
| **Quickstart notebook** | [`notebooks/quickstart.ipynb`](notebooks/quickstart.ipynb) — boots the whole stack in-process, imports the starter, runs a real MAS query. Runs top-to-bottom in ~2 min for ~$0.30 of OpenRouter fees. |

For anyone wanting to just try it: **open the notebook.** It boots the policy
server in-process (no separate `uvicorn`), issues an API key, imports the
starter, builds a real 6-node MAS with parallel critic+verifier, and runs one
query against OpenRouter — all in eight cells.

## Install

```bash
pip install agensflow-langgraph
# For the notebook / examples:
pip install agensflow-mcp langchain-openai jupyter
```

## 60-second example

```python
from langgraph.graph import StateGraph
from langchain_openai import ChatOpenAI
from agensflow_langgraph import agensflow, record_reward

# OpenRouter unifies every provider under one API key — one env var, many models
def or_model(model_id: str):
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=model_id,
    )

# Declare a per-node pool. The substrate learns which arm wins.
@agensflow(pool={
    "cheap":    or_model("openai/gpt-4o-mini"),
    "balanced": or_model("openai/gpt-4o"),
    "deep":     or_model("anthropic/claude-sonnet-4"),
})
async def classify_intent(state, model, config=None):
    return {"messages": [await model.ainvoke(state["messages"])]}

graph = StateGraph(State)
graph.add_node("classify_intent", classify_intent)
# ... rest of your graph unchanged

# Attach an explicit reward at the graph terminal
def terminal(state):
    record_reward(thread_id=state.get("thread_id"), quality=0.85)
    return {...}
```

Every model call inside a decorated node routes through the policy server;
UCB1 over per-node signatures picks the pool arm. Cost + tokens are captured
via LangChain's `AIMessage.usage_metadata` callback path.

## Two topology examples

- [`examples/evidence_heavy_mas/`](examples/evidence_heavy_mas/) — linear
  planner → memory → solver → verifier → evaluator with a verifier-gate
  revision loop. Exercises conditional edges + revision cycles.
- [`examples/parallel_critic_mas/`](examples/parallel_critic_mas/) — solver
  fans out to critic + verifier **in parallel**, evaluator merges both
  signals. Exercises parallel node execution + fan-in state reducers.

Both use real `ChatOpenAI(base_url=openrouter)` calls, structured Pydantic
handoffs via `.with_structured_output(method="function_calling")`, and
InMemorySaver checkpointing.

## Configuration

```bash
export AGENSFLOW_SERVER_URL="https://mcp.agensflow.ai"   # or your self-hosted URL
export AGENSFLOW_API_KEY="agf_..."                        # bearer token
export OPENROUTER_API_KEY="sk-or-..."                     # for the examples
```

Or pass them per-decorator:

```python
@agensflow(pool={...}, server_url="http://localhost:8000", tenant_key="agf_...")
```

## Documentation

- [Integration guide](docs/integration.md) — fail-open behavior, three reward
  paths (explicit / verifier / OpenRouter judge), signature derivation,
  idempotency, warm-start, streaming, checkpointer + thread_id, real-model
  cost capture
- [Starter policies](examples/starter_policies/) — how to import a converged
  policy for warm-start
- Individual example READMEs in `examples/*/README.md`

## License

Apache-2.0
