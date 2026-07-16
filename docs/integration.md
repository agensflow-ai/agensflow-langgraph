# Integration guide — `agensflow-langgraph`

The `@agensflow` decorator wraps a LangGraph node function into a substrate-routed
node. Your existing graph doesn't change — you swap in `agensflow_route` (or add
`@agensflow`) at the node level. Every model call inside the node routes through
your declared pool via UCB1 over per-node signatures.

## The decorator, top to bottom

```python
from agensflow_langgraph import agensflow

@agensflow(
    pool={"cheap": haiku, "balanced": sonnet, "deep": opus},

    # --- Signature ---
    signature=None,              # explicit signature override (bypasses auto-derive)
    node_name=None,              # explicit node name (bypasses metadata)

    # --- Fail-open ---
    fallback_action=None,        # which pool key to use if server unreachable
    fail_closed=False,           # if True, raise ServerUnreachable instead

    # --- Reward ---
    judge=False,                 # False | True | "openrouter:model" | callable
    capture_messages=False,      # send input/output messages to server (opt-in replay)
    redact=None,                 # (messages) -> messages — applied before judge/replay

    # --- Client ---
    server_url=None,             # else env AGENSFLOW_SERVER_URL
    tenant_key=None,             # else env AGENSFLOW_API_KEY
)
```

**Both sync and async node functions are supported.** The decorator detects
`inspect.iscoroutinefunction(fn)` and returns the matching wrapper. If your fn
accepts a `config` parameter, it's forwarded automatically.

## Fail-open vs. fail-closed

By default the adapter is **fail-open**: if the policy server is unreachable, the
graph proceeds with `fallback_action` (or the first pool key) and skips recording.
This is the right default because a routing sidecar going down shouldn't take out
the user's application.

```python
@agensflow(pool={"cheap": haiku, "smart": opus}, fallback_action="cheap")
def resilient_node(state, model):
    ...
```

For strict environments where you'd rather fail fast than route unadvised:

```python
@agensflow(pool={...}, fail_closed=True)
def strict_node(state, model):
    ...  # raises ServerUnreachable if the policy server times out
```

## Judge configuration — two entry points, one substrate

Judging can run either inside your MAS process (adapter-side) or inside the
policy server (server-side). Same rubric, same OpenRouter models, same substrate
outcome — the difference is WHERE the panel executes:

| Entry point | Who uses it | Judge lives in | Configured via |
|---|---|---|---|
| `POST /langgraph/reward/submit` | LangGraph / CrewAI / future MAS-framework adapters | `agensflow-langgraph`'s `judge.py` (this package) | `@agensflow(judge=...)` decorator kwarg + local `OPENROUTER_API_KEY` env var |
| `POST /judge/relative` + `POST /record` | Claude Code / codex / any client that talks to the MCP server without an adapter | `agensflow-mcp`'s `engine/relative_judge.py` | Server-side BYOK (`POST /auth/byok`) + judge_models per request |

**Same substrate.** Both paths write quality into `compute_reward` and
Welford-update the same bandit stats. The choice is a deployment/topology
decision: are you shipping a framework-adapted MAS (adapter-side), or a client
that hits MCP tools directly (server-side)?

Neither path duplicates the other — they run in different processes. A contract
test (`tests/integration/test_judge_panel_contract.py`) feeds identical rubric
responses to both implementations and asserts they compute the same quality, so
they stay in sync as the code evolves.

## Rewards — three ways to feed quality back

Every learned routing needs a reward signal. Pick one:

### 1. Explicit — you know the answer

```python
from agensflow_langgraph import record_reward

def graph_terminal(state):
    # ... produce final answer ...
    verified_ok = state["tests_passed"]
    record_reward(
        thread_id=state.get("thread_id"),
        quality=1.0 if verified_ok else 0.0,
        axes={"correctness": 1.0 if verified_ok else 0.0},
    )
    return {...}
```

`record_reward` submits to *all* decisions on that thread_id from the sidecar. If
you want per-decision granularity, pass `decision_id=...` explicitly.

### 2. Verifier callback — you have a scoring function

```python
def _my_verifier(input_msgs, output_msg, action) -> float:
    # Your logic — regex, unit test, custom scorer, etc.
    return 0.83

@agensflow(pool={...}, judge=_my_verifier)
def node(state, model): ...
```

### 3. LLM judge via OpenRouter — auto-scored quality

```bash
export OPENROUTER_API_KEY=sk-or-...
```

**Single-model judge** — cheap, fast, one call per decision:

```python
@agensflow(pool={...}, judge="openrouter:openai/gpt-4o-mini")
def node(state, model): ...

# Or, if AGENSFLOW_JUDGE env var is set:
@agensflow(pool={...}, judge=True)
def node(state, model): ...
```

**3-judge cross-family panel** — kills position bias + provides per-axis
disagreement signal for auditability:

```python
@agensflow(pool={...}, judge={"panel": [
    "x-ai/grok-4.3",
    "openai/gpt-5.4-mini",
    "qwen/qwen3.6-flash",
]})
def node(state, model): ...
```

Both judges run asynchronously after the node returns — never blocks the graph.
Both score 4 axes (correctness, completeness, precision, robustness), aggregate
to quality 0-1, submit via `/reward/submit`. The panel form takes ~6× the API
calls of the single-model form (3 judges × 2 orderings) — use it when you need
the cross-family consensus signal, not for every node in every graph.

**Judge is OFF by default.** No env-var auto-on; the opt-in has to be explicit
because judge calls are billed and can leak content.

## Redaction for sensitive workloads

`capture_messages=True` sends the node's input+output messages to the policy
server (for optional dashboard replay when the tenant has opted in). Judge
scoring also sees them. Apply a redaction hook to strip PII / secrets first:

```python
def redact_pii(msgs):
    return [{"role": m["role"], "content": "[REDACTED]"} for m in msgs]

@agensflow(pool={...}, capture_messages=True, redact=redact_pii, judge=True)
def sensitive_node(state, model): ...
```

The redaction hook is applied *before* messages leave your process. If it raises,
the payload is dropped (safe default).

## Signature derivation — the auto-ladder

The signature is the substrate's bandit key. AgensFlow resolves it via this ladder:

```
1. explicit `signature=` decorator kwarg
2. `configurable["agensflow_signature"]` on the config
3. explicit `node_name=` decorator kwarg (with checkpoint_ns prepended)
4. metadata["langgraph_node"] (with checkpoint_ns prepended)
5. fn.__name__
6. run_name from config
7. "default"
```

**`checkpoint_ns` is canonicalized first** — UUID and 32-hex segments are stripped
so subgraph signatures are stable across runs. Otherwise `f"{ns}:{node}"` would
create one signature per invocation and the substrate would learn nothing.

If your graph has natural stages within a signature (e.g., a maker/checker loop
where the maker and checker are the same node type but you want to route each
independently), encode the stage in your signature explicitly:

```python
@agensflow(pool={...}, signature="reviewer:maker")
def maker_node(state, model): ...

@agensflow(pool={...}, signature="reviewer:checker")
def checker_node(state, model): ...
```

Or use `configurable["agensflow_signature"]` at invocation time for per-run
overrides. AgensFlow doesn't impose stage semantics — you own the signature space.

## Idempotency and retries

Every `/policy/select` request carries an `idempotency_key` (hash of signature +
thread_id + step + pool keys). If the client retries after a timeout, the server
returns the same `decision_id` — no double-counted decisions.

Execute + reward endpoints are also idempotent by `decision_id`. A retried
`/reward/submit` overwrites quality but doesn't double-count in the bandit's
Welford update.

## Warm-start with `import_policy`

Load an exported policy JSON into your tenant's substrate state:

```python
from agensflow_langgraph import import_policy

result = import_policy(
    path_or_url="./policies/mas-security-v1.json",
    server_url="https://mcp.agensflow.ai",
    tenant_key="agf_...",
)
print(f"Merged {result['signatures_merged']} sigs, {result['actions_merged']} arms")
```

Policy files ship separately (downloadable from AgensFlow's docs page) — the
package doesn't bundle any priors because they'd be domain-specific.

Server-side merge uses Welford parallel-merge — imported (visits, mean, m2) are
combined with existing stats without loss of variance information.

## What NOT to do

- Don't mutate graph state with decision_ids. Use the sidecar (`record_reward(
  thread_id=...)`) or pass `decision_id=` explicitly. Undeclared state keys break
  LangGraph's `TypedDict`/Pydantic state validation.
- Don't decorate a supervisor node with `@agensflow` and expect it to route the
  whole subgraph. The decorator only wraps model calls *inside* that specific
  node — subgraph children each need their own decorator.
- Don't share a pool dict across processes without explicit intent. The substrate
  learns per-tenant; if you want to A/B two tenants' policies against the same
  pool, use separate tenant_keys.

## Advanced patterns

The [`examples/evidence_heavy_mas/`](../examples/evidence_heavy_mas/) example
exercises the LangGraph features most production MAS deployments use. Each is
worth calling out individually.

### OpenRouter as the unified model provider

One `OPENROUTER_API_KEY` unlocks Anthropic, OpenAI, Qwen, Grok, Gemini, Llama —
every model in the catalog — through `ChatOpenAI` with the base URL overridden:

```python
from langchain_openai import ChatOpenAI

def or_model(model_id: str) -> ChatOpenAI:
    return ChatOpenAI(
        base_url="https://openrouter.ai/api/v1",
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=model_id,
    )

pool = {
    "cheap":    or_model("meta-llama/llama-3.3-70b-instruct"),
    "balanced": or_model("openai/gpt-4o"),
    "deep":     or_model("anthropic/claude-sonnet-4"),
}
```

OpenRouter surfaces `usage_metadata` on the returned `AIMessage` in the canonical
LangChain 0.3+ shape, so AgensFlow's `CostCapture` picks up input/output/reasoning
tokens automatically. Verified end-to-end in
[`tests/integration/test_costcapture_real_langchain.py`](../tests/integration/test_costcapture_real_langchain.py)
and the evidence_heavy example's own real-LLM smoke test.

### Structured output via `.with_structured_output` before `@agensflow`

The decorator wraps *any* `Runnable`, not just `ChatModel`. Bind your Pydantic
output class first, then feed the bound Runnable to the pool:

```python
from pydantic import BaseModel

class PlannerOutput(BaseModel):
    goal: str
    subproblem: str
    constraints: list[str]

def wrap(model_id: str, schema):
    return or_model(model_id).with_structured_output(
        schema, method="function_calling"   # forces clean JSON via tool-use
    )

planner_pool = {
    "fast": wrap("openai/gpt-4o-mini", PlannerOutput),
    "deep": wrap("anthropic/claude-sonnet-4", PlannerOutput),
}
```

**Why `method="function_calling"`:** some models (Anthropic via OpenRouter,
notably) return JSON wrapped in ` ```markdown fences``` ` when left to freely
emit — which breaks strict parsing. The function-calling path forces schema
compliance via tool-use.

### Conditional edges around decorated nodes

`add_conditional_edges` routes based on state; decorated nodes work on both
sides of the branch. The evidence_heavy example gates on the verifier's verdict
and loops back to the solver on `unsupported`:

```python
def verifier_gate(state) -> str:
    verdict = state.get("verifier_verdict", "unsupported")
    revisions = state.get("revision_count", 0)
    if verdict in ("supported", "partial") or revisions >= MAX_REVISIONS:
        return "evaluator"
    return "solver"

graph.add_conditional_edges(
    "verifier",
    verifier_gate,
    {"evaluator": "evaluator", "solver": "bump_revision"},
)
graph.add_edge("bump_revision", "solver")
```

When the loop closes, `solver` is invoked again — the decorator treats each
invocation as a distinct decision (idempotency key includes `langgraph_step`,
which increments on each visit) so the bandit sees separate outcomes for the
first draft vs. the revision.

### `InMemorySaver` checkpointer + `thread_id`

Compiling with a checkpointer lets LangGraph resume interrupted runs. AgensFlow
doesn't care whether a checkpointer is installed — it reads `configurable.thread_id`
either way and uses it as the sidecar key:

```python
from langgraph.checkpoint.memory import InMemorySaver

compiled = graph.compile(checkpointer=InMemorySaver())
await compiled.ainvoke(
    initial_state,
    config={"configurable": {"thread_id": "user_session_42"}},
)
```

For distributed deployments where the graph runs across workers, swap
`InMemorySaver` for `SqliteSaver` / `PostgresSaver` — the AgensFlow sidecar
stays process-local (v1 constraint), so use explicit `decision_id` on
`record_reward` if a worker other than the one that ran the graph submits the
reward.

### Streaming via `astream_events` — decorator fires once per node

The decorator fires exactly once per node execution, regardless of streaming
mode. Under `graph.astream_events()` the underlying LangChain streaming events
fire many times per node (once per token, once per intermediate event) but
`_finish` records the execution once when the node returns:

```python
async for event in compiled.astream_events(initial_state, config=config):
    if event["event"] == "on_chain_stream":
        print(event["data"].get("chunk"))
# Substrate sees 1 decision per node, not 1 per token.
```

### Verifying CostCapture is actually firing

Query `/langgraph/decisions` (or the exposed `tokens_input` / `tokens_output` on
the `DecisionRecordPublic`). If those are `None` after a rewarded run, the
LangChain callback isn't propagating — file a bug. For live inspection:

```bash
curl -H "Authorization: Bearer $AGENSFLOW_API_KEY" \
     "$AGENSFLOW_SERVER_URL/langgraph/decisions?limit=10" \
    | python -m json.tool
```

The evidence_heavy example run recorded (over 4 iterations × 5 nodes = 20
rewarded decisions): planner 1402/321 in/out tokens, memory 2814/220, solver
1517/163, verifier 1117/26, evaluator 653/155 — all captured from real
`ChatOpenAI(base_url=openrouter, ...)` responses via the standardized
`AIMessage.usage_metadata` shape.

## Environment variables (reference)

| Variable                 | Purpose                                         |
|--------------------------|-------------------------------------------------|
| `AGENSFLOW_SERVER_URL`   | Policy server base URL (default: `http://localhost:8000`) |
| `AGENSFLOW_API_KEY`      | Bearer token for the tenant                     |
| `AGENSFLOW_JUDGE`        | Default judge model (e.g. `openrouter:openai/gpt-4o-mini`); only reads when `judge=True` |
| `OPENROUTER_API_KEY`     | Required when using `judge="openrouter:..."`    |

Server-side environment (relevant if self-hosting `agensflow-mcp`):

| Variable                | Purpose                          |
|-------------------------|----------------------------------|
| `AGF_DATABASE_URL`      | Postgres or SQLite DSN           |
| `AGF_JWT_SECRET`        | Server-side secret (production)  |
| `AGF_ENV`               | `dev` / `test` / `production`    |
