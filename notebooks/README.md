# Notebooks

## `token_budgets_workshop.ipynb`

Companion notebook to the TwoSetAI Workshop #3 talk *"From Token Maxing to Token Budgeting"* (2026-08-14).

Replays 780 real Claude Code + `agent-skills-main` runs across 3 signatures × 64-arm action space. Sections 1-5 are deterministic replays from shipped data (`data/workshop_runs.jsonl`) — no LLM calls, no API keys. Section 6 optionally fires ONE live LangGraph node against OpenRouter (~$0.02) to show the same audit trail on a different framework.

### Run

**Colab** (recommended): click the badge in cell 1. Auto-installs deps + clones the repo.

**Local**:
```bash
pip install agensflow-mcp agensflow-langgraph asgi-lifespan langchain-openai python-dotenv jupyter
jupyter notebook token_budgets_workshop.ipynb
```

### Knobs

- `PROFILE` (Setup cell) — one of `'thrifty'`, `'balanced'`, `'premium'`. Rescores the 780 runs under a different reward tradeoff; Section 2/3/4 argmax will shift. Try flipping it and re-running Sections 2-4 to see how the substrate picks different arms for different users.
- `MIN_N` (Setup cell) — minimum observations before an arm counts as "settled." Default 5.

### Section 6 requires

Set `OPENROUTER_API_KEY` in your env or a `.env` file in this folder. Costs ~$0.02 per run.

### Data

`data/workshop_runs.jsonl` — 780 rows, one per substrate decision. Fields: `signature`, `action`, `choice`, `choice_resolved`, `total_cost`, `total_latency`, `total_output_tokens`, `total_calls`, `judge_quality`, per-axis judge means, per-judge verdicts + Q_candidate, `reward_by_profile` for three profiles, `final_acceptance`.
