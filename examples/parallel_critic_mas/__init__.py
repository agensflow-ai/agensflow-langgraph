"""parallel_critic MAS — planner → memory → solver → [critic || verifier] → evaluator.

Exercises LangGraph's parallel node execution + fan-in with @agensflow-decorated
nodes on both branches. Same benchmark task set as evidence_heavy_mas so the
two examples can share the notebook demo.
"""
