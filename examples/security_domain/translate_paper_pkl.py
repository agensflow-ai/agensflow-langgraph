"""Translate the paper's e09 policy_graph.pkl into an OSS-import-shaped JSON.

The paper's substrate uses a rich 12-tuple signature space (regime + activation
flags + calibration params) and a factored action space (planner / memory /
web_search_{exa,tavily} / solver_{skill}_{tier} / verifier_{tier} / evaluator,
each with a `skip:X` variant). The OSS free-tier substrate uses a flat
per-node signature keyed by LangGraph node name, so the paper policy can't
import 1:1.

This translator projects the paper policy onto the OSS action space:

  Paper action                  →  OSS (node, arm)
  ---                              ---
  planner                          planner / default
  memory, web_search_exa,          memory / use     (all three aggregated —
    web_search_tavily                                the OSS memory node
                                                    handles corpus retrieval
                                                    for all three)
  skip:memory, skip:web_search_*   memory / skip
  solver_{skill}_{tier}            solver / {skill}-{tier}
  verifier_{tier}                  verifier / {tier}
  skip:verifier_{tier}             verifier / skip  (both fast+haiku skips
                                                    aggregated)
  evaluator                        evaluator / default
  skip:evaluator                   (dropped — OSS evaluator has no skip arm)
  skip:solver_*                    (dropped — solver is always necessary)

Aggregation: sum visits, weighted mean of reward_mean, Welford m2 merge for
variance. Per-signature discrimination is LOST — the OSS flat signature only
knows the node name, not the paper's regime + activation state.

Output: `examples/starter_policies/security_v1.json` in the shape
`{signature_str: {action_key: {visits, reward_mean, reward_m2, failure_count}}}`
that `agensflow_langgraph.aimport_policy` accepts.

Usage:
    python -m examples.security_domain.translate_paper_pkl
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import defaultdict
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
DEFAULT_PKL = REPO_ROOT / "research" / "e09_cross_domain_security" / "policy_graph.pkl"
DEFAULT_OUT = REPO_ROOT / "examples" / "starter_policies" / "security_v1.json"


# --------------------------------------------------------------------------- #
# Paper-action → OSS-(node, arm) map
# --------------------------------------------------------------------------- #

def _classify(action: str) -> tuple[str, str] | None:
    """Return (oss_node, oss_arm) for a paper action, or None to drop it."""
    if action == "planner":
        return ("planner", "default")

    if action in ("memory", "web_search_exa", "web_search_tavily"):
        return ("memory", "use")
    if action in ("skip:memory", "skip:web_search_exa", "skip:web_search_tavily"):
        return ("memory", "skip")

    if action.startswith("solver_"):
        # solver_{skill}_{tier}  →  (solver, {skill}-{tier})
        parts = action.removeprefix("solver_").split("_", 1)
        if len(parts) == 2:
            skill, tier = parts
            return ("solver", f"{skill}-{tier}")
        return None
    if action.startswith("skip:solver_"):
        return None  # drop — OSS solver has no skip arm

    if action.startswith("verifier_"):
        return ("verifier", action.removeprefix("verifier_"))
    if action.startswith("skip:verifier_"):
        return ("verifier", "skip")

    if action == "evaluator":
        return ("evaluator", "default")
    if action == "skip:evaluator":
        return None  # drop — OSS evaluator has no skip arm

    return None  # unknown action, drop


# --------------------------------------------------------------------------- #
# Welford aggregation
# --------------------------------------------------------------------------- #


def _welford_merge(a: dict, b: dict) -> dict:
    """Merge two (visits, sum, m2) accumulators. Uses parallel-Welford:
       https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    na, sa, ma = a["visits"], a["value_sum"], a["reward_m2"]
    nb, sb, mb = b["visits"], b["value_sum"], b["reward_m2"]
    n = na + nb
    if n == 0:
        return {"visits": 0, "value_sum": 0.0, "reward_m2": 0.0, "failure_count": 0}
    s = sa + sb
    ma_mean = sa / na if na else 0.0
    mb_mean = sb / nb if nb else 0.0
    delta = mb_mean - ma_mean
    m2 = ma + mb + (delta * delta) * (na * nb / n)
    return {
        "visits": n,
        "value_sum": s,
        "reward_m2": m2,
        "failure_count": a.get("failure_count", 0) + b.get("failure_count", 0),
    }


def _empty():
    return {"visits": 0, "value_sum": 0.0, "reward_m2": 0.0, "failure_count": 0}


# --------------------------------------------------------------------------- #
# Main translation
# --------------------------------------------------------------------------- #


def translate(pkl_path: Path, *, max_visits: int | None = None) -> dict:
    """Translate paper pkl → OSS-shaped policy WITH per-regime signatures.

    Paper signature tuples are 12-wide: (regime, activation_flags[7], calibration[4]).
    The first element is the regime label — one of `evidence_heavy`,
    `straightforward`, `ambiguous`. We collapse the other 11 dimensions but
    PRESERVE regime, producing OSS signatures of the form `f"{node}:{regime}"`
    (e.g., `solver:evidence_heavy`).

    This gives the substrate per-regime routing: for each (node, regime) pair,
    it tracks arm stats separately, so `solver:evidence_heavy` can prefer a
    different arm than `solver:straightforward`. Matches the paper's routing
    behavior in the flat OSS substrate.

    If `max_visits` is set, each (signature, arm) cell's visits are capped;
    reward_m2 and failure_count scale proportionally so the empirical variance
    stays consistent. reward_mean is exact.
    """
    with open(pkl_path, "rb") as f:
        paper = pickle.load(f)

    # aggregated[(oss_node, regime)][arm] = accumulator
    aggregated: dict[tuple[str, str], dict[str, dict]] = defaultdict(lambda: defaultdict(_empty))
    dropped: dict[str, int] = defaultdict(int)
    regime_counts: dict[str, int] = defaultdict(int)

    for sig_tuple, node in paper.items():
        # Paper signature is a tuple starting with regime label
        regime = sig_tuple[0] if isinstance(sig_tuple, tuple) and sig_tuple else "default"
        regime_counts[regime] += 1
        for action_key, visits in node.action_visits.items():
            classified = _classify(action_key)
            if classified is None:
                dropped[action_key] += 1
                continue
            oss_node, oss_arm = classified
            value_sum = node.action_value_sums.get(action_key, 0.0)
            reward_m2 = node.action_reward_m2.get(action_key, 0.0)
            failure_count = node.action_failure_count.get(action_key, 0)
            aggregated[(oss_node, regime)][oss_arm] = _welford_merge(
                aggregated[(oss_node, regime)][oss_arm],
                {"visits": visits, "value_sum": value_sum,
                 "reward_m2": reward_m2, "failure_count": failure_count},
            )

    # Convert value_sum → reward_mean for the OSS wire format, applying
    # visit cap if configured. Signature keys are now `f"{node}:{regime}"`.
    policy: dict[str, dict[str, dict]] = {}
    for (node_name, regime), arms in aggregated.items():
        sig_key = f"{node_name}:{regime}"
        policy[sig_key] = {}
        for arm_key, stats in arms.items():
            visits = stats["visits"]
            mean = (stats["value_sum"] / visits) if visits else 0.0
            reward_m2 = stats["reward_m2"]
            failure_count = stats["failure_count"]

            if max_visits is not None and visits > max_visits:
                scale = max_visits / visits
                visits = max_visits
                reward_m2 = reward_m2 * scale
                failure_count = int(round(failure_count * scale))

            policy[sig_key][arm_key] = {
                "visits": int(visits),
                "reward_mean": float(mean),
                "reward_m2": float(reward_m2),
                "failure_count": int(failure_count),
            }

    return {"policy": policy, "dropped": dict(dropped), "regime_counts": dict(regime_counts)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", type=Path, default=DEFAULT_PKL)
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--max-visits", type=int, default=20,
        help="Cap per-arm visits at this value (default 20). Reward means preserved; "
        "m2 + failure_count scaled proportionally. Lower → more UCB1 exploration."
    )
    args = ap.parse_args()

    if not args.pkl.exists():
        raise SystemExit(f"pkl not found: {args.pkl}")

    result = translate(args.pkl, max_visits=args.max_visits)
    policy = result["policy"]

    total_sigs = len(policy)
    total_arms = sum(len(v) for v in policy.values())
    total_visits = sum(a["visits"] for arms in policy.values() for a in arms.values())

    payload = {
        "contract_version": "v1",
        "policy": policy,
        "n_signatures": total_sigs,
        "n_actions": total_arms,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    print(f"  translated {args.pkl.name} → {args.output.name}")
    print(f"    {total_sigs} signatures  {total_arms} arms  {total_visits} total visits")
    for node in sorted(policy):
        arms = policy[node]
        print(f"    {node:<10} {len(arms)} arms:")
        for arm_key in sorted(arms):
            s = arms[arm_key]
            print(f"      {arm_key:<20} v={s['visits']:>5}  μ={s['reward_mean']:.3f}")
    if result["dropped"]:
        print(f"    dropped (not represented in OSS): {dict(result['dropped'])}")


if __name__ == "__main__":
    main()
