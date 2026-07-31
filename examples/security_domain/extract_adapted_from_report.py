"""Rebuild security_v1_adapted.json with per-(node, regime) signatures.

Merges two sources — zero fresh LLM cost:

  1. Paper priors from `security_v1.json` (already per-regime after the v0.1.4
     translator upgrade).
  2. The 120 real runs from a `verify_warmstart` report — for each task-metric,
     attribute its quality to every node's chosen arm at that task's regime.

Scenario-class → regime mapping matches the paper's e09 encoding:

  * C1, C6, C8 → straightforward (factual, definitional, numerical extraction)
  * C2, C3, C4, C7 → evidence_heavy (synthesis, cross-doc, reasoning)
  * C5 → ambiguous (no answer in corpus)

Usage:
    python -m examples.security_domain.extract_adapted_from_report \\
        --report reports/verify_20260719_162928.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent.parent
DEFAULT_STARTER = REPO_ROOT / "examples" / "starter_policies" / "security_v1.json"
DEFAULT_OUT = REPO_ROOT / "examples" / "starter_policies" / "security_v1_adapted.json"


SCENARIO_TO_REGIME = {
    "C1": "straightforward", "C6": "straightforward", "C8": "straightforward",
    "C2": "evidence_heavy",   "C3": "evidence_heavy",   "C4": "evidence_heavy",  "C7": "evidence_heavy",
    "C5": "ambiguous",
}


def _empty():
    return {"visits": 0, "value_sum": 0.0, "reward_m2": 0.0, "failure_count": 0}


def _welford_add(acc: dict, quality: float) -> dict:
    """Add one sample to a running Welford accumulator."""
    n = acc["visits"] + 1
    old_mean = acc["value_sum"] / acc["visits"] if acc["visits"] else 0.0
    delta = quality - old_mean
    new_mean = old_mean + delta / n
    m2 = acc["reward_m2"] + delta * (quality - new_mean)
    return {
        "visits": n,
        "value_sum": acc["value_sum"] + quality,
        "reward_m2": m2,
        "failure_count": acc["failure_count"],
    }


def _welford_merge(a: dict, b: dict) -> dict:
    """Merge two Welford accumulators (parallel formula)."""
    na, sa, ma = a["visits"], a["value_sum"], a["reward_m2"]
    nb, sb, mb = b["visits"], b["value_sum"], b["reward_m2"]
    n = na + nb
    if n == 0:
        return _empty()
    ma_mean = sa / na if na else 0.0
    mb_mean = sb / nb if nb else 0.0
    delta = mb_mean - ma_mean
    m2 = ma + mb + (delta * delta) * (na * nb / n)
    return {
        "visits": n,
        "value_sum": sa + sb,
        "reward_m2": m2,
        "failure_count": a.get("failure_count", 0) + b.get("failure_count", 0),
    }


def build_adapted(starter_path: Path, report_path: Path) -> dict:
    # 1. Load paper-priors starter (already per-regime keys)
    starter = json.loads(starter_path.read_text())
    prior_policy = starter.get("policy", {})

    # Convert prior to accumulator shape for merging
    accs: dict[str, dict[str, dict]] = defaultdict(lambda: defaultdict(_empty))
    for sig, arms in prior_policy.items():
        for arm, stats in arms.items():
            visits = int(stats.get("visits", 0))
            mean = float(stats.get("reward_mean", 0.0))
            accs[sig][arm] = {
                "visits": visits,
                "value_sum": mean * visits,
                "reward_m2": float(stats.get("reward_m2", 0.0)),
                "failure_count": int(stats.get("failure_count", 0)),
            }

    # 2. Layer report evidence on top
    report = json.loads(report_path.read_text())
    task_metrics = report.get("task_metrics", [])
    added = 0
    skipped = 0
    unmapped_classes: set[str] = set()

    for tm in task_metrics:
        if tm.get("status") != "ok":
            skipped += 1
            continue
        sc = tm.get("scenario_class")
        regime = SCENARIO_TO_REGIME.get(sc)
        if regime is None:
            unmapped_classes.add(str(sc))
            skipped += 1
            continue
        quality = float(tm.get("quality", 0.0))
        for step in tm.get("trace", []):
            node = step.get("node")
            arm = step.get("action")
            if not node or not arm or arm == "<fell-open>":
                continue
            sig_key = f"{node}:{regime}"
            accs[sig_key][arm] = _welford_add(accs[sig_key][arm], quality)
        added += 1

    # 3. Serialize to wire format
    policy: dict[str, dict[str, dict]] = {}
    for sig, arms in accs.items():
        policy[sig] = {}
        for arm, acc in arms.items():
            visits = acc["visits"]
            mean = (acc["value_sum"] / visits) if visits else 0.0
            policy[sig][arm] = {
                "visits": int(visits),
                "reward_mean": float(mean),
                "reward_m2": float(acc["reward_m2"]),
                "failure_count": int(acc["failure_count"]),
            }

    return {
        "policy": policy,
        "n_signatures": len(policy),
        "n_actions": sum(len(v) for v in policy.values()),
        "runs_layered": added,
        "runs_skipped": skipped,
        "unmapped_classes": sorted(unmapped_classes),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--starter", type=Path, default=DEFAULT_STARTER)
    ap.add_argument("--report", type=Path, required=True,
                    help="Path to a verify_warmstart report JSON")
    ap.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args()

    if not args.starter.exists():
        raise SystemExit(f"starter not found: {args.starter}")
    if not args.report.exists():
        raise SystemExit(f"report not found: {args.report}")

    result = build_adapted(args.starter, args.report)
    payload = {
        "contract_version": "v1",
        "policy": result["policy"],
        "n_signatures": result["n_signatures"],
        "n_actions": result["n_actions"],
        "source": (
            f"paper priors ({args.starter.name}) + "
            f"{result['runs_layered']} real runs from {args.report.name}, "
            f"per-(node, regime) attributed"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2))

    print(f"  wrote {args.output.name}")
    print(f"    signatures: {result['n_signatures']}   arms: {result['n_actions']}")
    print(f"    layered {result['runs_layered']} report runs, skipped {result['runs_skipped']}")
    if result["unmapped_classes"]:
        print(f"    unmapped classes (dropped): {result['unmapped_classes']}")
    print()
    for sig in sorted(result["policy"]):
        print(f"    {sig}:")
        ranked = sorted(result["policy"][sig].items(),
                        key=lambda kv: -kv[1].get("reward_mean", 0))
        for arm, s in ranked:
            v = int(s.get("visits", 0))
            m = s.get("reward_mean", 0)
            marker = " ← preferred" if arm == ranked[0][0] else ""
            print(f"      {arm:<18} v={v:>3}  μ={m:.3f}{marker}")


if __name__ == "__main__":
    main()
