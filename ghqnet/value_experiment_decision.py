"""Pure, fail-closed development gate for the frozen value-target experiment."""

import argparse
import hashlib
import json
import math
from pathlib import Path


PRIMARY_RULE = (
    "Mean paired primary WDL logloss delta <= -0.01, at least2/3 seeds improve, "
    "and fixed representative seed20261125 has nonpositive paired delta."
)
POLICY_RULE = (
    "Mean nonforced policy NLL on the unchanged legacy panel worsens by <=0.02 "
    "versus baseline for both teacher and pooled neural sources; fixed "
    "representative also respects this tolerance."
)
THRESHOLDS = {
    "mean_paired_logloss_delta_max": -0.01,
    "minimum_improving_seeds": 2,
    "representative_logloss_delta_max": 0.0,
    "legacy_policy_nll_delta_max": 0.02,
}


def _number(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{label} must be finite")
    return value


def _positive_int(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _metrics(entry, label):
    primary = entry["primary_selfplay"]["overall"]["all"]["wdl"]
    value = _number(primary["logloss"], f"{label} primary logloss")
    count = _positive_int(primary["labeled_samples"], f"{label} primary labels")
    if value < 0:
        raise ValueError(f"{label} primary logloss cannot be negative")
    policy, counts = {}, {}
    for group in ("teacher", "neural"):
        row = entry["legacy_diagnostic"]["pooled_policy"][group]
        policy[group] = _number(row["soft_target_nll"], f"{label} {group} policy NLL")
        counts[group] = _positive_int(row["non_forced_samples"], f"{label} {group} policy rows")
        if policy[group] < 0:
            raise ValueError(f"{label} {group} policy NLL cannot be negative")
    return value, policy, (count, counts)


def decide(report, config):
    """Recompute the frozen gate from arm metrics, never a selected best seed.

    Invalid or incomplete reports return an ineligible decision with an error.
    Arm A's policy retention is diagnostic; B is the proposed value-target change.
    This gate permits development tactics only, never deployment or promotion.
    """
    result = {
        "version": 1,
        "purpose": "Frozen value-target development advancement gate",
        "eligible_for_tactics": False,
        "valid_comparison": False,
        "thresholds": dict(THRESHOLDS),
        "checks": {},
        "evidence": {},
        "errors": [],
        "winner": None,
        "promoted": False,
        "playing_strength_established": False,
        "retention_gate_arm": "B",
        "note": "Arm A retention is diagnostic. Passing permits development tactics only.",
    }
    try:
        seeds = config["training"]["seeds"]
        representative = config["training"]["representative_seed"]
        criteria = config["decision"]["advance_to_tactics_if"]
        if (config.get("version") != 1 or seeds != [20261125, 20261126, 20261127]
                or representative != 20261125 or PRIMARY_RULE not in criteria or POLICY_RULE not in criteria):
            raise ValueError("Configuration does not match the frozen decision protocol")
        steps = _positive_int(config["training"]["steps"], "planned steps")
        batch_size = _positive_int(config["training"]["batch_size"], "planned batch size")
        if (report.get("status") != "complete" or report.get("complete") is not True
                or report.get("comparison_eligible") is not True
                or report.get("training_performed") is not True
                or report.get("inputs_unchanged") is not True
                or not isinstance(report.get("comparison"), dict)):
            raise ValueError("Comparison is incomplete or its integrity checks did not pass")
        arms = report["arms"]
        if not isinstance(arms, list) or len(arms) != 6:
            raise ValueError("Require exactly six completed arms")
        expected = {(seed, arm) for seed in seeds for arm in ("A", "B")}
        by_key = {}
        initial_states = set()
        for arm in arms:
            key = arm["seed"], arm["arm"]
            if key not in expected or key in by_key:
                raise ValueError("Missing, duplicate, or unexpected seed/arm")
            if arm.get("complete") is not True or arm.get("usable_for_comparison") is not True:
                raise ValueError("Every arm must be complete and usable for comparison")
            metrics = arm["training"]
            if metrics.get("steps") != steps or metrics.get("trained_examples") != steps * batch_size:
                raise ValueError("Actual training exposure differs from the frozen policy budget")
            _positive_int(metrics["trained_value_examples"], "actual value exposures")
            initial = arm["initial_model_sha256"], arm["initial_optimizer_sha256"]
            if not all(isinstance(value, str) and value for value in (*initial, arm["order_sha256"])):
                raise ValueError("Missing initial state or policy-order hashes")
            initial_states.add(initial)
            by_key[key] = arm
        if set(by_key) != expected or len(initial_states) != 1:
            raise ValueError("All arms must restore the same model and optimizer")
        baseline_value, baseline_policy, baseline_counts = _metrics(report["baseline"], "baseline")
        per_seed = []
        for seed in seeds:
            a, b = by_key[(seed, "A")], by_key[(seed, "B")]
            if (a["order_sha256"] != b["order_sha256"]
                    or a["training"]["trained_value_examples"] != b["training"]["trained_value_examples"]):
                raise ValueError("Paired arms must match policy order and actual value exposures")
            av, ap, ac = _metrics(a, f"seed {seed} arm A")
            bv, bp, bc = _metrics(b, f"seed {seed} arm B")
            if ac != baseline_counts or bc != baseline_counts:
                raise ValueError("All arms must use the same held-out metric panels")
            per_seed.append({
                "seed": seed,
                "primary_logloss": {"A": av, "B": bv},
                "paired_logloss_delta": bv - av,
                "legacy_policy_nll_delta_vs_baseline": {
                    "A": {group: ap[group] - baseline_policy[group] for group in baseline_policy},
                    "B": {group: bp[group] - baseline_policy[group] for group in baseline_policy},
                },
                "value_exposures_each_arm": a["training"]["trained_value_examples"],
            })
        representative_row = next(row for row in per_seed if row["seed"] == representative)
        mean_delta = sum(row["paired_logloss_delta"] for row in per_seed) / len(seeds)
        improved = sum(row["paired_logloss_delta"] < 0 for row in per_seed)
        retention = {
            arm: {group: sum(row["legacy_policy_nll_delta_vs_baseline"][arm][group]
                            for row in per_seed) / len(seeds)
                  for group in baseline_policy}
            for arm in ("A", "B")
        }
        # Only compensate for floating-point subtraction at exact boundaries.
        at_most = lambda value, limit: value <= limit + 1e-12
        checks = {
            "complete_integrity_and_exposure": True,
            "mean_primary_improvement": at_most(mean_delta, THRESHOLDS["mean_paired_logloss_delta_max"]),
            "at_least_two_seeds_improve": improved >= THRESHOLDS["minimum_improving_seeds"],
            "representative_primary_nonworse": at_most(representative_row["paired_logloss_delta"], 0),
        }
        for group in baseline_policy:
            checks[f"B_mean_{group}_policy_retention"] = at_most(retention["B"][group], 0.02)
            checks[f"B_representative_{group}_policy_retention"] = at_most(
                representative_row["legacy_policy_nll_delta_vs_baseline"]["B"][group], 0.02)
        result.update(valid_comparison=True, eligible_for_tactics=all(checks.values()), checks=checks,
                      evidence={"per_seed": per_seed, "representative_seed": representative,
                                "baseline_primary_logloss": baseline_value,
                                "baseline_legacy_policy_nll": baseline_policy,
                                "mean_paired_logloss_delta": mean_delta,
                                "improving_seeds": improved,
                                "mean_legacy_policy_nll_delta_vs_baseline": retention})
    except (KeyError, TypeError, ValueError, IndexError) as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
        result["checks"]["complete_integrity_and_exposure"] = False
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Decision output must be new")
    report_bytes, config_bytes = args.comparison.read_bytes(), args.plan.read_bytes()
    config = json.loads(config_bytes)
    result = decide(json.loads(report_bytes), config.get("config", config))
    result["inputs"] = {
        "comparison": {"path": str(args.comparison.resolve()), "sha256": hashlib.sha256(report_bytes).hexdigest()},
        "plan": {"path": str(args.plan.resolve()), "sha256": hashlib.sha256(config_bytes).hexdigest()},
        "decision_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, allow_nan=False))
    return result


if __name__ == "__main__":
    main()
