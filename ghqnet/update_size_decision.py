"""Select one fixed-seed update-size candidate for a bounded match experiment.

Prediction metrics qualify or rank candidates; they do not establish playing
strength. A valid comparison always yields one candidate, explicitly marked
exploratory when neither learning rate passes the frozen proxy criteria.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re


def _number(value, label, *, positive=False):
    if (isinstance(value, bool) or not isinstance(value, (int, float))
            or not math.isfinite(value) or value < 0 or (positive and value == 0)):
        raise ValueError(f"{label} must be finite and {'positive' if positive else 'nonnegative'}")
    return value


def _positive_int(value, label):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _sha(value, label):
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{label} must be a SHA-256 digest")
    return value


def _metrics(entry, label):
    primary = entry["primary_selfplay"]["overall"]["all"]["wdl"]
    loss = _number(primary["logloss"], f"{label} primary logloss")
    count = _positive_int(primary["labeled_samples"], f"{label} primary label count")
    policy, policy_counts = {}, {}
    for group in ("teacher", "neural"):
        row = entry["legacy_diagnostic"]["pooled_policy"][group]
        policy[group] = _number(row["soft_target_nll"], f"{label} {group} policy NLL")
        policy_counts[group] = _positive_int(row["non_forced_samples"], f"{label} {group} policy count")
    return loss, policy, (count, policy_counts)


def decide(report, plan):
    """Fail closed on invalid evidence; never pick a seed or promote a model."""
    result = {
        "version": 1,
        "purpose": "Frozen update-size calibration selection for bounded actual matches",
        "valid_comparison": False,
        "selected_learning_rate": None,
        "representative_seed": None,
        "candidate_checkpoint": None,
        "candidate_checkpoint_sha256": None,
        "proxy_qualified": False,
        "exploratory": False,
        "checks": {},
        "configurations": [],
        "errors": [],
        "promoted": False,
        "playing_strength_established": False,
        "note": "Selection permits tactical checks; hard tactical validity gates precede the fixed arena.",
    }
    try:
        training = plan["training"]
        seeds, rates = training["seeds"], training["learning_rates"]
        if (not isinstance(seeds, list) or len(seeds) != 3
                or any(type(seed) is not int or seed < 0 for seed in seeds) or len(set(seeds)) != 3):
            raise ValueError("Require three distinct integer seeds")
        if not isinstance(rates, list) or len(rates) != 2:
            raise ValueError("Require two learning rates")
        for rate in rates:
            _number(rate, "learning rate", positive=True)
        if len(set(rates)) != 2:
            raise ValueError("Learning rates must be distinct")
        representative = training["representative_seed"]
        if type(representative) is not int or representative not in seeds:
            raise ValueError("Representative seed must belong to the frozen seed list")
        steps = _positive_int(training["steps"], "planned steps")
        batch_size = _positive_int(training["batch_size"], "batch size")
        expected_policy = _positive_int(training["expected_policy_rows"], "expected policy rows")
        expected_value = _positive_int(training["expected_value_rows"], "expected value rows")
        if expected_policy != steps * batch_size or expected_value > expected_policy:
            raise ValueError("Frozen exposure counts are inconsistent")
        initial = (_sha(plan["baseline"]["model_sha256"], "baseline model"),
                   _sha(plan["baseline"]["optimizer_sha256"], "baseline optimizer"))
        tolerance = _number(plan["decision"]["policy_tolerance"], "policy tolerance")
        near_tie = _number(plan["decision"]["near_tie"], "near-tie tolerance")
        result["thresholds"] = {"policy_tolerance": tolerance, "near_tie": near_tie,
                                "minimum_improving_seeds": 2}
        if (report.get("status") != "complete" or report.get("complete") is not True
                or report.get("comparison_eligible") is not True
                or report.get("training_performed") is not True
                or report.get("inputs_unchanged") is not True):
            raise ValueError("Comparison is incomplete or failed its integrity checks")
        arms = report["arms"]
        if not isinstance(arms, list) or len(arms) != 6:
            raise ValueError("Require exactly six completed seed/rate arms")
        baseline_loss, baseline_policy, counts = _metrics(report["baseline"], "baseline")
        expected = {(seed, rate) for seed in seeds for rate in rates}
        by_key = {}
        for arm in arms:
            seed, rate = arm["seed"], arm["learning_rate"]
            if type(seed) is not int:
                raise ValueError("Arm seed must be an integer")
            _number(rate, "arm learning rate", positive=True)
            key = seed, rate
            if key not in expected or key in by_key:
                raise ValueError("Duplicate or unexpected seed/rate arm")
            if arm.get("complete") is not True or arm.get("usable_for_comparison") is not True:
                raise ValueError("Every arm must be complete and usable")
            actual = arm["training"]
            if any(type(actual[name]) is not int or actual[name] != count for name, count in (
                    ("steps", steps), ("trained_examples", expected_policy),
                    ("trained_value_examples", expected_value))):
                raise ValueError("Actual exposure differs from the frozen step, policy, or value budget")
            if (arm["initial_model_sha256"], arm["initial_optimizer_sha256"]) != initial:
                raise ValueError("Every arm must restore the frozen model and Adam state")
            _sha(arm["order_sha256"], "policy order")
            _sha(arm["mask_sha256"], "value eligibility mask")
            if not isinstance(arm["checkpoint"], str) or not arm["checkpoint"].strip():
                raise ValueError("Every arm must identify its checkpoint")
            _sha(arm["checkpoint_sha256"], "checkpoint")
            loss, policy, arm_counts = _metrics(arm, f"seed {seed} rate {rate}")
            if arm_counts != counts:
                raise ValueError("Held-out metric panels differ across arms")
            by_key[key] = (arm, loss, policy)
        if set(by_key) != expected:
            raise ValueError("Missing planned seed/rate arm")
        for seed in seeds:
            if len({by_key[(seed, rate)][0]["order_sha256"] for rate in rates}) != 1:
                raise ValueError("Paired learning rates must have identical policy order")
            if len({by_key[(seed, rate)][0]["mask_sha256"] for rate in rates}) != 1:
                raise ValueError("Paired learning rates must have identical value eligibility masks")

        at_most = lambda value, limit: value <= limit + 1e-12
        configurations = []
        for rate in rates:
            rows = [{"seed": seed, "primary_logloss": by_key[(seed, rate)][1],
                     "primary_delta_vs_baseline": by_key[(seed, rate)][1] - baseline_loss,
                     "policy_delta_vs_baseline": {
                         group: by_key[(seed, rate)][2][group] - baseline_policy[group]
                         for group in baseline_policy}}
                    for seed in seeds]
            mean_loss = sum(row["primary_logloss"] for row in rows) / len(rows)
            rep = next(row for row in rows if row["seed"] == representative)
            policy_delta = {group: sum(row["policy_delta_vs_baseline"][group] for row in rows) / len(rows)
                            for group in baseline_policy}
            checks = {
                "mean_primary_nonworse": at_most(mean_loss, baseline_loss),
                "at_least_two_seeds_improve": sum(row["primary_delta_vs_baseline"] < 0 for row in rows) >= 2,
                "representative_primary_nonworse": at_most(rep["primary_logloss"], baseline_loss),
            }
            for group in baseline_policy:
                checks[f"mean_{group}_policy_retention"] = at_most(policy_delta[group], tolerance)
                checks[f"representative_{group}_policy_retention"] = at_most(
                    rep["policy_delta_vs_baseline"][group], tolerance)
            configurations.append({"learning_rate": rate, "proxy_qualified": all(checks.values()),
                                   "checks": checks, "mean_primary_logloss": mean_loss,
                                   "mean_primary_delta_vs_baseline": mean_loss - baseline_loss,
                                   "mean_policy_delta_vs_baseline": policy_delta, "per_seed": rows})
        qualified = [entry for entry in configurations if entry["proxy_qualified"]]
        pool = qualified or configurations
        best_loss = min(entry["mean_primary_logloss"] for entry in pool)
        chosen = min((entry for entry in pool if at_most(entry["mean_primary_logloss"], best_loss + near_tie)),
                     key=lambda entry: entry["learning_rate"])
        rate = chosen["learning_rate"]
        candidate = by_key[(representative, rate)][0]
        result.update(
            valid_comparison=True, selected_learning_rate=rate, representative_seed=representative,
            candidate_checkpoint=candidate["checkpoint"], candidate_checkpoint_sha256=candidate["checkpoint_sha256"],
            proxy_qualified=bool(qualified), exploratory=not bool(qualified),
            selection_pool="proxy-qualified" if qualified else "all-valid-exploratory",
            configurations=configurations,
            checks={"complete_integrity_and_exposure": True},
            baseline={"primary_logloss": baseline_loss, "policy_nll": baseline_policy},
        )
    except (KeyError, TypeError, ValueError, IndexError) as error:
        result["errors"].append(f"{type(error).__name__}: {error}")
        result["checks"]["complete_integrity_and_exposure"] = False
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", "--comparison", dest="report", type=Path, required=True)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise ValueError("Decision output must be new")
    report_bytes, plan_bytes = args.report.read_bytes(), args.plan.read_bytes()
    plan = json.loads(plan_bytes)
    result = decide(json.loads(report_bytes), plan.get("config", plan))
    if result["candidate_checkpoint"] is not None:
        path = Path(result["candidate_checkpoint"])
        if not path.is_absolute():
            path = args.report.parent / path
        result["candidate_checkpoint"] = str(path.resolve())
        digest = hashlib.sha256()
        with path.open("rb") as checkpoint:
            for chunk in iter(lambda: checkpoint.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != result["candidate_checkpoint_sha256"]:
            raise ValueError("Selected checkpoint content differs from its frozen report digest")
    result["inputs"] = {
        "report": {"path": str(args.report.resolve()), "sha256": hashlib.sha256(report_bytes).hexdigest()},
        "plan": {"path": str(args.plan.resolve()), "sha256": hashlib.sha256(plan_bytes).hexdigest()},
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
