import copy
from contextlib import redirect_stdout
import io
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ghqnet.update_size_decision import decide, main


SEEDS = [20261125, 20261126, 20261127]
RATES = [0.0003, 0.0001]
PLAN = {
    "training": {"seeds": SEEDS, "representative_seed": SEEDS[0], "learning_rates": RATES,
                 "steps": 32, "batch_size": 128, "expected_policy_rows": 4096, "expected_value_rows": 2176},
    "baseline": {"model_sha256": "a" * 64, "optimizer_sha256": "b" * 64},
    "decision": {"policy_tolerance": 0.02, "near_tie": 0.005},
}


def measured(loss=0.3, teacher=2.0, neural=3.0):
    return {
        "primary_selfplay": {"overall": {"all": {"wdl": {"logloss": loss, "labeled_samples": 1024}}}},
        "legacy_diagnostic": {"pooled_policy": {
            "teacher": {"soft_target_nll": teacher, "non_forced_samples": 400},
            "neural": {"soft_target_nll": neural, "non_forced_samples": 500},
        }},
    }


def comparison(large=(0.27, 0.27, 0.27), small=(0.28, 0.28, 0.28)):
    report = dict(status="complete", complete=True, comparison_eligible=True, training_performed=True,
                  inputs_unchanged=True, baseline=measured(), arms=[])
    for rate, losses in zip(RATES, (large, small)):
        for seed, loss in zip(SEEDS, losses):
            report["arms"].append(dict(
                learning_rate=rate, seed=seed, complete=True, usable_for_comparison=True,
                initial_model_sha256="a" * 64, initial_optimizer_sha256="b" * 64,
                order_sha256=f"{seed:064x}", checkpoint=f"seed-{seed}-lr-{rate}.pt",
                mask_sha256="d" * 64,
                checkpoint_sha256=f"{seed + int(rate * 10000):064x}",
                training=dict(steps=32, trained_examples=4096, trained_value_examples=2176),
                **measured(loss)))
    return report


def set_policy(arm, group, value):
    arm["legacy_diagnostic"]["pooled_policy"][group]["soft_target_nll"] = value


def config(decision, rate):
    return next(row for row in decision["configurations"] if row["learning_rate"] == rate)


class UpdateSizeDecisionTests(unittest.TestCase):
    def test_qualified_selection_is_fixed_seed_without_promotion_or_input_mutation(self):
        report = comparison(large=(0.28, 0.20, 0.29))
        before, plan_before = copy.deepcopy(report), copy.deepcopy(PLAN)
        result = decide(report, PLAN)
        self.assertTrue(result["valid_comparison"])
        self.assertTrue(result["proxy_qualified"])
        self.assertFalse(result["exploratory"])
        self.assertEqual(result["selected_learning_rate"], 0.0003)
        self.assertEqual(result["representative_seed"], SEEDS[0])
        self.assertIn(str(SEEDS[0]), result["candidate_checkpoint"])
        self.assertFalse(result["promoted"])
        self.assertFalse(result["playing_strength_established"])
        self.assertEqual(report, before)
        self.assertEqual(PLAN, plan_before)

    def test_near_tie_uses_smaller_rate_with_inclusive_boundary(self):
        for loss, expected in ((0.274, 0.0001), (0.275, 0.0001), (0.27501, 0.0003)):
            with self.subTest(loss=loss):
                result = decide(comparison(small=(loss,) * 3), PLAN)
                self.assertEqual(result["selected_learning_rate"], expected)

    def test_qualified_pool_precedes_a_better_but_unqualified_loss(self):
        report = comparison()
        for arm in report["arms"]:
            if arm["learning_rate"] == 0.0003:
                set_policy(arm, "teacher", 2.03)
        result = decide(report, PLAN)
        self.assertEqual(result["selected_learning_rate"], 0.0001)
        self.assertTrue(result["proxy_qualified"])
        self.assertFalse(config(result, 0.0003)["proxy_qualified"])

    def test_no_qualified_rate_still_selects_one_explicit_exploratory_candidate(self):
        report = comparison(large=(0.32,) * 3, small=(0.323,) * 3)
        for arm in report["arms"]:
            set_policy(arm, "teacher", 2.04)
        result = decide(report, PLAN)
        self.assertTrue(result["valid_comparison"])
        self.assertFalse(result["proxy_qualified"])
        self.assertTrue(result["exploratory"])
        self.assertEqual(result["selected_learning_rate"], 0.0001)
        self.assertTrue(all(not row["proxy_qualified"] for row in result["configurations"]))

    def test_primary_qualification_needs_mean_two_seeds_and_fixed_representative(self):
        cases = [((0.29, 0.29, 0.34), "mean_primary_nonworse"),
                 ((0.3, 0.3, 0.2), "at_least_two_seeds_improve"),
                 ((0.31, 0.2, 0.2), "representative_primary_nonworse")]
        for losses, failed in cases:
            with self.subTest(failed=failed):
                result = decide(comparison(large=losses), PLAN)
                row = config(result, 0.0003)
                self.assertFalse(row["checks"][failed])
                self.assertFalse(row["proxy_qualified"])

    def test_zero_representative_and_exact_policy_tolerance_are_allowed(self):
        report = comparison(large=(0.3, 0.27, 0.27))
        for arm in report["arms"]:
            for group, base in (("teacher", 2.0), ("neural", 3.0)):
                set_policy(arm, group, base + 0.02)
        result = decide(report, PLAN)
        self.assertTrue(config(result, 0.0003)["proxy_qualified"])

    def test_each_policy_group_checks_mean_and_representative(self):
        for group, base in (("teacher", 2.0), ("neural", 3.0)):
            for scope, deltas in (("mean", (0.0, 0.04, 0.04)), ("representative", (0.04, 0.0, 0.0))):
                with self.subTest(group=group, scope=scope):
                    report = comparison()
                    for arm, delta in zip(report["arms"][:3], deltas):
                        set_policy(arm, group, base + delta)
                    result = decide(report, PLAN)
                    checks = config(result, 0.0003)["checks"]
                    self.assertFalse(checks[f"{scope}_{group}_policy_retention"])
                    self.assertFalse(config(result, 0.0003)["proxy_qualified"])

    def test_unordered_report_and_rate_order_do_not_change_selection(self):
        report = comparison(small=(0.274,) * 3)
        expected = decide(report, PLAN)
        report["arms"].reverse()
        plan = copy.deepcopy(PLAN)
        plan["training"]["learning_rates"].reverse()
        actual = decide(report, plan)
        self.assertEqual(actual["selected_learning_rate"], expected["selected_learning_rate"])
        self.assertEqual(actual["candidate_checkpoint"], expected["candidate_checkpoint"])

    def test_incomplete_duplicate_unexpected_or_changed_inputs_fail_closed(self):
        mutations = [lambda r: r.update(complete=False), lambda r: r.update(inputs_unchanged=False),
                     lambda r: r.update(comparison_eligible=False), lambda r: r.update(training_performed=False),
                     lambda r: r["arms"].pop(), lambda r: r["arms"].__setitem__(5, copy.deepcopy(r["arms"][0])),
                     lambda r: r["arms"][0].update(seed=1), lambda r: r["arms"][0].update(learning_rate=0.001),
                     lambda r: r["arms"][0].update(usable_for_comparison=False)]
        for mutate in mutations:
            report = comparison()
            mutate(report)
            result = decide(report, PLAN)
            self.assertFalse(result["valid_comparison"])
            self.assertIsNone(result["candidate_checkpoint"])
            self.assertTrue(result["errors"])

    def test_frozen_initial_state_exposures_orders_and_checkpoint_identity_are_required(self):
        mutations = [lambda a: a.update(initial_model_sha256="c" * 64),
                     lambda a: a.update(initial_optimizer_sha256="c" * 64),
                     lambda a: a.update(order_sha256="c" * 64),
                     lambda a: a.update(mask_sha256="c" * 64),
                     lambda a: a.update(checkpoint=""), lambda a: a.update(checkpoint_sha256="bad"),
                     lambda a: a["training"].update(steps=31),
                     lambda a: a["training"].update(trained_examples=4095),
                     lambda a: a["training"].update(trained_value_examples=2175)]
        for mutate in mutations:
            report = comparison()
            mutate(report["arms"][0])
            result = decide(report, PLAN)
            self.assertFalse(result["valid_comparison"])
            self.assertIsNone(result["selected_learning_rate"])
        report = comparison()
        for arm in report["arms"]:
            arm["initial_model_sha256"] = "c" * 64
        self.assertFalse(decide(report, PLAN)["valid_comparison"])

    def test_nonfinite_boolean_missing_and_mismatched_metrics_fail_closed(self):
        for value in (None, float("nan"), float("inf"), True, -0.1):
            for kind in ("loss", "policy"):
                with self.subTest(value=value, kind=kind):
                    report = comparison()
                    if kind == "loss":
                        report["arms"][0]["primary_selfplay"]["overall"]["all"]["wdl"]["logloss"] = value
                    else:
                        set_policy(report["arms"][0], "neural", value)
                    self.assertFalse(decide(report, PLAN)["valid_comparison"])
        for key in ("non_forced_samples", "soft_target_nll"):
            report = comparison()
            del report["baseline"]["legacy_diagnostic"]["pooled_policy"]["teacher"][key]
            self.assertFalse(decide(report, PLAN)["valid_comparison"])
        report = comparison()
        report["arms"][0]["primary_selfplay"]["overall"]["all"]["wdl"]["labeled_samples"] = 1023
        self.assertFalse(decide(report, PLAN)["valid_comparison"])
        report = comparison()
        report["arms"][0]["legacy_diagnostic"]["pooled_policy"]["neural"]["non_forced_samples"] = 499
        self.assertFalse(decide(report, PLAN)["valid_comparison"])

    def test_invalid_frozen_plan_is_rejected(self):
        mutations = [lambda p: p["training"].update(seeds=[1, 1, 2]),
                     lambda p: p["training"].update(learning_rates=[0.0001, 0.0001]),
                     lambda p: p["training"].update(representative_seed=0),
                     lambda p: p["training"].update(expected_policy_rows=4000),
                     lambda p: p["training"].update(expected_value_rows=4097),
                     lambda p: p["decision"].update(near_tie=float("nan")),
                     lambda p: p["decision"].update(policy_tolerance=-0.01)]
        for mutate in mutations:
            plan = copy.deepcopy(PLAN)
            mutate(plan)
            self.assertFalse(decide(comparison(), plan)["valid_comparison"])

    def test_cli_records_frozen_input_hashes_resolves_checkpoint_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            report, plan, output = folder / "report.json", folder / "plan.json", folder / "decision.json"
            comparison_report = comparison()
            candidate = comparison_report["arms"][0]
            (folder / candidate["checkpoint"]).write_bytes(b"frozen diagnostic checkpoint")
            candidate["checkpoint_sha256"] = hashlib.sha256(b"frozen diagnostic checkpoint").hexdigest()
            report.write_text(json.dumps(comparison_report))
            plan.write_text(json.dumps({"config": PLAN}))
            args = ["--comparison", str(report), "--plan", str(plan), "--output", str(output)]
            with redirect_stdout(io.StringIO()):
                result = main(args)
            self.assertTrue(result["valid_comparison"])
            self.assertEqual(len(result["inputs"]["report"]["sha256"]), 64)
            self.assertEqual(len(result["inputs"]["plan"]["sha256"]), 64)
            self.assertEqual(Path(result["candidate_checkpoint"]).parent, folder.resolve())
            self.assertEqual(json.loads(output.read_text()), result)
            with self.assertRaises(ValueError):
                main(args)

    def test_cli_refuses_to_publish_changed_or_missing_checkpoint(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            report, plan, output = folder / "report.json", folder / "plan.json", folder / "decision.json"
            comparison_report = comparison()
            checkpoint = folder / comparison_report["arms"][0]["checkpoint"]
            report.write_text(json.dumps(comparison_report))
            plan.write_text(json.dumps(PLAN))
            args = ["--report", str(report), "--plan", str(plan), "--output", str(output)]
            with self.assertRaises(FileNotFoundError):
                main(args)
            self.assertFalse(output.exists())
            checkpoint.write_bytes(b"different checkpoint")
            with self.assertRaisesRegex(ValueError, "checkpoint content differs"):
                main(args)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
