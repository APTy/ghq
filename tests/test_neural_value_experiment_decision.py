import copy
from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

from ghqnet.value_experiment_decision import decide, main


CONFIG = json.loads((Path(__file__).parents[1] / "training/experiments/value-targets-20261121.json").read_text())


def measured(logloss, teacher=2.0, neural=3.0):
    return {
        "primary_selfplay": {"overall": {"all": {"wdl": {"logloss": logloss, "labeled_samples": 1024}}}},
        "legacy_diagnostic": {"pooled_policy": {
            "teacher": {"soft_target_nll": teacher, "non_forced_samples": 400},
            "neural": {"soft_target_nll": neural, "non_forced_samples": 500},
        }},
    }


def comparison(deltas=(-0.02, -0.02, -0.02)):
    result = dict(status="complete", complete=True, comparison_eligible=True, training_performed=True,
                  inputs_unchanged=True, comparison={}, baseline=measured(0.9), arms=[])
    for seed, delta in zip(CONFIG["training"]["seeds"], deltas):
        for name, value in (("A", 0.8), ("B", 0.8 + delta)):
            result["arms"].append(dict(arm=name, seed=seed, complete=True, usable_for_comparison=True,
                                       training=dict(steps=32, trained_examples=4096, trained_value_examples=2176),
                                       initial_model_sha256="model", initial_optimizer_sha256="adam",
                                       order_sha256=str(seed), **measured(value)))
    return result


class ValueExperimentDecisionTests(unittest.TestCase):
    def test_complete_passing_comparison_is_only_eligible_for_tactics(self):
        report = comparison()
        before = copy.deepcopy(report)
        decision = decide(report, CONFIG)
        self.assertTrue(decision["eligible_for_tactics"])
        self.assertTrue(decision["valid_comparison"])
        self.assertFalse(decision["promoted"])
        self.assertIsNone(decision["winner"])
        self.assertEqual(report, before)

    def test_each_primary_condition_is_required(self):
        for deltas, failed in [((-0.005, -0.005, -0.005), "mean_primary_improvement"),
                               ((0, 0, -0.1), "at_least_two_seeds_improve"),
                               ((0.01, -0.1, -0.1), "representative_primary_nonworse")]:
            with self.subTest(deltas=deltas):
                decision = decide(comparison(deltas), CONFIG)
                self.assertTrue(decision["valid_comparison"])
                self.assertFalse(decision["eligible_for_tactics"])
                self.assertFalse(decision["checks"][failed])

    def test_threshold_boundary_and_zero_representative_are_allowed(self):
        decision = decide(comparison((0, -0.015, -0.015)), CONFIG)
        self.assertTrue(decision["eligible_for_tactics"])

    def test_retention_gates_teacher_and_neural_mean_and_representative(self):
        for group in ("teacher", "neural"):
            for scope in ("mean", "representative"):
                report = comparison()
                for arm in report["arms"]:
                    if arm["arm"] == "B" and (scope == "mean" or arm["seed"] == 20261125):
                        arm["legacy_diagnostic"]["pooled_policy"][group]["soft_target_nll"] += 0.03
                decision = decide(report, CONFIG)
                self.assertFalse(decision["eligible_for_tactics"])
                self.assertFalse(decision["checks"][f"B_{scope}_{group}_policy_retention"])

    def test_control_retention_is_diagnostic(self):
        report = comparison()
        for arm in report["arms"]:
            if arm["arm"] == "A":
                arm["legacy_diagnostic"]["pooled_policy"]["teacher"]["soft_target_nll"] += 1
        decision = decide(report, CONFIG)
        self.assertTrue(decision["eligible_for_tactics"])
        self.assertEqual(decision["evidence"]["mean_legacy_policy_nll_delta_vs_baseline"]["A"]["teacher"], 1)

    def test_incomplete_duplicate_and_changed_input_reports_fail_closed(self):
        mutations = [lambda r: r.update(complete=False), lambda r: r.update(inputs_unchanged=False),
                     lambda r: r["arms"].pop(), lambda r: r["arms"].__setitem__(5, copy.deepcopy(r["arms"][0])),
                     lambda r: r["arms"][0].update(usable_for_comparison=False),
                     lambda r: r["arms"][0].update(initial_model_sha256="different"),
                     lambda r: r["arms"][0].update(order_sha256="different"),
                     lambda r: r["arms"][0]["training"].update(trained_value_examples=2000),
                     lambda r: r["arms"][0]["training"].update(trained_examples=4000)]
        for mutate in mutations:
            report = comparison()
            mutate(report)
            decision = decide(report, CONFIG)
            self.assertFalse(decision["eligible_for_tactics"])
            self.assertFalse(decision["valid_comparison"])
            self.assertTrue(decision["errors"])

    def test_nonfinite_missing_and_mismatched_panel_metrics_fail_closed(self):
        for value in (None, float("nan"), float("inf"), True):
            report = comparison()
            report["arms"][0]["primary_selfplay"]["overall"]["all"]["wdl"]["logloss"] = value
            self.assertFalse(decide(report, CONFIG)["valid_comparison"])
        report = comparison()
        report["arms"][0]["legacy_diagnostic"]["pooled_policy"]["neural"]["non_forced_samples"] = 499
        self.assertFalse(decide(report, CONFIG)["valid_comparison"])
        report = comparison()
        del report["baseline"]["legacy_diagnostic"]
        self.assertFalse(decide(report, CONFIG)["valid_comparison"])
        report = comparison()
        report["arms"][1]["legacy_diagnostic"]["pooled_policy"]["teacher"]["soft_target_nll"] = float("nan")
        self.assertFalse(decide(report, CONFIG)["valid_comparison"])

    def test_changed_protocol_cannot_silently_change_thresholds(self):
        config = copy.deepcopy(CONFIG)
        config["decision"]["advance_to_tactics_if"][1] = "Any improvement qualifies"
        self.assertFalse(decide(comparison(), config)["valid_comparison"])

    def test_cli_records_inputs_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            report, plan, output = folder / "comparison.json", folder / "plan.json", folder / "decision.json"
            report.write_text(json.dumps(comparison()))
            plan.write_text(json.dumps(CONFIG))
            args = ["--comparison", str(report), "--plan", str(plan), "--output", str(output)]
            with redirect_stdout(io.StringIO()):
                decision = main(args)
            self.assertTrue(decision["eligible_for_tactics"])
            self.assertEqual(len(decision["inputs"]["comparison"]["sha256"]), 64)
            self.assertEqual(json.loads(output.read_text()), decision)
            with self.assertRaises(ValueError):
                main(args)

    def test_cli_accepts_frozen_collection_plan_wrapper(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            report, plan, output = folder / "comparison.json", folder / "plan.json", folder / "decision.json"
            report.write_text(json.dumps(comparison()))
            plan.write_text(json.dumps({"version": 1, "config": CONFIG, "tasks": []}))
            with redirect_stdout(io.StringIO()):
                decision = main(["--comparison", str(report), "--plan", str(plan), "--output", str(output)])
            self.assertTrue(decision["eligible_for_tactics"])


if __name__ == "__main__":
    unittest.main()
