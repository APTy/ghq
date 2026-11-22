"""The experiment freezes masks and budgets before any optional optimization."""
from copy import deepcopy
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Value ablation tests require torch")

from ghqnet import ablate_value_targets as ablation
from ghqnet.curriculum import fingerprint
from ghqnet.data import Sample, write_replay
from ghqnet.model import NetworkConfig, PolicyValueNet, load_checkpoint, save_checkpoint
from ghqnet.rules import e


def provenance():
    final = e.BaseBoard()
    final.push(e.Move.from_uci("skip"))
    final.push(e.Move.from_uci("skip"))
    player = {"model_id": "model-sha", "search": {
        "kind": "neural", "simulations": 16, "cpuct": 1.5, "max_nodes": 256,
        "max_depth": 256, "root_noise": True, "dirichlet_alpha": .3,
        "noise_fraction": .25, "temperature_turns": 8, "temperature": 1., "late_temperature": 0.}}
    return {"version": 1, "collection_kind": "self", "players": {"RED": deepcopy(player), "BLUE": player},
            "terminal": {"winner": None, "reason": "double-skip"},
            "termination_reason": "double-skip", "final_state": final.serialize()}


def row(board, identity, eligible=True):
    actions = sorted(move.uci() for move in board.generate_legal_moves())
    return Sample(board.serialize(), actions, [float(action == "skip") for action in actions],
                  outcome=1, source="mcts:new" if eligible else "teacher", game_id=identity, family_id=identity,
                  value_provenance=provenance() if eligible else None)


def fixture(root):
    board = e.BaseBoard()
    training = [row(board, f"train-{index}", index < 4) for index in range(8)]
    board.push(e.Move.from_uci("rhe1"))
    validation = [row(board, f"dev-{index}", index < 4) for index in range(6)]
    validation[-1].source = "mcts:old"
    write_replay(root / "training.jsonl", training)
    write_replay(root / "validation.jsonl", validation)
    (root / "partition.json").write_text(json.dumps({"partition": {
        "train_states": sorted({fingerprint(item) for item in training}),
        "validation_states": sorted({fingerprint(item) for item in validation})}}))
    model = PolicyValueNet(NetworkConfig(blocks=0, channels=4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    for parameter in model.parameters():
        optimizer.state[parameter] = {"step": torch.tensor(7.), "exp_avg": torch.zeros_like(parameter),
                                      "exp_avg_sq": torch.ones_like(parameter)}
    save_checkpoint(root / "checkpoint.pt", model, optimizer)
    return training, validation


def mock_metrics(*args, **kwargs):
    group = {"all": {"samples": 1, "wdl": {"logloss": .5, "brier": .3, "accuracy": .7}},
             "non_forced": {"samples": 1, "policy": {"top1_agreement": .3, "top5_agreement": .6, "soft_target_nll": 2.5}}}
    return {"overall": deepcopy(group), "by_source": {"fixed-source": deepcopy(group)}}


def fake_fit(model, rows, **options):
    optimizer = options["optimizer"]
    steps = options["max_examples"] // options["batch_size"]
    for state in optimizer.state.values():
        state["step"] += steps
    chosen = [item for item, include in zip(rows, options["value_mask"]) if include and item.outcome is not None]
    with torch.no_grad():
        next(model.parameters()).add_(.001)
    return optimizer, {"steps": steps, "interrupted": False, "trained_examples": options["max_examples"],
                       "trained_value_examples": len(chosen),
                       "trained_value_examples_by_source": {source: sum(item.source == source for item in chosen)
                                                             for source in {item.source for item in chosen}}}


class ValueAblationTests(unittest.TestCase):
    def args(self, root, *extra):
        return ablation.parser().parse_args([
            "--checkpoint", str(root / "checkpoint.pt"), "--training", str(root / "training.jsonl"),
            "--validation", str(root / "validation.jsonl"), "--partition", str(root / "partition.json"),
            "--output", str(root / "out"), "--model-id", "model-sha", "--seeds", "17", "19",
            "--steps", "2", "--batch-size", "4", "--threads", "1",
            "--min-training-games", "1", "--min-validation-games", "1",
            "--min-training-families", "1", "--min-validation-families", "1",
            "--min-training-rows", "1", "--min-validation-rows", "1",
            "--min-control-contrast", "0", *extra])

    def test_seed_masks_have_identical_policy_order_and_exact_per_batch_value_counts(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _ = fixture(Path(directory))
            eligible = [index < 4 for index in range(8)]
            plan = ablation.make_seed_plan(rows, eligible, seed=17, steps=2, batch_size=4)
            self.assertEqual(plan, ablation.make_seed_plan(rows, eligible, seed=17, steps=2, batch_size=4))
            expected = list(range(8))
            random.Random(17).shuffle(expected)
            self.assertEqual(plan["order"], expected)
            self.assertEqual(len(set(plan["order"])), 8)
            self.assertEqual(plan["labels"]["A"]["rows"], plan["labels"]["B"]["rows"])
            self.assertTrue(all(item["A_value_labels"] == item["B_value_labels"] for item in plan["per_batch"]))
            self.assertEqual(plan["masks"]["B"], eligible)
            self.assertEqual(plan["labels"]["B"]["wdl"], {"win": 0, "draw": 4, "loss": 0})
            self.assertEqual(plan["control_outside_selfplay_labels"] + plan["overlap"], 4)
            with self.assertRaisesRegex(ValueError, "no-repeat"):
                ablation.make_seed_plan(rows, eligible, seed=17, steps=3, batch_size=4)

    def test_prepare_default_freezes_manifest_without_inference_training_or_raw_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            inputs = {path: ablation.file_sha(path) for path in root.iterdir()}
            with patch.object(ablation, "fit", side_effect=AssertionError("must not train")), \
                    patch.object(ablation, "evaluate_bounded", side_effect=AssertionError("must not infer")), \
                    patch.object(PolicyValueNet, "forward", side_effect=AssertionError("must not execute network")):
                report = ablation.run_ablation(self.args(root))
            self.assertEqual(report["status"], "prepared")
            self.assertFalse(report["training_performed"])
            self.assertFalse(report["comparison_eligible"])
            self.assertTrue(report["inputs_unchanged"])
            manifest = json.loads((root / "out/manifest.json").read_text())
            self.assertEqual(len(manifest["seeds"]), 2)
            self.assertEqual(manifest["primary_validation"]["families"], 4)
            self.assertEqual(manifest["legacy_validation"]["rows"], 2)
            for name in ("curriculum.py", "rules.py", "engine.py", "search.py"):
                self.assertTrue(any(Path(path).name == name for path in manifest["inputs"]))
            self.assertTrue(all(ablation.file_sha(path) == digest for path, digest in inputs.items()))
            with self.assertRaisesRegex(ValueError, "new directory"):
                ablation.run_ablation(self.args(root))

    def test_execution_reloads_equal_states_enforces_budgets_and_saves_only_diagnostics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            seen = []

            def fitting(model, rows, **options):
                self.assertTrue((root / "out/manifest.json").exists())
                seen.append((ablation.tree_sha(model.state_dict()),
                             ablation.tree_sha(options["optimizer"].state_dict()), options["seed"],
                             options["max_examples"], [item.state for item in rows]))
                return fake_fit(model, rows, **options)

            with patch.object(ablation, "fit", side_effect=fitting), \
                    patch.object(ablation, "evaluate_bounded", side_effect=mock_metrics):
                report = ablation.run_ablation(self.args(root, "--execute"))
            self.assertTrue(report["complete"])
            self.assertTrue(report["comparison_eligible"])
            self.assertTrue(report["training_performed"])
            self.assertFalse(report["promoted"])
            self.assertIsNone(report["winner"])
            self.assertEqual(len(seen), 4)
            self.assertEqual(len({item[0] for item in seen}), 1)
            self.assertEqual(len({item[1] for item in seen}), 1)
            self.assertEqual([item[2] for item in seen], [17, 17, 19, 19])
            self.assertEqual([item[3] for item in seen], [8] * 4)
            self.assertTrue(all(item[4] == seen[0][4] for item in seen))
            self.assertEqual(len(report["comparison"]["paired"]), 2)
            self.assertEqual(report["comparison"]["direction"], "B minus A")
            self.assertIn("legacy_policy_retention", report["comparison"]["mean_deltas"])
            for arm in report["arms"]:
                _, checkpoint = load_checkpoint(root / "out" / arm["checkpoint"])
                self.assertFalse(checkpoint["metadata"]["resumable"])
                self.assertNotIn("optimizer_state", checkpoint)
                self.assertNotIn("curriculum_state", checkpoint["metadata"])
                self.assertTrue(all(value == 9 for value in arm["final_optimizer_steps"].values()))

    def test_real_tiny_fit_enforces_masks_and_reports_separate_legacy_policy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            report = ablation.run_ablation(self.args(root, "--execute", "--steps", "1",
                                                     "--batch-size", "8", "--seeds", "17"))
            self.assertTrue(report["comparison_eligible"])
            for entry in [report["baseline"], *report["arms"]]:
                self.assertEqual(entry["primary_selfplay"]["overall"]["all"]["samples"], 4)
                self.assertEqual(entry["legacy_diagnostic"]["overall"]["all"]["samples"], 2)
                self.assertEqual(entry["validation_diagnostic"]["overall"]["all"]["samples"], 6)
                for group in ("teacher", "neural"):
                    self.assertEqual(entry["legacy_diagnostic"]["pooled_policy"][group]["non_forced_samples"], 1)
            for arm in report["arms"]:
                self.assertEqual(arm["training"]["trained_examples"], 8)
                self.assertEqual(arm["training"]["trained_value_examples"], 4)
                self.assertEqual(arm["training"]["steps"], 1)
                self.assertTrue(all(value == 8 for value in arm["final_optimizer_steps"].values()))

    def test_pooled_policy_weights_nonforced_rows_instead_of_source_counts(self):
        def source(count, nll):
            return {"non_forced": {"samples": count, "policy": {
                "soft_target_nll": nll, "top1_agreement": .5, "top5_agreement": .75}}}

        metrics = {"by_source": {"ghq-cpu-4": source(2, 8), "mcts:first": source(3, 1),
                                  "mcts:second": source(1, 5), "mcts:forced-only": source(0, None)}}
        pooled = ablation.pooled_policy_metrics(metrics)
        self.assertEqual(pooled["teacher"]["soft_target_nll"], 8)
        self.assertEqual(pooled["neural"]["soft_target_nll"], 2)
        self.assertEqual(pooled["neural"]["non_forced_samples"], 4)

    def test_missing_legacy_panel_is_rejected_before_evaluation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, validation = fixture(root)
            write_replay(root / "validation.jsonl", validation[:4])
            with patch.object(ablation, "evaluate_bounded", side_effect=AssertionError("must not infer")), \
                    self.assertRaisesRegex(ValueError, "legacy validation"):
                ablation.run_ablation(self.args(root))

    def test_rejects_insufficient_current_selfplay_unknown_models_and_weak_contrast(self):
        for flag, value in (("--min-training-games", "5"), ("--min-training-families", "5"),
                            ("--min-validation-games", "5"), ("--min-validation-families", "5"),
                            ("--min-training-rows", "5"), ("--min-validation-rows", "5"),
                            ("--model-id", "unknown-model")):
            with self.subTest(flag=flag), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture(root)
                args = self.args(root, flag, value)
                if flag == "--model-id":
                    args.model_id = [value]
                with self.assertRaisesRegex(ValueError, "Insufficient eligible"):
                    ablation.run_ablation(args)
                self.assertFalse(json.loads((root / "out/report.json").read_text())["comparison_eligible"])
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training, _ = fixture(root)
            for item in training:
                item.value_provenance = provenance()
            write_replay(root / "training.jsonl", training)
            with self.assertRaisesRegex(ValueError, "control contrast"):
                ablation.run_ablation(self.args(root, "--min-control-contrast", "0.2"))

    def test_game_family_or_registry_leakage_and_missing_identity_are_rejected(self):
        for issue in ("family", "game", "registry", "missing"):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                training, validation = fixture(root)
                if issue == "family":
                    validation[0].family_id = training[0].family_id
                elif issue == "game":
                    validation[0].game_id = training[0].game_id
                elif issue == "registry":
                    board = e.BaseBoard()
                    board.push(e.Move.from_uci("rtd1"))
                    validation[0] = row(board, "new-unregistered")
                else:
                    validation[0].family_id = ""
                write_replay(root / "validation.jsonl", validation)
                with self.assertRaises(ValueError):
                    ablation.run_ablation(self.args(root))

    def test_incorrect_value_exposure_invalidates_comparison(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)

            def fitting(model, rows, **options):
                optimizer, report = fake_fit(model, rows, **options)
                report["trained_value_examples"] -= 1
                return optimizer, report

            with patch.object(ablation, "fit", side_effect=fitting), \
                    patch.object(ablation, "evaluate_bounded", side_effect=mock_metrics), \
                    self.assertRaisesRegex(ValueError, "exposure"):
                ablation.run_ablation(self.args(root, "--execute"))
            report = json.loads((root / "out/report.json").read_text())
            self.assertFalse(report["comparison_eligible"])
            self.assertIsNone(report["comparison"])

    def test_changed_input_at_completion_and_deadline_cannot_produce_comparison(self):
        for problem in ("input", "deadline"):
            with self.subTest(problem=problem), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture(root)

                def evaluation(*args, **kwargs):
                    if problem == "deadline":
                        raise ablation.AblationDeadline("test deadline")
                    with (root / "training.jsonl").open("a") as stream:
                        stream.write("\n")
                    return mock_metrics()

                with patch.object(ablation, "fit", side_effect=fake_fit), \
                        patch.object(ablation, "evaluate_bounded", side_effect=evaluation):
                    if problem == "input":
                        with self.assertRaisesRegex(ValueError, "Input changed"):
                            ablation.run_ablation(self.args(root, "--execute"))
                        report = json.loads((root / "out/report.json").read_text())
                        self.assertFalse(report["inputs_unchanged"])
                    else:
                        report = ablation.run_ablation(self.args(root, "--execute"))
                        self.assertEqual(report["status"], "incomplete-deadline")
                self.assertFalse(report["comparison_eligible"])
                self.assertTrue(all(not arm["usable_for_comparison"] for arm in report["arms"]))

    def test_argument_validation_defaults_and_bounds(self):
        arguments = []
        for name in ("checkpoint", "training", "validation", "partition", "output"):
            arguments.extend(("--" + name, "/unused/" + name))
        args = ablation.parser().parse_args(arguments + ["--model-id", "sha"])
        self.assertFalse(args.execute)
        self.assertEqual(args.seeds, [20261125, 20261126, 20261127])
        self.assertEqual(args.min_control_contrast, .2)
        self.assertEqual(args.min_training_families, 16)
        for field, value in (("seeds", [1, 1]), ("seeds", [-1]), ("model_id", [""]),
                             ("steps", 0), ("learning_rate", float("nan")), ("max_seconds", 0),
                             ("min_control_contrast", 1.1), ("min_validation_rows", 8193)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                invalid = deepcopy(args)
                setattr(invalid, field, value)
                ablation.validate_args(invalid)


if __name__ == "__main__":
    unittest.main()
