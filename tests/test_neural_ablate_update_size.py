"""Gentler updates preserve data exposure and restore weights/Adam independently."""
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
    raise unittest.SkipTest("Update-size tests require torch")

from ghqnet import ablate_update_size as ablation
from ghqnet.curriculum import fingerprint
from ghqnet.data import Sample, write_replay
from ghqnet.model import NetworkConfig, PolicyValueNet, load_checkpoint, save_checkpoint
from ghqnet.rules import e


def fixture(root):
    terminal = e.BaseBoard()
    terminal.push(e.Move.from_uci("skip"))
    terminal.push(e.Move.from_uci("skip"))
    player = {"model_id": "model-sha", "search": {
        "kind": "neural", "simulations": 16, "cpuct": 1.5, "max_nodes": 256,
        "max_depth": 256, "root_noise": True, "dirichlet_alpha": .3,
        "noise_fraction": .25, "temperature_turns": 8, "temperature": 1., "late_temperature": 0.}}
    provenance = {"version": 1, "collection_kind": "self", "players": {"RED": deepcopy(player), "BLUE": player},
                  "terminal": {"winner": None, "reason": "double-skip"},
                  "termination_reason": "double-skip", "final_state": terminal.serialize()}

    def row(board, identity, eligible):
        actions = sorted(move.uci() for move in board.generate_legal_moves())
        return Sample(board.serialize(), actions, [float(action == "skip") for action in actions],
                      outcome=1, source="mcts:new" if eligible else "teacher", game_id=identity,
                      family_id=identity, value_provenance=deepcopy(provenance) if eligible else None)

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
    return {"overall": deepcopy(group), "by_source": {"teacher": deepcopy(group)}}


def fake_fit(model, rows, **options):
    optimizer = options["optimizer"]
    steps = options["max_examples"] // options["batch_size"]
    for state in optimizer.state.values():
        state["step"] += steps
    chosen = [row for row, include in zip(rows, options["value_mask"]) if include]
    with torch.no_grad():
        next(model.parameters()).add_(options["learning_rate"])
    return optimizer, {"steps": steps, "interrupted": False, "trained_examples": options["max_examples"],
                       "trained_value_examples": len(chosen),
                       "trained_value_examples_by_source": {source: sum(row.source == source for row in chosen)
                                                             for source in {row.source for row in chosen}}}


class UpdateSizeTests(unittest.TestCase):
    def args(self, root, *extra):
        return ablation.parser().parse_args([
            "--checkpoint", str(root / "checkpoint.pt"), "--training", str(root / "training.jsonl"),
            "--validation", str(root / "validation.jsonl"), "--partition", str(root / "partition.json"),
            "--output", str(root / "out"), "--model-id", "model-sha", "--seeds", "17", "19",
            "--steps", "2", "--batch-size", "4", "--threads", "1",
            "--min-training-games", "1", "--min-validation-games", "1",
            "--min-training-families", "1", "--min-validation-families", "1",
            "--min-training-rows", "1", "--min-validation-rows", "1", *extra])

    def test_plan_matches_seeded_prefix_without_repetition_or_ineligible_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            rows, _ = fixture(Path(directory))
            for steps in (1, 2):
                plan = ablation.make_seed_plan(rows, [index < 4 for index in range(8)], seed=17, steps=steps, batch_size=4)
                order = list(range(8))
                random.Random(17).shuffle(order)
                self.assertEqual(plan["order"], order[:steps * 4])
                self.assertEqual(plan["mask"], [index < 4 and index in plan["order"] for index in range(8)])
                self.assertEqual(sum(batch["value_labels"] for batch in plan["per_batch"]), plan["labels"]["rows"])
                self.assertEqual(plan, ablation.make_seed_plan(rows, [index < 4 for index in range(8)], seed=17,
                                                               steps=steps, batch_size=4))
            with self.assertRaisesRegex(ValueError, "no-repeat"):
                ablation.make_seed_plan(rows, [True] * 8, seed=17, steps=3, batch_size=4)
            with self.assertRaisesRegex(ValueError, "boolean"):
                ablation.make_seed_plan(rows, [1] * 8, seed=17, steps=1, batch_size=4)

    def test_preparation_freezes_inputs_mask_identity_without_inference_or_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            original = {path: ablation.file_sha(path) for path in root.iterdir()}
            with patch.object(ablation, "fit", side_effect=AssertionError("must not train")), \
                    patch.object(ablation, "evaluate_bounded", side_effect=AssertionError("must not infer")), \
                    patch.object(PolicyValueNet, "forward", side_effect=AssertionError("must not forward")):
                report = ablation.run_ablation(self.args(root))
            self.assertEqual(report["status"], "prepared")
            self.assertTrue(report["inputs_unchanged"])
            self.assertFalse(report["training_performed"])
            self.assertFalse(report["comparison_eligible"])
            manifest = json.loads((root / "out/manifest.json").read_text())
            self.assertEqual(manifest["configuration"]["learning_rates"], [.0003, .0001])
            self.assertEqual(manifest["primary_validation"]["rows"], 4)
            self.assertEqual(manifest["legacy_validation"]["rows"], 2)
            self.assertEqual(len(manifest["seeds"]), 2)
            self.assertTrue(all(ablation.file_sha(path) == digest for path, digest in original.items()))
            with self.assertRaisesRegex(ValueError, "new directory"):
                ablation.run_ablation(self.args(root))

    def test_each_rate_reloads_identical_weights_and_adam_except_rate_with_matched_exposure(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            seen = []

            def fitting(model, rows, **options):
                self.assertTrue((root / "out/manifest.json").exists())
                saved = deepcopy(options["optimizer"].state_dict())
                self.assertEqual(saved["param_groups"][0]["lr"], options["learning_rate"])
                saved["param_groups"][0]["lr"] = .001
                seen.append((ablation.tree_sha(model.state_dict()), ablation.tree_sha(saved),
                             options["seed"], options["max_examples"], options["value_mask"]))
                return fake_fit(model, rows, **options)

            with patch.object(ablation, "fit", side_effect=fitting), \
                    patch.object(ablation, "evaluate_bounded", side_effect=mock_metrics):
                report = ablation.run_ablation(self.args(root, "--execute"))
            self.assertTrue(report["comparison_eligible"])
            self.assertFalse(report["promoted"])
            self.assertIsNone(report["winner"])
            self.assertEqual(len(seen), 4)
            self.assertEqual(len({entry[0] for entry in seen}), 1)
            self.assertEqual(len({entry[1] for entry in seen}), 1)
            self.assertEqual([entry[2] for entry in seen], [17, 17, 19, 19])
            self.assertEqual([entry[3] for entry in seen], [8] * 4)
            self.assertTrue(all(entry[4] == [True] * 4 + [False] * 4 for entry in seen))
            self.assertEqual(len(report["comparison"]["rates"]), 2)
            for arm in report["arms"]:
                self.assertEqual(arm["training"]["trained_value_examples"], 4)
                self.assertTrue(all(step == 9 for step in arm["final_optimizer_steps"].values()))
                _, saved = load_checkpoint(root / "out" / arm["checkpoint"])
                self.assertFalse(saved["metadata"]["resumable"])
                self.assertNotIn("optimizer_state", saved)
                self.assertNotIn("curriculum_state", saved["metadata"])

    def test_real_small_fit_has_finite_separate_panels_and_exact_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture(root)
            report = ablation.run_ablation(self.args(root, "--execute", "--steps", "1", "--batch-size", "8", "--seeds", "17"))
            self.assertTrue(report["comparison_eligible"])
            for entry in [report["baseline"], *report["arms"]]:
                self.assertEqual(entry["primary_selfplay"]["overall"]["all"]["samples"], 4)
                self.assertEqual(entry["legacy_diagnostic"]["overall"]["all"]["samples"], 2)
                for group in ("teacher", "neural"):
                    self.assertEqual(entry["legacy_diagnostic"]["pooled_policy"][group]["non_forced_samples"], 1)
            for arm in report["arms"]:
                self.assertEqual(arm["training"]["steps"], 1)
                self.assertEqual(arm["training"]["trained_examples"], 8)
                self.assertEqual(arm["training"]["trained_value_examples"], 4)

    def test_bad_exposures_counters_or_nonfinite_training_do_not_become_eligible(self):
        for issue in ("policy", "value", "source", "counter", "model", "optimizer", "metrics"):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture(root)

                def fitting(model, rows, **options):
                    optimizer, metrics = fake_fit(model, rows, **options)
                    if issue == "policy":
                        metrics["trained_examples"] -= 1
                    elif issue == "value":
                        metrics["trained_value_examples"] -= 1
                    elif issue == "source":
                        metrics["trained_value_examples_by_source"] = {"wrong": 4}
                    elif issue == "counter":
                        next(iter(optimizer.state.values()))["step"] += 1
                    elif issue == "model":
                        with torch.no_grad():
                            next(model.parameters()).fill_(float("nan"))
                    elif issue == "optimizer":
                        next(iter(optimizer.state.values()))["exp_avg"].fill_(float("inf"))
                    else:
                        metrics["seconds"] = float("nan")
                    return optimizer, metrics

                with patch.object(ablation, "fit", side_effect=fitting), \
                        patch.object(ablation, "evaluate_bounded", side_effect=mock_metrics), self.assertRaises(ValueError):
                    ablation.run_ablation(self.args(root, "--execute"))
                report = json.loads((root / "out/report.json").read_text())
                self.assertFalse(report["comparison_eligible"])
                self.assertIsNone(report["comparison"])

    def test_nonfinite_eval_input_changes_manifest_changes_and_deadline_fail_closed(self):
        for issue in ("nan", "input", "manifest", "deadline"):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture(root)

                def evaluation(*args, **kwargs):
                    metrics = mock_metrics()
                    if issue == "nan":
                        metrics["overall"]["all"]["wdl"]["logloss"] = float("nan")
                    elif issue in ("input", "manifest"):
                        path = root / ("training.jsonl" if issue == "input" else "out/manifest.json")
                        with path.open("a") as stream:
                            stream.write("\n")
                    else:
                        raise ablation.AblationDeadline("test deadline")
                    return metrics

                with patch.object(ablation, "fit", side_effect=fake_fit), \
                        patch.object(ablation, "evaluate_bounded", side_effect=evaluation):
                    if issue == "deadline":
                        report = ablation.run_ablation(self.args(root, "--execute"))
                        self.assertEqual(report["status"], "incomplete-deadline")
                    else:
                        with self.assertRaises(ValueError):
                            ablation.run_ablation(self.args(root, "--execute"))
                        report = json.loads((root / "out/report.json").read_text())
                self.assertFalse(report["comparison_eligible"])
                self.assertTrue(all(not arm["usable_for_comparison"] for arm in report["arms"]))

    def test_unknown_provenance_missing_legacy_and_partition_overlap_reject_before_training(self):
        for issue in ("unknown", "legacy", "family", "game", "registry"):
            with self.subTest(issue=issue), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                training, validation = fixture(root)
                args = self.args(root)
                if issue == "unknown":
                    args.model_id = ["unknown"]
                elif issue == "legacy":
                    validation = validation[:4]
                elif issue == "family":
                    validation[0].family_id = training[0].family_id
                elif issue == "game":
                    validation[0].game_id = training[0].game_id
                else:
                    state = json.loads((root / "partition.json").read_text())
                    state["partition"]["validation_states"] = state["partition"]["train_states"]
                    (root / "partition.json").write_text(json.dumps(state))
                write_replay(root / "validation.jsonl", validation)
                with patch.object(ablation, "fit", side_effect=AssertionError("must not train")), self.assertRaises(ValueError):
                    ablation.run_ablation(args)

    def test_defaults_and_invalid_argument_bounds(self):
        args = self.args(Path("/unused"))
        self.assertEqual(args.learning_rates, [.0003, .0001])
        self.assertFalse(args.execute)
        for field, value in (("seeds", [1, 1]), ("learning_rates", []), ("learning_rates", [.001, .001]),
                             ("learning_rates", [float("nan")]), ("learning_rates", [0]),
                             ("learning_rates", [.0001, .000100000000000000001]),
                             ("model_id", [""]), ("steps", 0), ("max_seconds", 0),
                             ("min_validation_rows", 8193), ("min_training_rows", 9)):
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                invalid = deepcopy(args)
                setattr(invalid, field, value)
                ablation.validate_args(invalid)


if __name__ == "__main__":
    unittest.main()
