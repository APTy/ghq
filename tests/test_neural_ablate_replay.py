import copy
import json
import math
from pathlib import Path
import random
import struct
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Ablation tests require torch")

from ghqnet import ablate_replay as ablation
from ghqnet.curriculum import fingerprint
from ghqnet.data import Sample, write_replay
from ghqnet.model import NetworkConfig, PolicyValueNet, load_checkpoint, save_checkpoint
from ghqnet.rules import e


def sample(board, family):
    actions = sorted(move.uci() for move in board.generate_legal_moves())
    return Sample(board.serialize(), actions, [float(action == "skip") for action in actions],
                  outcome=0, game_id=family, family_id=family, source="ghq-cpu-4")


def fixture(root):
    board = e.BaseBoard()
    training = [sample(board, "train"), sample(board, "train")]
    board.push(e.Move.from_uci("rhe1"))
    validation = [sample(board, "validation")]
    replay, heldout, state = (root / name for name in
                             ("replay-buffer-0011.jsonl", "validation-buffer-0011.jsonl", "curriculum-0011.json"))
    write_replay(replay, training, kind="training")
    write_replay(heldout, validation, kind="validation")
    state.write_text(json.dumps({"history": [], "partition": {
        "train_states": sorted({fingerprint(row) for row in training}),
        "validation_states": sorted({fingerprint(row) for row in validation})}}))
    model = PolicyValueNet(NetworkConfig(blocks=0, channels=4))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=1e-4)
    parameter = next(model.parameters())
    optimizer.state[parameter] = {"step": torch.tensor(7.), "exp_avg": torch.zeros_like(parameter),
                                  "exp_avg_sq": torch.ones_like(parameter)}
    checkpoint = root / "checkpoint-0011.pt"
    metadata = {"iteration": 11, "seed": 20261121}
    for key, path in (("replay_buffer", replay), ("validation_buffer", heldout), ("curriculum_state", state)):
        metadata.update({key: path.name, key + "_sha256": ablation.file_sha(path)})
    save_checkpoint(checkpoint, model, optimizer, **metadata)
    return checkpoint, training, validation


def mock_metrics(*args, **kwargs):
    group = {"all": {"wdl": {"logloss": .5, "brier": .3, "accuracy": .7}},
             "non_forced": {"policy": {"top1_agreement": .3, "top5_agreement": .6, "soft_target_nll": 2.5}}}
    return {"overall": copy.deepcopy(group), "by_source": {"ghq-cpu-4": copy.deepcopy(group)}}


def advance_optimizer(optimizer):
    for state in optimizer.state.values():
        state["step"] += 1


class AblationArgumentTests(unittest.TestCase):
    def args(self, extra=()):
        return ablation.parser().parse_args(["/missing/checkpoint.pt", "--output", "/unused", *extra])

    def test_invalid_or_unpaired_rates_and_bounds(self):
        for rates in ([.001], [.001, .001], [.001, 0], [.001, math.nan], [.001, math.inf], [.001, 1.1]):
            args = self.args()
            args.learning_rates = rates
            with self.subTest(rates=rates), self.assertRaises(ValueError):
                ablation.validate_args(args)
        for name, value in (("epochs", 0), ("epochs", 5), ("batch_size", 513), ("threads", 65),
                            ("seed", -1), ("max_seconds", math.inf), ("max_seconds", 0)):
            args = self.args()
            setattr(args, name, value)
            with self.subTest(field=name), self.assertRaises(ValueError):
                ablation.validate_args(args)

    def test_validate_only_never_loads_or_trains(self):
        with patch.object(ablation, "run_ablation", side_effect=AssertionError("must not run")):
            self.assertEqual(ablation.main(["/missing/checkpoint.pt", "--output", "/unused", "--validate-only"]), 0)

    def test_shuffle_identity_matches_fit_algorithm(self):
        import hashlib
        expected = hashlib.sha256()
        rng, order = random.Random(41), list(range(7))
        for _ in range(3):
            rng.shuffle(order)
            for index in order:
                expected.update(struct.pack("<Q", index))
        self.assertEqual(ablation.shuffled_order_sha(7, 3, 41), expected.hexdigest())
        self.assertNotEqual(ablation.shuffled_order_sha(7, 3, 42), expected.hexdigest())

    def test_explicit_lr_override_preserves_every_other_optimizer_field(self):
        model = torch.nn.Linear(1, 1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=.001)
        parameter = next(model.parameters())
        optimizer.state[parameter] = {"step": torch.tensor(8.), "exp_avg": torch.ones_like(parameter),
                                      "exp_avg_sq": torch.ones_like(parameter) * 2}
        before = copy.deepcopy(optimizer.state_dict())
        restored, report = ablation.make_optimizer(model, before, .0003)
        after = restored.state_dict()
        self.assertEqual(report["restored_optimizer_sha256"], ablation.tree_sha(before))
        self.assertEqual(after["param_groups"][0]["lr"], .0003)
        after["param_groups"][0]["lr"] = .001
        self.assertEqual(ablation.tree_sha(after), ablation.tree_sha(before))
        self.assertEqual(before["param_groups"][0]["lr"], .001)


class AblationRunnerTests(unittest.TestCase):
    def args(self, checkpoint, output):
        return ablation.parser().parse_args([str(checkpoint), "--output", str(output), "--device", "cpu",
                                             "--threads", "1", "--batch-size", "2", "--max-seconds", "60"])

    def test_equal_initial_states_and_budgets_with_diagnostic_outputs(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint, training, validation = fixture(root)
            identities = {path: ablation.file_sha(path) for path in root.iterdir()}
            seen = []

            def fake_fit(model, rows, **options):
                optimizer = options["optimizer"]
                seen.append({"model": ablation.tree_sha(model.state_dict()),
                             "state": ablation.tree_sha(optimizer.state_dict()["state"]),
                             "rate": optimizer.param_groups[0]["lr"], "seed": options["seed"],
                             "rows": [row.state for row in rows], "validation": options["validation_samples"]})
                with torch.no_grad():
                    next(model.parameters()).add_(optimizer.param_groups[0]["lr"])
                advance_optimizer(optimizer)
                return optimizer, {"steps": 1, "interrupted": False, "samples": len(rows)}

            with patch.object(ablation, "fit", side_effect=fake_fit), patch.object(ablation, "evaluate_bounded", side_effect=mock_metrics):
                report = ablation.run_ablation(self.args(checkpoint, root / "out"))
            self.assertTrue(report["complete"])
            self.assertTrue(report["comparison_eligible"])
            self.assertIsNone(report["winner"])
            self.assertEqual(len(seen), 2)
            self.assertEqual(seen[0]["model"], seen[1]["model"])
            self.assertEqual(seen[0]["state"], seen[1]["state"])
            self.assertEqual([row["rate"] for row in seen], [.001, .0003])
            self.assertEqual(seen[0]["rows"], seen[1]["rows"])
            self.assertEqual(seen[0]["seed"], seen[1]["seed"])
            self.assertEqual([row["validation"] for row in seen], [[], []])
            self.assertTrue(all(ablation.file_sha(path) == digest for path, digest in identities.items()))
            for arm in report["arms"]:
                _, payload = load_checkpoint(root / "out" / arm["checkpoint"], "cpu")
                self.assertFalse(payload["metadata"]["resumable"])
                self.assertNotIn("replay_buffer", payload["metadata"])
                self.assertNotIn("curriculum_state", payload["metadata"])
                self.assertEqual(payload["optimizer_state"]["param_groups"][0]["lr"], arm["learning_rate"])
            events = [json.loads(line)["event"] for line in (root / "out/events.jsonl").read_text().splitlines()]
            self.assertEqual(events[-1], "comparison-complete")
            self.assertEqual(events.count("arm-trained"), 2)

    def test_real_tiny_fit_and_save_load_preserve_isolated_arm_starts(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint, _, _ = fixture(root)
            before = ablation.file_sha(checkpoint)
            report = ablation.run_ablation(self.args(checkpoint, root / "out"))
            self.assertTrue(report["complete"])
            self.assertEqual(report["expected_steps_per_arm"], 1)
            self.assertEqual([arm["training"]["steps"] for arm in report["arms"]], [1, 1])
            self.assertEqual([arm["initial_optimizer_steps"]["0"] for arm in report["arms"]], [7, 7])
            self.assertEqual([arm["final_optimizer_steps"]["0"] for arm in report["arms"]], [8, 8])
            self.assertEqual(len({arm["initial_model_sha256"] for arm in report["arms"]}), 1)
            self.assertEqual(len({arm["restored_optimizer_sha256"] for arm in report["arms"]}), 1)
            self.assertEqual(len({arm["checkpoint_sha256"] for arm in report["arms"]}), 2)
            self.assertEqual(ablation.file_sha(checkpoint), before)
            for arm in report["arms"]:
                model, payload = load_checkpoint(root / "out" / arm["checkpoint"], "cpu")
                self.assertFalse(payload["metadata"]["resumable"])
                self.assertEqual(arm["validation"]["overall"]["all"]["samples"], 1)

    def test_gradient_preflight_is_discardable_and_advances_one_step(self):
        import time
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint, rows, _ = fixture(root)
            model, payload = load_checkpoint(checkpoint, "cpu")
            identity = ablation.tree_sha(model.state_dict())
            old_threads = torch.get_num_threads()
            try:
                torch.set_num_threads(1)
                report = ablation.gradient_preflight(model, payload["optimizer_state"], rows,
                    device="cpu", rate=.001, seed=42, batch_size=2, deadline=time.perf_counter() + 60)
            finally:
                torch.set_num_threads(old_threads)
            self.assertTrue(report["complete"])
            self.assertTrue(report["discarded_before_baseline_and_arms"])
            self.assertNotEqual(ablation.tree_sha(model.state_dict()), identity)
            restored, payload = load_checkpoint(checkpoint, "cpu")
            self.assertEqual(ablation.tree_sha(restored.state_dict()), identity)
            self.assertEqual(float(payload["optimizer_state"]["state"][0]["step"]), 7)

    def test_incomplete_optimizer_budget_cannot_compare_or_write_arm(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint, _, _ = fixture(root)
            def incomplete(model, rows, **options):
                return options["optimizer"], {"steps": 0, "interrupted": True}
            with patch.object(ablation, "fit", side_effect=incomplete), patch.object(ablation, "evaluate_bounded", side_effect=mock_metrics):
                report = ablation.run_ablation(self.args(checkpoint, root / "out"))
            self.assertFalse(report["complete"])
            self.assertFalse(report["comparison_eligible"])
            self.assertIsNone(report["comparison"])
            self.assertTrue(report["inputs_unchanged"])
            self.assertEqual(list((root / "out").glob("arm-*.pt")), [])

    def test_evaluation_deadline_after_training_never_marks_arm_usable(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint, _, _ = fixture(root)
            def complete(model, rows, **options):
                advance_optimizer(options["optimizer"])
                return options["optimizer"], {"steps": 1, "interrupted": False}
            with patch.object(ablation, "fit", side_effect=complete), patch.object(ablation, "evaluate_bounded",
                    side_effect=[mock_metrics(), ablation.AblationDeadline("evaluation deadline")]):
                report = ablation.run_ablation(self.args(checkpoint, root / "out"))
            self.assertFalse(report["complete"])
            self.assertIsNone(report["comparison"])
            self.assertFalse(report["arms"][0]["usable_for_comparison"])

    def test_tampered_input_rejected_before_fit(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint, _, _ = fixture(root)
            with (root / "replay-buffer-0011.jsonl").open("a") as stream:
                stream.write("\n")
            with patch.object(ablation, "fit", side_effect=AssertionError("must not train")), self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                ablation.run_ablation(self.args(checkpoint, root / "out"))
            self.assertEqual(json.loads((root / "out/report.json").read_text())["status"], "failed")

    def test_final_hash_check_detects_input_mutation(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint, _, _ = fixture(root)
            def mutate(model, rows, **options):
                with (root / "replay-buffer-0011.jsonl").open("a") as stream:
                    stream.write("\n")
                advance_optimizer(options["optimizer"])
                return options["optimizer"], {"steps": 1, "interrupted": False}
            with patch.object(ablation, "fit", side_effect=mutate), patch.object(ablation, "evaluate_bounded", side_effect=mock_metrics), self.assertRaisesRegex(ValueError, "Input changed"):
                ablation.run_ablation(self.args(checkpoint, root / "out"))
            report = json.loads((root / "out/report.json").read_text())
            self.assertFalse(report["comparison_eligible"])
            self.assertIsNone(report["comparison"])

    def test_fabricated_complete_step_report_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint, _, _ = fixture(root)
            def no_steps(model, rows, **options):
                return options["optimizer"], {"steps": 1, "interrupted": False}
            with patch.object(ablation, "fit", side_effect=no_steps), patch.object(ablation, "evaluate_bounded", side_effect=mock_metrics), self.assertRaisesRegex(ValueError, "Actual Adam step"):
                ablation.run_ablation(self.args(checkpoint, root / "out"))
            self.assertFalse(json.loads((root / "out/report.json").read_text())["comparison_eligible"])

    def test_validation_does_not_create_gradients_or_update_parameters(self):
        class ConstantModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = torch.nn.Parameter(torch.tensor(1.))

            def forward(self, boards, actions, mask):
                if torch.is_grad_enabled():
                    raise AssertionError("Validation enabled gradients")
                return (self.weight.expand(mask.shape).masked_fill(~mask, -torch.inf),
                        self.weight.expand(len(boards), 3))

        model = ConstantModel()
        before = ablation.tree_sha(model.state_dict())
        metrics = ablation.evaluate_bounded(model, [sample(e.BaseBoard(), "validation")],
                                             device="cpu", batch_size=2, deadline=None)
        self.assertIsNone(model.weight.grad)
        self.assertEqual(ablation.tree_sha(model.state_dict()), before)
        self.assertEqual(metrics["overall"]["all"]["samples"], 1)

    def test_refuses_existing_output_and_overlapping_partition(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint, train, validation = fixture(root)
            with self.assertRaisesRegex(ValueError, "new directory"):
                ablation.run_ablation(self.args(checkpoint, root))
            state = {"partition": {"train_states": [fingerprint(train[0])], "validation_states": [fingerprint(train[0])]}}
            with self.assertRaisesRegex(ValueError, "overlapping"):
                ablation.verify_partition(train, validation, state, None)


if __name__ == "__main__":
    unittest.main()
