from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
except ImportError:
    raise unittest.SkipTest("Neural training tests require numpy and torch")

from ghqnet.data import Sample, label_game, read_replay, write_replay
from ghqnet.model import NetworkConfig, PolicyValueNet, load_checkpoint, save_checkpoint
from ghqnet.rules import e
from ghqnet.selfplay import play_game
from ghqnet.train import batch, fit, losses, prepare
from ghqnet import train


def sample_from(board, outcome=None):
    actions = sorted(move.uci() for move in board.generate_legal_moves())
    return Sample(board.serialize(), actions, [float(action == "skip") for action in actions],
                  outcome=outcome, game_id="test", source="test")


def tiny_samples():
    board = e.BaseBoard()
    first = sample_from(board, 0)
    board.push(e.Move.reinforce(e.INFANTRY, e.A1))
    return [first, sample_from(board, 0)]


class NeuralTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(313)

    def test_labels_follow_each_states_player_including_same_player_actions(self):
        red = e.BaseBoard()
        red_midturn = red.copy()
        red_midturn.push(e.Move.reinforce(e.INFANTRY, e.A1))
        blue = red.mirror()
        blue_midturn = red_midturn.mirror()
        samples = [sample_from(board) for board in (red, red_midturn, blue, blue_midturn)]
        label_game(samples, {"winner": "RED", "reason": "hq-capture"})
        self.assertEqual([sample.outcome for sample in samples], [0, 0, 2, 2])
        label_game(samples, {"winner": "BLUE", "reason": "hq-capture"})
        self.assertEqual([sample.outcome for sample in samples], [2, 2, 0, 0])
        label_game(samples, {"winner": None, "reason": "double-skip"})
        self.assertEqual([sample.outcome for sample in samples], [1, 1, 1, 1])
        label_game(samples, None)
        self.assertEqual([sample.outcome for sample in samples], [None] * 4)

    def test_truncated_actual_teacher_game_retains_policy_without_value_labels(self):
        samples, result = play_game(teacher=True, teacher_ms=0, max_turns=1, seed=11)
        self.assertTrue(samples)
        self.assertTrue(result["truncated"])
        self.assertIsNone(result["terminal"])
        self.assertEqual(result["reason"], "max-turns")
        self.assertTrue(all(sample.outcome is None for sample in samples))
        for sample in samples:
            sample.validate()
            self.assertEqual(sum(sample.policy), 1)
        model = PolicyValueNet(NetworkConfig(blocks=0, channels=8))
        policy, value = losses(model, batch(prepare(samples), "cpu"))
        self.assertTrue(torch.isfinite(policy))
        self.assertEqual(value.item(), 0)

    def test_replay_roundtrip_rejects_malformed_targets_and_legal_lists(self):
        good = sample_from(e.BaseBoard(), 1)
        bad = [replace(good, actions=good.actions[:-1], policy=good.policy[:-1]),
               replace(good, actions=good.actions + [good.actions[0]], policy=good.policy + [0]),
               replace(good, policy=[0.0] * len(good.actions)),
               replace(good, policy=[float("nan")] + good.policy[1:]),
               replace(good, policy=[-1.0] + good.policy[1:]),
               replace(good, policy=good.policy[:-1]), replace(good, outcome=3)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.jsonl"
            write_replay(path, [good], seed=13)
            self.assertEqual(read_replay(path), [good])
            for sample in bad:
                with self.subTest(sample=sample):
                    write_replay(path, [sample])
                    with self.assertRaises(ValueError):
                        read_replay(path)
            write_replay(path, [good])
            lines = path.read_text().splitlines()
            header = json.loads(lines[0])
            header["rules_sha256"] = "different rules"
            lines[0] = json.dumps(header)
            path.write_text("\n".join(lines))
            with self.assertRaisesRegex(ValueError, "rules"):
                read_replay(path)

    def test_variable_legal_actions_padding_has_finite_loss_and_gradients(self):
        samples = tiny_samples()
        samples[1].outcome = None
        records = prepare(samples)
        self.assertNotEqual(len(records[0][1]), len(records[1][1]))
        tensors = batch(records, "cpu")
        self.assertTrue((~tensors[2]).any())
        model = PolicyValueNet(NetworkConfig(blocks=1, channels=8))
        policy, value = losses(model, tensors)
        self.assertTrue(torch.isfinite(policy + value))
        (policy + value).backward()
        self.assertTrue(all(torch.isfinite(parameter.grad).all() for parameter in model.parameters()
                            if parameter.grad is not None))

    def test_small_actual_positions_learn_policy_and_outcome(self):
        model = PolicyValueNet(NetworkConfig(blocks=0, channels=8))
        _, report = fit(model, tiny_samples(), epochs=16, batch_size=2,
                        learning_rate=0.01, device="cpu", seed=17)
        self.assertEqual(report["value_labels"], 2)
        self.assertTrue(np.isfinite(report["after"]["total"]))
        self.assertLess(report["after"]["policy"], report["before"]["policy"] * 0.5)
        self.assertLess(report["after"]["value"], report["before"]["value"] * 0.5)

    def test_checkpoint_optimizer_resume_matches_uninterrupted_training(self):
        model = PolicyValueNet(NetworkConfig(blocks=0, channels=8))
        reference = deepcopy(model)
        # Hold batch order fixed: this checks optimizer continuation, not
        # whether separate fit calls restore a data-shuffling RNG stream.
        samples = tiny_samples()[:1]
        fit(reference, samples, epochs=2, batch_size=1, learning_rate=0.005, seed=1)
        optimizer, _ = fit(model, samples, epochs=1, batch_size=1, learning_rate=0.005, seed=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            save_checkpoint(path, model, optimizer, step=1)
            resumed, checkpoint = load_checkpoint(path)
            resumed_optimizer = torch.optim.AdamW(resumed.parameters())
            resumed_optimizer.load_state_dict(checkpoint["optimizer_state"])
            fit(resumed, samples, epochs=1, batch_size=1, seed=1, optimizer=resumed_optimizer)
            for name, parameter in resumed.state_dict().items():
                torch.testing.assert_close(parameter, reference.state_dict()[name], rtol=1e-5, atol=1e-7)
            self.assertTrue(all(state["step"].item() == 2 for state in resumed_optimizer.state.values()))

    def test_zero_example_budget_is_noop_without_preparation_or_model_execution(self):
        model = PolicyValueNet(NetworkConfig(blocks=0, channels=4))
        optimizer, _ = fit(model, tiny_samples(), epochs=1, batch_size=2)
        before, saved = deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())
        model.train()
        with patch.object(train, "prepare", side_effect=AssertionError("must not prepare")), \
                patch.object(model, "forward", side_effect=AssertionError("must not execute")):
            returned, report = fit(model, tiny_samples(), epochs=5, optimizer=optimizer,
                                   max_examples=0, validation_samples=tiny_samples())
        self.assertIs(returned, optimizer)
        self.assertTrue(model.training)
        self.assertEqual((report["planned_examples"], report["trained_examples"], report["steps"]), (0, 0, 0))
        self.assertEqual(report["epochs_completed"], 0)
        self.assertEqual(report["stopped_reason"], "example_budget")
        self.assertTrue(report["example_budget_exhausted"])
        self.assertFalse(report["interrupted"])
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        self.assertEqual(optimizer.state_dict()["param_groups"], saved["param_groups"])
        for index, state in optimizer.state_dict()["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(value, saved["state"][index][key], rtol=0, atol=0)
        _, empty = fit(model, [], max_examples=0)
        self.assertEqual(empty["trained_examples"], 0)

    def test_partial_batch_from_resumed_optimizer_matches_explicit_prefix(self):
        model = PolicyValueNet(NetworkConfig(blocks=0, channels=4))
        optimizer, _ = fit(model, tiny_samples(), epochs=1, batch_size=2)
        saved_optimizer = deepcopy(optimizer.state_dict())
        samples = [replace(tiny_samples()[i % 2], game_id=str(i)) for i in range(7)]
        indices = list(range(len(samples)))
        random.Random(83).shuffle(indices)
        selected = [samples[i] for i in indices[:3]]
        # Unselected rows are deliberately invalid: a small update must neither
        # encode nor validate the rest of a large replay buffer.
        for index in indices[3:]:
            samples[index] = replace(samples[index], policy=[])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "resume.pt"
            save_checkpoint(path, model, optimizer)
            resumed, checkpoint = load_checkpoint(path)
            resumed_optimizer = torch.optim.AdamW(resumed.parameters())
            resumed_optimizer.load_state_dict(checkpoint["optimizer_state"])
            _, report = fit(resumed, samples, epochs=4, batch_size=2, seed=83,
                            optimizer=resumed_optimizer, max_examples=3, max_seconds=30)
        reference_optimizer = torch.optim.AdamW(model.parameters())
        reference_optimizer.load_state_dict(saved_optimizer)
        records = prepare(selected)
        model.train()
        for start in (0, 2):
            reference_optimizer.zero_grad(set_to_none=True)
            policy, value = losses(model, batch(records[start:start + 2], "cpu"))
            (policy + value).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, error_if_nonfinite=True)
            reference_optimizer.step()
        self.assertEqual((report["planned_examples"], report["trained_examples"], report["steps"]), (3, 3, 2))
        self.assertEqual(report["epochs_completed"], 0)
        self.assertEqual(report["training_measurement_scope"], "shuffled_prefix")
        self.assertFalse(report["interrupted"])
        for key, value in resumed.state_dict().items():
            torch.testing.assert_close(value, model.state_dict()[key], rtol=0, atol=0)
        self.assertTrue(all(state["step"].item() == 3 for state in resumed_optimizer.state.values()))

    def recorded_fit(self, *, count=5, **options):
        """Observe training row order while retaining real Adam step calls."""
        rows = [replace(tiny_samples()[0], game_id=str(i)) for i in range(count)]
        model = PolicyValueNet(NetworkConfig(blocks=0, channels=4))
        observed = []

        def fake_prepare(samples, deadline=None):
            return [(row.game_id, None, None, row.outcome) for row in samples]

        def fake_losses(model, records):
            observed.append([row[0] for row in records])
            parameter = next(model.parameters())
            return parameter.square().mean(), parameter.sum() * 0

        with patch.object(train, "prepare", side_effect=fake_prepare), \
                patch.object(train, "batch", side_effect=lambda rows, device: rows), \
                patch.object(train, "losses", side_effect=fake_losses):
            _, report = fit(model, rows, max_seconds=30, **options)
        return observed, report

    def test_cap_across_epochs_preserves_uncapped_shuffle_prefix(self):
        capped, report = self.recorded_fit(epochs=3, batch_size=3, seed=71, max_examples=8)
        uncapped, full = self.recorded_fit(epochs=3, batch_size=3, seed=71)
        self.assertEqual([len(rows) for rows in capped], [3, 2, 3])
        self.assertEqual(capped, uncapped[:3])
        self.assertEqual((report["trained_examples"], report["steps"], report["epochs_completed"]), (8, 3, 1))
        self.assertEqual(report["stopped_reason"], "example_budget")
        self.assertFalse(report["interrupted"])
        self.assertEqual((full["planned_examples"], full["trained_examples"], full["epochs_completed"]), (15, 15, 3))
        self.assertEqual(full["stopped_reason"], "epochs_complete")

    def test_epochs_are_upper_bound_even_with_unused_example_budget(self):
        capped, report = self.recorded_fit(epochs=2, batch_size=3, seed=71, max_examples=12)
        uncapped, _ = self.recorded_fit(epochs=2, batch_size=3, seed=71)
        self.assertEqual(capped, uncapped)
        self.assertEqual((report["planned_examples"], report["trained_examples"], report["steps"]), (10, 10, 4))
        self.assertEqual(report["epochs_completed"], 2)
        self.assertFalse(report["example_budget_exhausted"])
        self.assertEqual(report["stopped_reason"], "epochs_complete")

    def test_deadline_reports_only_successful_optimizer_exposures(self):
        model = PolicyValueNet(NetworkConfig(blocks=0, channels=4))
        optimizer = torch.optim.AdamW(model.parameters())
        actual_step = optimizer.step
        now = [0.0]

        def expire_after_step(*args, **kwargs):
            result = actual_step(*args, **kwargs)
            now[0] = 2.0
            return result

        with patch.object(train.time, "perf_counter", side_effect=lambda: now[0]), \
                patch.object(optimizer, "step", side_effect=expire_after_step):
            _, report = fit(model, tiny_samples() * 3, epochs=2, batch_size=2,
                            optimizer=optimizer, max_examples=5, max_seconds=1)
        self.assertEqual((report["planned_examples"], report["trained_examples"], report["steps"]), (5, 2, 1))
        self.assertTrue(report["interrupted"])
        self.assertFalse(report["example_budget_exhausted"])
        self.assertEqual(report["stopped_reason"], "deadline")
        self.assertEqual(report["epochs_completed"], 0)

    def test_preparation_deadline_and_invalid_example_budgets(self):
        model = PolicyValueNet(NetworkConfig(blocks=0, channels=4))
        for cap in (-1, True, 1.5):
            with self.subTest(cap=cap), self.assertRaisesRegex(ValueError, "max_examples"):
                fit(model, tiny_samples(), max_examples=cap)
        with patch.object(train, "prepare", side_effect=train._PreparationDeadline):
            _, report = fit(model, tiny_samples(), max_examples=1)
        self.assertEqual((report["planned_examples"], report["trained_examples"], report["steps"]), (1, 0, 0))
        self.assertTrue(report["preparation_interrupted"])
        self.assertEqual(report["stopped_reason"], "deadline")


if __name__ == "__main__":
    unittest.main()
