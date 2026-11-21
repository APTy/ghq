from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

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


if __name__ == "__main__":
    unittest.main()
