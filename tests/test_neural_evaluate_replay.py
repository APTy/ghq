from dataclasses import replace
import json
import math
from pathlib import Path
import tempfile
import unittest

try:
    import numpy as np
    import torch
except ImportError:
    raise unittest.SkipTest("Replay evaluation tests require numpy and torch")

from ghqnet.data import Sample, write_replay
from ghqnet.evaluate_replay import (aggregate, evaluate_checkpoint, evaluate_samples,
                                    file_sha256, row_metrics, select_indices)
from ghqnet.model import NetworkConfig, PolicyValueNet, save_checkpoint
from ghqnet.rules import e


def sample_from(board, outcome=None, source="ghq-cpu-3", family="test"):
    actions = sorted(move.uci() for move in board.generate_legal_moves())
    target = "skip" if "skip" in actions else actions[0]
    return Sample(board.serialize(), actions, [float(action == target) for action in actions],
                  outcome=outcome, game_id=family, family_id=family, source=source)


class SkipModel(torch.nn.Module):
    def forward(self, boards, actions, mask):
        logits = (actions[:, :, 4] == 5).float() * 4
        return logits.masked_fill(~mask, -torch.inf), boards.new_tensor([.7, .2, .1]).log().expand(len(boards), -1)


class ReplayEvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_soft_target_metrics_skip_rates_and_labeled_only_wdl(self):
        sample = sample_from(e.BaseBoard(), outcome=2)
        skip = sample.actions.index("skip")
        other = next(index for index in range(len(sample.actions)) if index != skip)
        sample.policy[skip], sample.policy[other] = .7, .3
        probabilities = np.full(len(sample.actions), .4 / (len(sample.actions) - 1))
        probabilities[skip] = .6
        row = row_metrics(sample, np.log(probabilities), np.log([.2, .3, .5]))
        self.assertEqual(row["top1"], 1)
        self.assertEqual(row["top5"], 1)
        self.assertAlmostEqual(row["policy_nll"], -.7 * math.log(.6) - .3 * math.log(probabilities[other]))
        self.assertAlmostEqual(row["target_skip_mass"], .7)
        self.assertAlmostEqual(row["wdl_logloss"], -math.log(.5))
        self.assertAlmostEqual(row["wdl_brier"], .2**2 + .3**2 + .5**2)
        self.assertEqual(row["wdl_accuracy"], 1)
        unlabeled = row_metrics(replace(sample, outcome=None), np.log(probabilities), np.log([.9, .05, .05]))
        summary = aggregate([row, unlabeled])
        self.assertEqual(summary["wdl"]["class_counts"], {"win": 0, "draw": 0, "loss": 1})
        self.assertEqual(summary["wdl"]["unlabeled_samples"], 1)
        self.assertAlmostEqual(summary["wdl"]["logloss"], -math.log(.5))
        self.assertIsNone(aggregate([unlabeled])["wdl"]["accuracy"])
        self.assertEqual(summary["fresh_turn_skip"]["predicted_argmax_rate"], 1)
        # Place the teacher's chosen action strictly below the model's top five.
        bad_probs = np.arange(1, len(sample.actions) + 1, dtype=float)
        bad_probs[skip] = .001
        bad_probs /= bad_probs.sum()
        missed = row_metrics(sample, np.log(bad_probs), np.log([.2, .3, .5]))
        self.assertEqual(missed["top1"], 0)
        self.assertEqual(missed["top5"], 0)

    def test_lexical_ties_and_action_order_do_not_change_scores(self):
        sample = sample_from(e.BaseBoard())
        sample.policy = [1 / len(sample.actions)] * len(sample.actions)
        logs = np.log(sample.policy)
        before = row_metrics(sample, logs, np.log([.4, .3, .3]))
        reverse = replace(sample, actions=sample.actions[::-1], policy=sample.policy[::-1])
        after = row_metrics(reverse, logs[::-1], np.log([.4, .3, .3]))
        for field in ("top1", "top5", "target_skip", "predicted_skip", "policy_nll"):
            self.assertAlmostEqual(before[field], after[field])
        with self.assertRaisesRegex(ValueError, "Nonfinite"):
            row_metrics(sample, np.full(len(logs), np.nan), np.log([.4, .3, .3]))

    def test_forced_and_automatic_rows_do_not_inflate_nonforced_and_phases(self):
        board = e.BaseBoard()
        fresh = sample_from(board, 0)
        board.push(e.Move.reinforce(e.INFANTRY, e.A1))
        second = sample_from(board, 1, source="mcts:abc")
        board.push(e.Move.reinforce(e.INFANTRY, e.B1))
        third = sample_from(board, 2, source="mcts:abc")
        forced = e.BaseBoard()
        forced.clear_board()
        for square, piece in ((e.A1, e.Piece(e.HQ, e.RED)), (e.H8, e.Piece(e.HQ, e.BLUE)),
                              (e.D4, e.Piece(e.ARTILLERY, e.RED, e.ORIENT_N)),
                              (e.D6, e.Piece(e.INFANTRY, e.BLUE))):
            forced.set_piece_at(square, piece)
        automatic = sample_from(forced)
        self.assertEqual(len(automatic.actions), 1)
        result = evaluate_samples(SkipModel(), [fresh, second, third, automatic], batch_size=3)
        self.assertEqual(result["overall"]["all"]["samples"], 4)
        self.assertEqual(result["overall"]["non_forced"]["samples"], 3)
        self.assertEqual(result["overall"]["single_legal_action"]["samples"], 1)
        self.assertEqual(result["overall"]["automatic_capture"]["samples"], 1)
        self.assertEqual(set(result["by_phase"]), {"fresh_turn_start", "second_action", "third_action", "automatic_bombard"})
        self.assertEqual(result["overall"]["all"]["fresh_turn_skip"]["samples"], 1)
        self.assertEqual(result["by_source"]["mcts:abc"]["all"]["samples"], 2)
        self.assertEqual(result["overall"]["all"]["wdl"]["class_counts"], {"win": 1, "draw": 1, "loss": 1})
        self.assertEqual(result["by_phase"]["automatic_bombard"]["non_forced"]["samples"], 0)

    def test_sampling_is_repeatable_bounded_and_balances_game_families(self):
        base = sample_from(e.BaseBoard())
        samples = [replace(base, family_id=family) for family in ("a",) * 9 + ("b",) * 3 + ("c",) * 2]
        selected = select_indices(samples, max_samples=3, seed=73)
        self.assertEqual(selected, select_indices(samples, max_samples=3, seed=73))
        self.assertEqual({samples[index].family_id for index in selected}, {"a", "b", "c"})
        self.assertEqual(select_indices(samples), list(range(len(samples))))
        with self.assertRaises(ValueError):
            select_indices(samples, 0)

    def test_tiny_checkpoint_replay_hashes_mirror_counts_and_no_mutation(self):
        board = e.BaseBoard()
        samples = [sample_from(board, 0), sample_from(board.mirror(), 2)]
        model = PolicyValueNet(NetworkConfig(blocks=0, channels=4))
        with tempfile.TemporaryDirectory() as folder:
            replay, checkpoint = Path(folder) / "validation.jsonl", Path(folder) / "checkpoint.pt"
            write_replay(replay, samples, kind="validation")
            save_checkpoint(checkpoint, model, iteration=1, validation_buffer_sha256=file_sha256(replay))
            before = {path: file_sha256(path) for path in (replay, checkpoint)}
            report = evaluate_checkpoint(checkpoint, replay, batch_size=2)
            self.assertEqual(report["sampling"]["available_samples"], 2)
            self.assertEqual(report["sampling"]["available_canonical_unique_states"], 1)
            self.assertTrue(report["holdout"]["matches_checkpoint_validation_buffer"])
            self.assertFalse(report["holdout"]["historical_training_overlap_verified_absent"])
            self.assertEqual(report["checkpoint_sha256"], before[checkpoint])
            self.assertEqual(report["replay_sha256"], before[replay])
            self.assertEqual(report["rules_sha256"], report["checkpoint_rules_sha256"])
            self.assertEqual(report["rules_sha256"], report["replay_rules_sha256"])
            self.assertEqual({path: file_sha256(path) for path in (replay, checkpoint)}, before)
            json.dumps(report, allow_nan=False)

    def test_known_training_data_and_malformed_outcomes_are_rejected(self):
        model = PolicyValueNet(NetworkConfig(blocks=0, channels=4))
        with tempfile.TemporaryDirectory() as folder:
            replay, checkpoint = Path(folder) / "validation.jsonl", Path(folder) / "checkpoint.pt"
            sample = sample_from(e.BaseBoard())
            write_replay(replay, [sample], kind="training")
            save_checkpoint(checkpoint, model)
            with self.assertRaisesRegex(ValueError, "marked as training"):
                evaluate_checkpoint(checkpoint, replay)
            write_replay(replay, [sample], kind="validation")
            save_checkpoint(checkpoint, model, replay_buffer_sha256=file_sha256(replay))
            with self.assertRaisesRegex(ValueError, "matches the checkpoint training"):
                evaluate_checkpoint(checkpoint, replay)
            save_checkpoint(checkpoint, model)
            write_replay(replay, [replace(sample, outcome=True)], kind="validation")
            with self.assertRaisesRegex(ValueError, "integer W/D/L"):
                evaluate_checkpoint(checkpoint, replay)
            write_replay(replay, [replace(sample, policy=[0] * len(sample.actions))], kind="validation")
            with self.assertRaisesRegex(ValueError, "sum to one"):
                evaluate_checkpoint(checkpoint, replay)
            replay.write_text("[]\n")
            with self.assertRaisesRegex(ValueError, "header must be an object"):
                evaluate_checkpoint(checkpoint, replay)


if __name__ == "__main__":
    unittest.main()
