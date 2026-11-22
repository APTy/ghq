import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
except ImportError:
    raise unittest.SkipTest("Neural model tests require numpy and torch")

from ghqnet.encoding import BOARD_CHANNELS, encode_actions, encode_board
from ghqnet.model import (NetworkConfig, PolicyValueNet, TorchEvaluator,
                          load_checkpoint, save_checkpoint)
from ghqnet.rules import e


class NeuralModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(21)
        self.model = PolicyValueNet(NetworkConfig(blocks=1, channels=8))
        self.board = e.BaseBoard()
        self.moves = list(self.board.generate_legal_moves())

    def batch(self):
        boards = torch.from_numpy(np.stack([encode_board(self.board)] * 2))
        actions = torch.from_numpy(np.stack([encode_actions(self.board, self.moves)] * 2))
        mask = torch.ones(actions.shape[:2], dtype=torch.bool)
        mask[1, 7:] = False
        actions[1, 7:] = -999  # Masked padding must never reach an embedding.
        return boards, actions, mask

    def test_shapes_legal_mask_and_action_order_equivariance(self):
        boards, actions, mask = self.batch()
        policy, wdl = self.model(boards, actions, mask)
        self.assertEqual(policy.shape, mask.shape)
        self.assertEqual(wdl.shape, (2, 3))
        self.assertTrue(torch.isneginf(policy[~mask]).all())
        self.assertTrue(torch.isfinite(policy[mask]).all())
        self.assertTrue(torch.isfinite(wdl).all())
        probabilities = policy.softmax(-1)
        torch.testing.assert_close(probabilities.sum(-1), torch.ones(2))
        self.assertEqual(probabilities[~mask].count_nonzero().item(), 0)
        order = torch.arange(len(self.moves) - 1, -1, -1)
        shuffled, repeated_wdl = self.model(boards, actions[:, order], mask[:, order])
        torch.testing.assert_close(shuffled, policy[:, order])
        torch.testing.assert_close(repeated_wdl, wdl)
        with self.assertRaisesRegex(ValueError, "at least one"):
            self.model(boards, actions, torch.zeros_like(mask))

    def test_policy_and_wdl_loss_backpropagate_into_shared_trunk(self):
        boards, actions, mask = self.batch()
        policy, wdl = self.model(boards, actions, mask)
        target_policy = mask.float() / mask.sum(-1, keepdim=True)
        log_policy = policy.log_softmax(-1).masked_fill(~mask, 0)
        loss = -(target_policy * log_policy).sum(-1).mean()
        loss = loss + torch.nn.functional.cross_entropy(wdl, torch.tensor([0, 2]))
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        for parameter in (self.model.trunk[0].weight, self.model.policy_head[0].weight,
                          self.model.value_head[0].weight, self.model.kind_embedding.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_evaluator_batch_matches_single_and_mirror(self):
        evaluator = TorchEvaluator(self.model, "cpu")
        # Identical canonical positions, with independent physical legal orders.
        mirror = self.board.mirror()
        mirror_moves = list(mirror.generate_legal_moves())
        before = self.board.serialize()
        first = evaluator.predict(self.board, self.moves)
        batch = evaluator.predict_batch([self.board, mirror], [self.moves, mirror_moves])
        np.testing.assert_allclose(first[0], batch[0][0], atol=1e-7)
        self.assertAlmostEqual(first[1], batch[0][1], places=6)
        self.assertAlmostEqual(batch[0][1], batch[1][1], places=6)
        mirror_priors = dict(zip(map(tuple, encode_actions(mirror, mirror_moves)), batch[1][0]))
        np.testing.assert_allclose(batch[0][0], [mirror_priors[tuple(a)] for a in encode_actions(self.board, self.moves)], atol=1e-7)
        self.assertAlmostEqual(sum(first[0]), 1, places=6)
        self.assertLessEqual(abs(first[1]), 1)
        self.assertEqual(self.board.serialize(), before)
        self.assertEqual(evaluator.predict_batch([], []), [])
        with self.assertRaisesRegex(ValueError, "no supplied legal"):
            evaluator.predict(self.board, [])

    def test_terminal_uses_exact_current_player_value(self):
        board = self.board.copy()
        board.set_piece_at(e.H1, None)
        evaluator = TorchEvaluator(self.model)
        self.assertEqual(evaluator.predict(board, []), ([], -1.0))
        board.turn = e.BLUE
        self.assertEqual(evaluator.predict(board, []), ([], 1.0))

    def test_encoded_batch_matches_legacy_with_mixed_counts_and_terminal_rows(self):
        evaluator = TorchEvaluator(self.model)
        advanced = self.board.copy()
        advanced.push(e.Move.from_uci("rrd1"))
        boards = [self.board, advanced, advanced.mirror()]
        moves = [list(board.generate_legal_moves()) for board in boards]
        self.assertGreater(len(set(map(len, moves))), 1)
        features = [encode_board(board) for board in boards]
        descriptors = [encode_actions(board, actions) for board, actions in zip(boards, moves)]
        snapshots = [array.copy() for array in features + descriptors]
        expected = evaluator.predict_batch(boards, moves)
        with patch("ghqnet.model.encode_board", side_effect=AssertionError("Re-encoded board")), \
                patch("ghqnet.model.encode_actions", side_effect=AssertionError("Re-encoded actions")), \
                patch("ghqnet.model.terminal_value", side_effect=AssertionError("Re-ran rules")):
            actual = evaluator.predict_encoded_batch(features, descriptors)
        self.assertEqual(actual, expected)
        for array, before in zip(features + descriptors, snapshots):
            np.testing.assert_array_equal(array, before)
        terminal = self.board.copy()
        terminal.set_piece_at(e.H1, None)
        with_terminal = evaluator.predict_batch([boards[0], terminal, *boards[1:]], [moves[0], [], *moves[1:]])
        self.assertEqual(with_terminal, [expected[0], ([], -1.0), *expected[1:]])
        # A singleton legal-action row gets exactly probability one, even
        # when a wider row determines the batch's padding width.
        singleton = evaluator.predict_encoded_batch(features[:2], [descriptors[0], descriptors[1][:1]])
        self.assertEqual(singleton[1][0], [1.0])
        self.assertEqual(evaluator.predict_encoded_batch([], []), [])

    def test_encoded_batch_input_validation_precedes_model_execution(self):
        evaluator = TorchEvaluator(self.model)
        board, actions = encode_board(self.board), encode_actions(self.board, self.moves)
        nonfinite = board.copy()
        nonfinite[0, 0, 0] = np.nan
        invalid_actions = []
        for column, maximum in enumerate((64, 64, 64, 8, 5, 7)):
            for value in (-1, maximum + 1):
                row = actions[:1].copy()
                row[0, column] = value
                invalid_actions.append(row)
        cases = [([board], []), ([board.astype(np.float64)], [actions]),
                 ([board[:, :, :7]], [actions]), ([nonfinite], [actions]),
                 ([board], [actions.astype(np.float32)]), ([board], [actions[:, :5]]),
                 ([board], [np.empty((0, 6), np.int64)]),
                 *[([board], [bad]) for bad in invalid_actions]]
        with patch.object(self.model, "forward", side_effect=AssertionError("Malformed data reached model")):
            for boards, descriptors in cases:
                with self.subTest(shapes=[getattr(x, "shape", None) for x in boards + descriptors]), \
                        self.assertRaises(ValueError):
                    evaluator.predict_encoded_batch(boards, descriptors)

    def test_inference_profile_accumulates_phases_without_changing_outputs(self):
        evaluator, profile = TorchEvaluator(self.model), {}
        expected = evaluator.predict(self.board, self.moves)
        with patch("torch.cuda.synchronize", side_effect=AssertionError("CPU inference synchronized CUDA")):
            legacy = evaluator.predict_batch([self.board], [self.moves], profile=profile)[0]
            encoded = evaluator.predict_encoded_batch([encode_board(self.board)],
                                                      [encode_actions(self.board, self.moves)], profile=profile)[0]
        self.assertEqual(expected, legacy)
        self.assertEqual(expected, encoded)
        self.assertEqual(profile["calls"], 2)
        self.assertEqual(profile["rows"], 2)
        self.assertEqual(profile["timing_scope"], "wall_seconds")
        self.assertEqual(profile["device"], "cpu")
        for phase in ("encode_seconds", "packing_seconds", "h2d_seconds", "forward_seconds",
                      "d2h_postprocess_seconds"):
            self.assertGreaterEqual(profile[phase], 0)

    def test_checkpoint_roundtrip_optimizer_metadata_and_version_guard(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001)
        boards, actions, mask = self.batch()
        _, wdl = self.model(boards, actions, mask)
        wdl.square().mean().backward()
        optimizer.step()
        expected = TorchEvaluator(self.model).predict(self.board, self.moves)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"
            save_checkpoint(path, self.model, optimizer, step=7, label="tiny test")
            loaded, payload = load_checkpoint(path)
            self.assertEqual(loaded.config, self.model.config)
            self.assertEqual(payload["metadata"], {"step": 7, "label": "tiny test"})
            self.assertTrue(payload["optimizer_state"]["state"])
            actual = TorchEvaluator(loaded).predict(self.board, self.moves)
            np.testing.assert_array_equal(expected[0], actual[0])
            self.assertEqual(expected[1], actual[1])
            self.assertEqual([p.name for p in path.parent.iterdir()], [path.name])
            payload["rules_sha256"] = "outdated"
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "rules hash"):
                load_checkpoint(path)
            load_checkpoint(path, strict_rules=False)
            payload["encoding_version"] = -1
            torch.save(payload, path)
            with self.assertRaisesRegex(ValueError, "encoding_version"):
                load_checkpoint(path, strict_rules=False)


if __name__ == "__main__":
    unittest.main()
