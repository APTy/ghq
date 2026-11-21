import tempfile
from pathlib import Path
import unittest

try:
    import numpy as np
    import torch
except ImportError:
    raise unittest.SkipTest("Neural model tests require numpy and torch")

from ghqnet.encoding import encode_actions, encode_board
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
