"""Value eligibility must not change replay data, policy targets or row budgets."""
from copy import deepcopy
from dataclasses import asdict
import random
import unittest
from unittest.mock import patch

try:
    import numpy as np
    import torch
except ImportError:
    raise unittest.SkipTest("Neural training tests require numpy and torch")

from ghqnet import train
from ghqnet.data import Sample
from ghqnet.model import NetworkConfig, PolicyValueNet
from ghqnet.rules import e
from ghqnet.train import batch, fit, losses, prepare


def samples():
    board = e.BaseBoard()
    result = []
    for index, outcome in enumerate((0, 2, None, 1, 2)):
        actions = sorted(move.uci() for move in board.generate_legal_moves())
        result.append(Sample(board.serialize(), actions,
                             [float(action == "skip") for action in actions],
                             outcome=outcome, game_id=str(index),
                             source="teacher" if index % 2 else "mcts"))
    return result


class NeuralValueTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(313)

    def model(self):
        return PolicyValueNet(NetworkConfig(blocks=0, channels=4))

    def test_mask_preserves_raw_records_and_policy_and_normalizes_over_eligible_labels(self):
        rows = samples()
        original = deepcopy([asdict(row) for row in rows])
        mask = np.asarray([True, False, True, True, False], dtype=np.bool_)
        full = batch(prepare(rows), "cpu")
        masked = batch(prepare(rows, value_mask=mask), "cpu")
        self.assertEqual([asdict(row) for row in rows], original)
        self.assertEqual(masked[-1].tolist(), [0, -1, -1, 1, -1])
        for unmasked_tensor, masked_tensor in zip(full[:-1], masked[:-1]):
            torch.testing.assert_close(unmasked_tensor, masked_tensor, rtol=0, atol=0)
        model = self.model()
        full_policy, _ = losses(model, full)
        masked_policy, masked_value = losses(model, masked)
        torch.testing.assert_close(full_policy, masked_policy, rtol=0, atol=0)
        _, logits = model(*masked[:3])
        expected = torch.nn.functional.cross_entropy(logits[[0, 3]], torch.tensor([0, 1]))
        torch.testing.assert_close(masked_value, expected, rtol=0, atol=0)

    def test_no_eligible_labels_has_finite_policy_training_and_zero_value_head_gradient(self):
        model = self.model()
        policy, value = losses(model, batch(prepare(samples(), value_mask=[False] * 5), "cpu"))
        self.assertTrue(torch.isfinite(policy))
        self.assertEqual(value.item(), 0)
        (policy + value).backward()
        self.assertTrue(all(parameter.grad is None or torch.isfinite(parameter.grad).all()
                            for parameter in model.parameters()))
        self.assertTrue(all(parameter.grad is None or torch.count_nonzero(parameter.grad) == 0
                            for parameter in model.value_head.parameters()))
        # Adam history/weight decay can still update a value head with zero
        # gradients, so this deliberately makes no unchanged-weights claim.
        _, report = fit(model, samples(), epochs=1, batch_size=2, value_mask=[False] * 5)
        self.assertEqual(report["trained_examples"], 5)
        self.assertEqual(report["trained_value_examples"], 0)
        self.assertEqual(report["trained_value_examples_by_source"], {"teacher": 0, "mcts": 0})

    def test_all_true_is_exactly_default_training(self):
        plain = self.model()
        explicit = deepcopy(plain)
        options = dict(epochs=2, batch_size=2, seed=19, max_examples=8)
        _, default = fit(plain, samples(), **options)
        _, masked = fit(explicit, samples(), value_mask=[True] * 5, **options)
        for name, parameter in plain.state_dict().items():
            torch.testing.assert_close(parameter, explicit.state_dict()[name], rtol=0, atol=0)
        for name in ("before", "after", "trained_examples", "trained_value_examples",
                     "trained_value_examples_by_source", "steps", "epochs_completed"):
            self.assertEqual(default[name], masked[name])
        self.assertEqual(default["value_labels"], 4)
        self.assertEqual(default["raw_value_labels"], 4)
        self.assertEqual(default["eligible_value_labels"], 4)

    def test_partial_prefix_mask_follows_shuffled_indices_and_short_final_batch(self):
        rows = samples()
        mask = [True, False, True, True, False]
        indices = list(range(len(rows)))
        random.Random(37).shuffle(indices)
        selected = indices[:3]
        model = self.model()
        reference = deepcopy(model)
        _, report = fit(model, rows, epochs=3, batch_size=2, seed=37,
                        max_examples=3, value_mask=mask)
        optimizer = torch.optim.AdamW(reference.parameters(), lr=1e-3, weight_decay=1e-4)
        records = prepare([rows[i] for i in selected], value_mask=[mask[i] for i in selected])
        reference.train()
        for start in (0, 2):
            optimizer.zero_grad(set_to_none=True)
            policy, value = losses(reference, batch(records[start:start + 2], "cpu"))
            (policy + value).backward()
            torch.nn.utils.clip_grad_norm_(reference.parameters(), 5.0, error_if_nonfinite=True)
            optimizer.step()
        self.assertEqual((report["trained_examples"], report["steps"]), (3, 2))
        self.assertEqual(report["raw_value_labels"], 4)
        self.assertEqual(report["eligible_value_labels"], 2)
        expected = [i for i in selected if mask[i] and rows[i].outcome is not None]
        self.assertEqual(report["trained_value_examples"], len(expected))
        for source in ("teacher", "mcts"):
            self.assertEqual(report["trained_value_examples_by_source"][source],
                             sum(rows[i].source == source for i in expected))
        for name, parameter in model.state_dict().items():
            torch.testing.assert_close(parameter, reference.state_dict()[name], rtol=0, atol=0)

    def test_mask_stays_aligned_across_epochs_and_counts_repeated_value_exposures(self):
        rows = samples()
        mask = [True, False, True, True, False]
        rng = random.Random(71)
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        selected = indices[:]
        rng.shuffle(indices)
        selected += indices[:3]
        _, report = fit(self.model(), rows, epochs=3, batch_size=3, seed=71,
                        max_examples=8, value_mask=mask)
        expected = [i for i in selected if mask[i] and rows[i].outcome is not None]
        self.assertEqual((report["trained_examples"], report["steps"], report["epochs_completed"]), (8, 3, 1))
        self.assertEqual(report["trained_value_examples"], len(expected))
        self.assertEqual(sum(report["trained_value_examples_by_source"].values()), len(expected))

    def test_deadline_and_zero_budget_count_only_successful_value_exposures(self):
        rows = samples()
        mask = [True, False, True, True, False]
        indices = list(range(len(rows)))
        random.Random(19).shuffle(indices)
        model = self.model()
        optimizer = torch.optim.AdamW(model.parameters())
        actual_step, now = optimizer.step, [0.0]

        def expire(*args, **kwargs):
            actual_step(*args, **kwargs)
            now[0] = 2.0

        with patch.object(train.time, "perf_counter", side_effect=lambda: now[0]), \
                patch.object(optimizer, "step", side_effect=expire):
            _, report = fit(model, rows, epochs=2, batch_size=2, seed=19, optimizer=optimizer,
                            max_examples=3, max_seconds=1, value_mask=mask)
        self.assertEqual((report["trained_examples"], report["steps"]), (2, 1))
        self.assertEqual(report["trained_value_examples"],
                         sum(mask[i] and rows[i].outcome is not None for i in indices[:2]))
        self.assertTrue(report["interrupted"])
        with patch.object(train, "prepare", side_effect=AssertionError("must not prepare")):
            _, zero = fit(model, rows, optimizer=optimizer, max_examples=0, value_mask=mask)
        self.assertEqual(zero["trained_value_examples"], 0)
        self.assertEqual(zero["eligible_value_labels"], 2)

    def test_validation_stays_unmasked_unless_separately_requested(self):
        rows = samples()
        _, default = fit(self.model(), rows, epochs=1, value_mask=[False] * 5,
                         validation_samples=rows)
        self.assertEqual(default["before"]["value"], 0)
        self.assertGreater(default["validation_before"]["value"], 0)
        self.assertGreater(default["validation_after"]["value"], 0)
        _, masked = fit(self.model(), rows, epochs=1, value_mask=[False] * 5,
                        validation_samples=rows, validation_value_mask=[False] * 5)
        self.assertEqual(masked["validation_before"]["value"], 0)
        self.assertEqual(masked["validation_after"]["value"], 0)

    def test_invalid_masks_fail_before_training_or_partial_preparation(self):
        rows = samples()
        for mask in ([True], [1] * 5, "false", [None] * 5, True, [[True]] * 5):
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "value_mask"):
                prepare(rows, value_mask=mask)
            with self.subTest(mask=mask), self.assertRaisesRegex(ValueError, "value_mask"):
                fit(self.model(), rows, max_examples=0, value_mask=mask)
        with self.assertRaisesRegex(ValueError, "validation_value_mask"):
            fit(self.model(), rows, validation_samples=rows, validation_value_mask=[True])


if __name__ == "__main__":
    unittest.main()
