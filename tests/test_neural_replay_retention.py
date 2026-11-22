"""Replay quota tests use lightweight records: no Torch or model inference."""
from dataclasses import dataclass
import itertools
import math
import unittest

from ghqnet.curriculum import retain_replay


@dataclass(frozen=True)
class Row:
    identity: int
    source: str
    priority: str


def rows(both=0, teacher=0, hard=0, ordinary=0):
    result = []
    for count, source, priority in ((both, "ghq-cpu-4", "hard"),
                                    (teacher, "ghq-cpu-4", "normal"),
                                    (hard, "mcts:example", "hard"),
                                    (ordinary, "mcts:example", "normal")):
        start = len(result)
        result.extend(Row(start + i, source, priority) for i in range(count))
    return result


def counts(records):
    return (sum(row.source.startswith("ghq-cpu") for row in records),
            sum(row.priority == "hard" for row in records))


class ReplayRetentionTests(unittest.TestCase):
    def test_hard_teacher_rows_satisfy_teacher_total_quota(self):
        examples = rows(both=40, hard=80, ordinary=300)
        for seed in range(5):
            retained = retain_replay(examples, 100, teacher_fraction=0.2, hard_fraction=0.1, seed=seed)
            teacher, hard = counts(retained)
            self.assertGreaterEqual(teacher, 20)
            self.assertGreaterEqual(hard, 10)
            self.assertEqual(len(retained), 100)
            self.assertEqual(len({row.identity for row in retained}), 100)

    def test_overlap_satisfies_both_rounded_quotas_in_one_slot(self):
        examples = rows(both=1, teacher=2, hard=2, ordinary=3)
        retained = retain_replay(examples, 1, teacher_fraction=0.5, hard_fraction=0.5, seed=31)
        self.assertEqual(counts(retained), (1, 1))
        self.assertEqual(len(retained), 1)

    def test_infeasible_tiny_rounding_uses_largest_remainder_and_teacher_tie(self):
        examples = rows(teacher=4, hard=4, ordinary=4)
        for teacher_fraction, hard_fraction, expected in ((0.2, 0.1, (1, 0)),
                                                          (0.1, 0.2, (0, 1)),
                                                          (0.5, 0.5, (1, 0))):
            with self.subTest(teacher=teacher_fraction, hard=hard_fraction):
                retained = retain_replay(examples, 1, teacher_fraction=teacher_fraction,
                                         hard_fraction=hard_fraction, seed=8)
                self.assertEqual(counts(retained), expected)
        retained = retain_replay(examples, 3, teacher_fraction=0.5, hard_fraction=0.5, seed=8)
        self.assertEqual(counts(retained), (2, 1))

    def test_scarce_buckets_are_kept_and_unused_capacity_filled(self):
        examples = rows(both=1, teacher=1, hard=2, ordinary=50)
        retained = retain_replay(examples, 20, teacher_fraction=0.5, hard_fraction=0.4, seed=19)
        self.assertEqual(counts(retained), (2, 3))
        self.assertEqual(len(retained), 20)

    def test_repeatable_ordered_sampling_without_mutating_input(self):
        examples = rows(both=10, teacher=20, hard=15, ordinary=100)
        original = list(examples)
        retained = retain_replay(examples, 50, seed=911)
        self.assertEqual(retained, retain_replay(examples, 50, seed=911))
        self.assertEqual(examples, original)
        self.assertEqual([row.identity for row in retained], sorted(row.identity for row in retained))
        self.assertEqual(retain_replay(examples, len(examples)), examples)

    def test_all_small_feasible_bucket_layouts_meet_both_targets(self):
        fractions = ((0, 0), (0.2, 0.1), (0.5, 0.5), (0.7, 0.3), (1, 0), (0, 1))
        for both, teacher, hard, ordinary in itertools.product(range(3), repeat=4):
            examples = rows(both, teacher, hard, ordinary)
            for limit in range(1, len(examples)):
                for teacher_fraction, hard_fraction in fractions:
                    teacher_target = min(both + teacher, math.ceil(limit * teacher_fraction))
                    hard_target = min(both + hard, math.ceil(limit * hard_fraction))
                    feasible = teacher_target + hard_target - min(both, teacher_target, hard_target) <= limit
                    retained = retain_replay(examples, limit, teacher_fraction=teacher_fraction,
                                             hard_fraction=hard_fraction, seed=73)
                    self.assertEqual(len(retained), limit)
                    self.assertEqual(len({row.identity for row in retained}), limit)
                    if feasible:
                        kept_teacher, kept_hard = counts(retained)
                        self.assertGreaterEqual(kept_teacher, teacher_target)
                        self.assertGreaterEqual(kept_hard, hard_target)


if __name__ == "__main__":
    unittest.main()
