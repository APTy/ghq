from copy import deepcopy
from types import SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Neural arena tests require torch")

from ghqnet.arena import create_openings, match, summarize_matches, validate_openings
from ghqnet.rules import e


class ArenaTests(unittest.TestCase):
    def test_openings_are_reproducible_distinct_and_full_state_pinned(self):
        manifest = create_openings(pairs=3, seed=91, turns=2)
        self.assertEqual(create_openings(pairs=3, seed=91, turns=2), manifest)
        self.assertEqual(len(validate_openings(manifest)), 3)
        self.assertEqual(len({item["canonical_state_id"] for item in manifest["openings"]}), 3)
        duplicate = deepcopy(manifest)
        duplicate["openings"].append(deepcopy(duplicate["openings"][0]))
        with self.assertRaises(ValueError):
            validate_openings(duplicate)
        bad = deepcopy(manifest); bad["rules_sha256"] = "changed"
        with self.assertRaises(ValueError):
            validate_openings(bad)
        with self.assertRaises(ValueError):
            create_openings(pairs=2, turns=0)

    def test_historical_match_uses_distinct_trees_and_exact_terminal_draw(self):
        trees = []
        class FakeSearch:
            def __init__(self, evaluator, **kwargs):
                self.evaluator = evaluator
                self.calls = []
                trees.append(self)
            def plan_turn(self, board, time_ms):
                self.calls.append((board.turn, time_ms))
                return SimpleNamespace(moves=["skip"])
        candidate, historical = object(), object()
        board = e.BaseBoard(); original = board.serialize()
        with patch("ghqnet.arena.MCTS", FakeSearch), patch("ghqnet.arena.cpu.analyze_json") as cpu_call:
            result = match(candidate, opponent_evaluator=historical, neural_color=e.RED,
                           seed=1, start_board=board, time_ms=2400, max_turns=3)
        self.assertEqual(result["outcome"], "draw")
        self.assertFalse(result["unfinished"])
        self.assertEqual(result["opponent"], "historical")
        self.assertEqual(trees[0].calls, [(e.RED, 2400)])
        self.assertEqual(trees[1].calls, [(e.BLUE, 2400)])
        self.assertEqual([tree.evaluator for tree in trees], [candidate, historical])
        self.assertEqual(board.serialize(), original)
        cpu_call.assert_not_called()

    def test_length_cutoff_stays_unfinished(self):
        with patch("ghqnet.arena.MCTS") as search:
            search.return_value.plan_turn.return_value = SimpleNamespace(moves=["skip"])
            result = match(object(), neural_color=e.RED, seed=1, opening_turns=0, max_turns=1)
        self.assertTrue(result["unfinished"])
        self.assertIsNone(result["outcome"])
        self.assertIsNone(result["terminal"])

    def test_pair_cluster_bootstrap_and_unknown_score_bounds(self):
        def game(opening, color, outcome):
            return {"opening_id": opening, "neural_color": color, "outcome": outcome,
                    "unfinished": outcome is None}
        rows = [game("a", color, "win") for color in ("RED", "BLUE")]
        one = summarize_matches(rows)
        self.assertIsNone(one["completed_only_score_ci95"])
        rows += [game("b", color, "loss") for color in ("RED", "BLUE")]
        report = summarize_matches(rows, bootstrap_samples=1000)
        self.assertEqual(report["completed_game_score"], .5)
        self.assertEqual(report["completed_only_score_ci95"], [0, 1])
        self.assertEqual(report["completed_opening_pairs"], 2)
        self.assertEqual(report["independent_opening_clusters"], 2)
        rows[1] = game("a", "BLUE", None); rows[3] = game("b", "BLUE", None)
        report = summarize_matches(rows, bootstrap_samples=100)
        self.assertEqual(report["completion_rate"], .5)
        self.assertEqual(report["totals"]["unfinished"], 2)
        self.assertEqual(report["unfinished_score_bounds"], [.25, .75])
        # An incomplete color pair contributes no bootstrap cluster yet.
        self.assertIsNone(summarize_matches(rows[:1])["completed_only_score_ci95"])

    def test_uniform_wins_do_not_produce_a_false_certainty_interval(self):
        rows = [{"opening_id": opening, "neural_color": color, "outcome": "win", "unfinished": False}
                for opening in ("a", "b") for color in ("RED", "BLUE")]
        report = summarize_matches(rows, bootstrap_samples=100)
        self.assertIsNone(report["completed_only_score_ci95"])
        self.assertEqual(report["ci_warning"], "degenerate_bootstrap_interval_suppressed")
        low, high = report["fully_finished_pair_score_ci95"]
        self.assertLess(low, .1)
        self.assertEqual(high, 1)


if __name__ == "__main__":
    unittest.main()
