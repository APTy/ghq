"""Per-game collection/search/evaluator metrics preserve actual play and labels."""
import json
import unittest
from unittest.mock import patch

try:
    import numpy
    import torch
except ImportError:
    raise unittest.SkipTest("Neural self-play tests require optional training dependencies")

from ghqnet.rules import e
from ghqnet.selfplay import play_game


class TimedEvaluator:
    def __init__(self, clock, duration=0.02):
        self.clock = clock
        self.duration = duration
        self.calls = 0

    def predict(self, board, legal_moves):
        self.calls += 1
        self.clock[0] += self.duration
        return [float(move.name == "Skip") for move in legal_moves], 0.0


def small_position(pieces=None):
    board = e.BaseBoard()
    board.clear_board()
    for square, symbol in (pieces or {"a1": "Q", "h8": "q"}).items():
        board.set_piece_at(e.parse_square(square), e.Piece.from_symbol(symbol))
    return board


class NeuralSelfplayTests(unittest.TestCase):
    def test_historical_roles_keep_only_learner_targets_and_preserve_state(self):
        current, history = TimedEvaluator([0.0], 0), TimedEvaluator([0.0], 0)
        state = small_position().serialize()
        samples, report = play_game(current, opponent="historical", opponent_evaluator=history,
                                    learner_color="RED", model_id="new", opponent_model_id="old",
                                    simulations=3, max_turns=2, start_state=state, seed=1)
        self.assertEqual(report["moves"], ["skip", "skip"])
        self.assertEqual(report["action_roles"], ["learner", "opponent"])
        self.assertEqual(report["action_sources"], ["mcts:new", "mcts:old"])
        self.assertEqual(report["sample_action_indices"], [0])
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].state, state)
        self.assertEqual(samples[0].outcome, 1)
        self.assertEqual(samples[0].source, "mcts:new")
        self.assertEqual(samples[0].model_id, "new")
        self.assertEqual(samples[0].priority, "normal")
        self.assertGreater(report["learner_evaluator_calls"], 0)
        self.assertGreater(report["opponent_evaluator_calls"], 0)
        self.assertEqual(report["learner_turn_start_skips"], 1)
        self.assertEqual(report["opponent_turn_start_skips"], 1)
        self.assertGreater(report["root_legal_actions"], report["root_visited_actions"])
        self.assertGreater(report["opposite_player_evaluator_calls"], 0)

    def test_cpu_opponent_uses_configured_search_without_recording_opponent_targets(self):
        evaluator = TimedEvaluator([0.0], 0)
        with patch("ghqnet.selfplay.cpu.analyze_json", return_value=json.dumps({"turnMoves": ["skip"], "nodes": 5})) as cpu:
            samples, report = play_game(evaluator, opponent="cpu", learner_color="BLUE",
                                        simulations=3, max_turns=2, start_state=small_position().serialize(),
                                        teacher_ms=3500, teacher_depth=4, teacher_beam_width=12)
        self.assertEqual(cpu.call_args.kwargs["time_ms"], 3500)
        self.assertEqual(cpu.call_args.kwargs["max_depth"], 4)
        self.assertEqual(cpu.call_args.kwargs["beam_width"], 12)
        self.assertEqual(report["action_roles"], ["opponent", "learner"])
        self.assertEqual(report["sample_action_indices"], [1])
        self.assertEqual(len(samples), 1)
        self.assertEqual(e.BaseBoard.deserialize(samples[0].state).turn, e.BLUE)
        self.assertEqual(samples[0].outcome, 1)
        self.assertEqual(report["teacher_search_calls"], 1)
        provenance = report["value_provenance"]
        self.assertEqual(provenance["collection_kind"], "cpu")
        self.assertEqual(provenance["players"]["RED"]["search"], {
            "kind": "cpu", "time_ms": 3500, "max_depth": 4, "beam_width": 12, "multi_pv": 1})
        self.assertEqual(provenance["players"]["BLUE"]["search"]["kind"], "neural")
        self.assertEqual(samples[0].value_provenance, provenance)

    def test_mixed_configuration_requires_color_and_historical_evaluator(self):
        evaluator = TimedEvaluator([0.0], 0)
        with self.assertRaisesRegex(ValueError, "learner_color"):
            play_game(evaluator, opponent="cpu")
        with self.assertRaisesRegex(ValueError, "opponent_evaluator"):
            play_game(evaluator, opponent="historical", learner_color="RED")
        with self.assertRaisesRegex(ValueError, "only one"):
            play_game(evaluator, start_state=small_position().serialize(), start_board=small_position())

    def test_tactical_priority_preserves_search_targets_and_model_provenance(self):
        board = small_position({"a1": "Q", "h8": "q", "g8": "I", "g6": "I"})
        samples, report = play_game(TimedEvaluator([0.0], 0), simulations=3, max_turns=1,
                                    start_board=board, model_id="checkpoint-sha-example")
        self.assertEqual(report["moves"], ["g6h7xh8"])
        self.assertEqual(samples[0].priority, "hard")
        self.assertEqual(samples[0].model_id, "checkpoint-sha-example")
        self.assertEqual(report["hard_examples"], 1)
        self.assertEqual(samples[0].policy[samples[0].actions.index("g6h7xh8")], 1)
        self.assertEqual(samples[0].outcome, 0)

    def test_hq_bombardment_is_priority_without_relabeling_action(self):
        board = small_position({"a1": "Q", "h8": "q", "a3": "r↓"})
        samples, report = play_game(TimedEvaluator([0.0], 0), simulations=1, max_turns=1,
                                    start_board=board)
        self.assertEqual(samples[0].priority, "hard")
        self.assertGreater(report["hard_examples"], 0)
        samples[0].validate()

    def test_neural_metrics_count_new_nodes_and_actual_predict_calls(self):
        clock = [0.0]
        evaluator = TimedEvaluator(clock)
        board = small_position()
        original = board.serialize()
        with patch("ghqnet.selfplay.time.perf_counter", side_effect=lambda: clock[0]):
            samples, report = play_game(evaluator, simulations=3, max_turns=1,
                                        seed=1, start_board=board)
        self.assertEqual(report["moves"], ["skip"])
        self.assertEqual(report["search_calls"], 1)
        self.assertEqual(report["search_simulations"], 3)
        # Root + opponent leaf + exact double-Skip terminal; only the two
        # nonterminal nodes invoke the network, and cached visits do not.
        self.assertEqual(report["search_nodes"], 3)
        self.assertEqual(report["evaluator_calls"], 2)
        self.assertEqual(evaluator.calls, 2)
        self.assertAlmostEqual(report["evaluator_seconds"], 0.04)
        self.assertAlmostEqual(report["search_seconds"], 0.04)
        self.assertAlmostEqual(report["seconds"], 0.04)
        self.assertEqual(report["teacher_search_calls"], 0)
        self.assertTrue(report["truncated"])
        self.assertEqual(report["reason"], "max-turns")
        self.assertTrue(all(sample.outcome is None for sample in samples))
        self.assertEqual(board.serialize(), original)
        for sample in samples:
            sample.validate()

    def test_metrics_are_per_game_even_when_evaluator_is_reused(self):
        clock = [0.0]
        evaluator = TimedEvaluator(clock)
        with patch("ghqnet.selfplay.time.perf_counter", side_effect=lambda: clock[0]):
            reports = [play_game(evaluator, simulations=3, max_turns=1, seed=1,
                                 start_board=small_position())[1] for _ in range(2)]
        self.assertEqual(evaluator.calls, 4)
        for report in reports:
            self.assertEqual(report["evaluator_calls"], 2)
            self.assertAlmostEqual(report["evaluator_seconds"], 0.04)

    def test_terminal_win_keeps_outcome_labels_and_exact_terminal_node_count(self):
        clock = [0.0]
        evaluator = TimedEvaluator(clock)
        board = small_position({"a1": "Q", "h8": "q", "h6": "R↑"})
        with patch("ghqnet.selfplay.time.perf_counter", side_effect=lambda: clock[0]):
            samples, report = play_game(evaluator, simulations=3, max_turns=2, start_board=board)
        self.assertFalse(report["truncated"])
        self.assertEqual(report["terminal"]["winner"], "RED")
        self.assertEqual(report["moves"], ["sbh8"])
        self.assertEqual(report["search_nodes"], 2)
        self.assertEqual(report["evaluator_calls"], 1)
        self.assertEqual([sample.outcome for sample in samples], [0])

    def test_teacher_time_and_nodes_count_each_plan_once(self):
        clock = [0.0]

        def teacher_analysis(*args, **kwargs):
            clock[0] += 0.15
            return json.dumps({"turnMoves": ["f2f3", "g2g3", "h2h3"], "nodes": 17})

        with patch("ghqnet.selfplay.time.perf_counter", side_effect=lambda: clock[0]), \
             patch("ghqnet.selfplay.cpu.analyze_json", side_effect=teacher_analysis) as teacher:
            samples, report = play_game(teacher=True, max_turns=1)
        self.assertEqual(teacher.call_count, 1)
        self.assertEqual(report["actions"], 3)
        self.assertEqual(report["teacher_search_calls"], 1)
        self.assertEqual(report["teacher_search_nodes"], 17)
        self.assertAlmostEqual(report["teacher_search_seconds"], 0.15)
        self.assertAlmostEqual(report["seconds"], 0.15)
        self.assertEqual(report["search_simulations"], 0)
        self.assertEqual(report["search_nodes"], 0)
        self.assertEqual(report["evaluator_calls"], 0)
        self.assertEqual(report["evaluator_seconds"], 0)
        self.assertTrue(report["truncated"])
        self.assertTrue(all(sample.outcome is None for sample in samples))
        self.assertEqual(report["value_provenance"]["collection_kind"], "teacher")
        self.assertEqual(report["value_provenance"]["players"]["RED"],
                         report["value_provenance"]["players"]["BLUE"])
        self.assertEqual(report["value_provenance"]["players"]["RED"]["search"]["kind"], "cpu")
        for sample in samples:
            sample.validate()


if __name__ == "__main__":
    unittest.main()
