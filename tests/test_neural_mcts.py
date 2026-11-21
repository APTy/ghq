"""Neural tree-search contracts, independent of Torch or trained weights."""
from pathlib import Path
import importlib.util
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ghqnet.mcts import MCTS
from ghqnet.rules import e, cpu, terminal_value


def position(pieces, turn=e.RED):
    board = e.BaseBoard()
    board.clear_board()
    for square, symbol in pieces.items():
        board.set_piece_at(e.parse_square(square), e.Piece.from_symbol(symbol))
    board.turn = turn
    return board


class Evaluator:
    def __init__(self, value=0.625, preferred=None):
        self.value = value
        self.preferred = preferred
        self.calls = []

    def predict(self, board, legal):
        self.calls.append(cpu.state_key(board))
        policy = [float(move.uci() == self.preferred) for move in legal]
        return policy if any(policy) else [1.0] * len(legal), self.value


class NeuralMCTSTests(unittest.TestCase):
    def replay(self, board, moves):
        after = board.copy()
        for move in moves:
            self.assertIsNone(terminal_value(after), "plan continued beyond game end")
            action = e.Move.from_uci(move)
            self.assertTrue(after.is_legal(action), (move, after.board_fen()))
            after.push(action)
        return after

    def test_same_player_action_does_not_negate_leaf_value(self):
        board = position({"a1": "Q", "h8": "q", "c3": "I"})
        result = MCTS(Evaluator(preferred="c3c4"), simulations=1).search(board)
        self.assertEqual(result.value, 0.625)
        self.assertEqual(result.current_player, e.RED)
        self.assertEqual(result.visits[result.actions.index("c3c4")], 1)

    def test_turn_change_negates_leaf_value_for_both_colors(self):
        board = position({"a1": "Q", "h8": "q", "c3": "I"})
        board.turn_moves = 2
        for probe, preferred in ((board, "c3c4"), (board.mirror(), "f6f5")):
            result = MCTS(Evaluator(preferred=preferred), simulations=1).search(probe)
            self.assertEqual(result.value, -0.625)
            self.assertEqual(result.current_player, probe.turn)

    def test_multilevel_backup_changes_sign_only_at_final_action(self):
        board = position({"a1": "Q", "h8": "q", "c3": "I"})
        board.turn_moves = 1
        result = MCTS(Evaluator(preferred="c3c4"), simulations=2).search(board)
        # First simulation sees a same-player leaf (+.625); the second
        # continues through the third action to an enemy leaf (-.625).
        self.assertEqual(result.visits[result.actions.index("c3c4")], 2)
        self.assertEqual(result.value, 0)

    def test_automatic_capture_keeps_player_and_action_budget(self):
        board = position({"a1": "Q", "h8": "q", "d4": "R↑", "d6": "i"})
        for probe in (board, board.mirror()):
            result = MCTS(Evaluator(), simulations=1).search(probe)
            self.assertEqual(len(result.actions), 1)
            self.assertEqual(result.value, 0.625)
            after = self.replay(probe, result.actions)
            self.assertEqual(after.turn, probe.turn)
            self.assertEqual(after.turn_moves, 0)

    def test_terminal_win_loss_and_draw_use_exact_values_without_model(self):
        for turn in (e.RED, e.BLUE):
            for winner in (e.RED, e.BLUE):
                pieces = {"a1": "Q"} if winner == e.RED else {"h8": "q"}
                model = Evaluator(value=-0.75)
                result = MCTS(model).search(position(pieces, turn))
                self.assertTrue(result.terminal)
                self.assertEqual(result.value, 1 if turn == winner else -1)
                self.assertEqual(result.actions, [])
                self.assertEqual(model.calls, [])
            board = position({"a1": "Q", "h8": "q"}, turn)
            board.did_offer_draw = board.did_accept_draw = True
            self.assertEqual(MCTS(Evaluator()).search(board).value, 0)

    def test_immediate_hq_win_survives_a_zero_policy_prior(self):
        board = position({"a1": "Q", "h8": "q", "h6": "R↑"})
        for probe in (board, board.mirror()):
            result = MCTS(Evaluator(value=-1, preferred="skip"), simulations=1).search(probe)
            self.assertEqual(result.value, 1)
            chosen = result.actions[result.policy.index(max(result.policy))]
            self.assertIsNotNone(terminal_value(self.replay(probe, [chosen])))

    def test_distinct_capture_targets_are_distinct_policy_edges(self):
        board = position({"a1": "Q", "h8": "q", "c3": "I", "d4": "r↑", "e3": "r↑"})
        result = MCTS(Evaluator(), simulations=0).search(board)
        self.assertIn("c3d3xd4", result.actions)
        self.assertIn("c3d3xe3", result.actions)
        self.assertEqual(len(result.actions), len(set(result.actions)))
        self.assertAlmostEqual(sum(result.policy), 1)

    def test_repeated_search_and_child_search_reuse_predictions(self):
        board = position({"a1": "Q", "h8": "q", "c3": "I"})
        model = Evaluator(preferred="c3c4")
        tree = MCTS(model, simulations=1)
        first = tree.search(board)
        self.assertEqual(sum(first.visits), 1)
        calls = len(model.calls)
        child = self.replay(board, ["c3c4"])
        tree.simulations = 0
        result = tree.search(child)
        self.assertEqual(result.nodes, 0)
        self.assertEqual(len(model.calls), calls)
        tree.simulations = 1
        second = tree.search(board)
        self.assertEqual(sum(second.visits), 2)
        tree.clear()
        tree.simulations = 0
        self.assertEqual(tree.search(board).nodes, 1)

    def test_cache_distinguishes_reserves_actions_used_pieces_and_draw_offer(self):
        board = position({"a1": "Q", "h8": "q", "c3": "I"})
        model = Evaluator()
        tree = MCTS(model, simulations=0)
        self.assertEqual(tree.search(board).nodes, 1)
        self.assertEqual(tree.search(board).nodes, 0)
        changes = (
            lambda b: b.add_to_reserve(e.INFANTRY, e.RED),
            lambda b: setattr(b, "turn_moves", 1),
            lambda b: setattr(b, "turn_pieces", e.BB_SQUARES[e.C3]),
            lambda b: setattr(b, "did_offer_draw", True),
        )
        for change in changes:
            probe = board.copy()
            change(probe)
            self.assertEqual(tree.search(probe).nodes, 1)

    def test_pending_free_capture_choices_and_allowance_have_separate_cache_entries(self):
        board = position({"a1": "Q", "h8": "q", "c3": "I", "e3": "I",
                          "d2": "I", "d3": "i", "c2": "r←"})
        before = board.serialize()
        tree = MCTS(Evaluator(), simulations=0)
        result = tree.search(board)
        self.assertEqual(set(result.actions), {"sfc2", "sfd3"})
        self.assertEqual(board.serialize(), before)
        # The second call must find the same normalized, lazily initialized
        # pending-capture state without changing the caller's board.
        self.assertEqual(tree.search(board).nodes, 0)
        exhausted = board.copy()
        list(exhausted.generate_legal_moves())
        exhausted.free_capture_num_allowed = 0
        other = tree.search(exhausted)
        self.assertEqual(other.nodes, 1)
        self.assertNotIn("sfc2", other.actions)
        self.assertIn("skip", other.actions)

    def test_node_cache_is_bounded(self):
        board = position({"a1": "Q", "h8": "q", "c3": "I"})
        result = MCTS(Evaluator(), simulations=50, max_nodes=3).search(board)
        self.assertEqual(result.simulations, 50)
        self.assertLessEqual(result.cache_size, 3)

    def test_search_and_turn_plan_preserve_input(self):
        board = e.BaseBoard()
        before, stack = board.serialize(), list(board.move_stack)
        tree = MCTS(Evaluator(), simulations=2)
        tree.search(board)
        result = tree.plan_turn(board, time_ms=1000)
        self.assertEqual(board.serialize(), before)
        self.assertEqual(board.move_stack, stack)
        self.assertTrue(result.complete)
        self.assertNotEqual(self.replay(board, result.moves).turn, board.turn)

    def test_complete_turn_includes_mandatory_captures_before_three_actions(self):
        board = position({"a1": "Q", "h8": "q", "d4": "R↑", "e4": "R↑",
                          "d6": "i", "e6": "i", "b2": "I", "c2": "I"})
        result = MCTS(Evaluator(), simulations=2).plan_turn(board, time_ms=1000)
        self.assertTrue(result.complete)
        after = self.replay(board, result.moves)
        self.assertNotEqual(after.turn, board.turn)
        self.assertEqual(set(result.moves[:2]), {"sbd6", "sbe6"})
        self.assertLessEqual(sum(e.Move.from_uci(move).name != "AutoCapture" for move in result.moves), 3)

    def test_zero_budget_completes_turn_without_model_calls(self):
        board = position({"a1": "Q", "h8": "q", "d4": "R↑", "d6": "i"})
        model = Evaluator()
        result = MCTS(model).plan_turn(board, time_ms=0)
        self.assertEqual(result.moves, ["sbd6", "skip"])
        self.assertTrue(result.complete)
        self.assertTrue(result.timed_out)
        self.assertEqual(model.calls, [])
        self.replay(board, result.moves)

    def test_turn_deadline_is_shared_not_reset_per_action(self):
        board = position({"a1": "Q", "h8": "q", "c3": "I"})
        clock = [0.0]
        model = Evaluator()
        predict = model.predict

        def expensive_predict(state, legal):
            clock[0] += 0.03
            return predict(state, legal)

        model.predict = expensive_predict
        with patch("ghqnet.mcts.time.perf_counter", side_effect=lambda: clock[0]):
            result = MCTS(model).plan_turn(board, time_ms=60)
        self.assertEqual(len(model.calls), 2)
        self.assertAlmostEqual(result.elapsed_ms, 60)
        self.assertTrue(result.complete)
        self.replay(board, result.moves)

    def test_invalid_evaluator_output_fails_clearly(self):
        board = position({"a1": "Q", "h8": "q"})
        model = Evaluator()
        model.predict = lambda board, legal: ([float("nan")] * len(legal), 0)
        with self.assertRaises(ValueError):
            MCTS(model).search(board)

    @unittest.skipUnless(importlib.util.find_spec("torch") is not None, "Requires optional Torch training dependencies")
    def test_real_torch_evaluator_search_and_shared_budget_turn_smoke(self):
        import torch
        from ghqnet.model import NetworkConfig, PolicyValueNet, TorchEvaluator
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        try:
            torch.manual_seed(31)
            evaluator = TorchEvaluator(PolicyValueNet(NetworkConfig(blocks=1, channels=8)))
            tree = MCTS(evaluator, simulations=4)
            board = position({"a1": "Q", "h8": "q", "c3": "I"})
            before = board.serialize()
            for source in (board, board.mirror()):
                for actions_used in (0, 2):
                    probe = source.copy()
                    probe.turn_moves = actions_used
                    checked = MCTS(evaluator, simulations=1).search(probe)
                    action = checked.actions[checked.visits.index(1)]
                    leaf = self.replay(probe, [action])
                    legal = list(leaf.generate_legal_moves())
                    _, leaf_value = evaluator.predict(leaf, legal)
                    expected = leaf_value if probe.turn == leaf.turn else -leaf_value
                    self.assertAlmostEqual(checked.value, expected, places=6)
            result = tree.search(board)
            self.assertEqual(result.simulations, 4)
            self.assertAlmostEqual(sum(result.policy), 1)
            self.assertGreaterEqual(result.value, -1)
            self.assertLessEqual(result.value, 1)
            first_action = result.actions[result.policy.index(max(result.policy))]
            child = self.replay(board, [first_action])
            tree.simulations = 0
            self.assertEqual(tree.search(child).nodes, 0)
            plan = tree.plan_turn(board, time_ms=200, simulations_per_action=4)
            self.assertTrue(plan.complete)
            self.replay(board, plan.moves)
            self.assertEqual(board.serialize(), before)
        finally:
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
