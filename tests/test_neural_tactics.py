from dataclasses import replace
import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Neural tactical tests require torch")

from ghqnet.rules import e, cpu, copy_board
from ghqnet.tactics import judge_plan, tactical_cases


class NeuralTacticsTests(unittest.TestCase):
    def test_fixture_pairs_are_nonterminal_legal_and_exact_mirrors(self):
        cases = tactical_cases()
        self.assertEqual(len(cases), 10)
        self.assertEqual(len(set(case.name for case in cases)), 10)
        for index in range(0, len(cases), 2):
            original, mirror = cases[index:index + 2]
            self.assertEqual(cpu.state_key(original.board.mirror()), cpu.state_key(mirror.board))
            self.assertNotEqual(original.board.turn, mirror.board.turn)
            for case in (original, mirror):
                self.assertIsNone(cpu._terminal(case.board))
                self.assertTrue(list(copy_board(case.board).generate_legal_moves()))

    def test_reported_bad_lines_are_legal_complete_and_fail_in_both_colors(self):
        for case in tactical_cases():
            with self.subTest(case=case.name):
                before = case.board.serialize()
                result = judge_plan(case, case.known_bad_moves)
                self.assertTrue(result["legal"], result)
                self.assertTrue(result["complete"], result)
                self.assertFalse(result["passed"], result)
                self.assertEqual(case.board.serialize(), before)

    def test_illegal_incomplete_and_extra_actions_fail_validation(self):
        case = tactical_cases()[0]
        self.assertFalse(judge_plan(case, ["not-a-move"])["legal"])
        self.assertFalse(judge_plan(case, [case.known_bad_moves[0]])["complete"])
        self.assertFalse(judge_plan(case, ["skip", "skip"])["legal"])
        self.assertFalse(judge_plan(case, [])["passed"])

    def test_exact_hq_capture_passes_without_extra_terminal_action(self):
        for case in tactical_cases():
            if case.objective != "hq_capture":
                continue
            enemy_hq = case.board.hq & case.board.occupied_co[not case.board.turn]
            winning = next(move for move in copy_board(case.board).generate_legal_moves()
                           if move.capture_preference is not None and
                           e.BB_SQUARES[move.capture_preference] & enemy_hq)
            self.assertTrue(judge_plan(case, [winning.uci()])["passed"])
            self.assertFalse(judge_plan(case, [winning.uci(), "skip"])["legal"])


if __name__ == "__main__":
    unittest.main()
