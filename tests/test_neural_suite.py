from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Neural suite tests require torch")

from ghqnet.data import Sample, write_replay
from ghqnet.rules import e, cpu, copy_board
from ghqnet.suite import (_capture_witness, _case, canonical_state_id, defense_cases,
                         finish_with_skip, immediate_hq_loss, judge_case, mine_replays,
                         mirror_move, play_plan, read_suite, verify_case, write_suite)
from ghqnet.tactics import frozen_results_summary


def position(pieces):
    board = e.BaseBoard()
    board.clear_board()
    for square, symbol in pieces.items():
        board.set_piece_at(e.parse_square(square), e.Piece.from_symbol(symbol))
    return board


def capture_case():
    board = position({"a1": "Q", "h8": "q", "d4": "I", "f5": "h↑"})
    objective, witness, category = _capture_witness(board, [])
    return _case(board, "test:capture", "game-a", {}, objective, witness, category, 4173)


class FrozenSuiteTests(unittest.TestCase):
    def test_capture_witness_is_positive_pure_and_mirror_equivalent(self):
        case = capture_case()
        board = verify_case(case)
        self.assertEqual(canonical_state_id(board), canonical_state_id(board.mirror()))
        before = board.serialize()
        self.assertTrue(judge_case(case, case["witness"])["passed"])
        self.assertFalse(judge_case(case, ["skip"])["passed"])
        mirrored = [mirror_move(move) for move in case["witness"]]
        self.assertTrue(judge_case(case, mirrored, mirrored=True)["passed"])
        self.assertFalse(judge_case(case, case["witness"] + ["skip"])["legal"])
        self.assertEqual(board.serialize(), before)
        board.turn_moves = 1
        self.assertNotEqual(canonical_state_id(board), case["canonical_state_id"])

    def test_two_action_hq_witness_is_exact_and_distinct_from_immediate_capture(self):
        board = position({"a1": "Q", "e5": "q", "c5": "I", "e3": "I"})
        self.assertFalse(any(move.capture_preference is not None for move in board.generate_legal_moves()))
        objective, witness, category = _capture_witness(board, ["c5d5", "e3e4xe5"])
        case = _case(board, "test:two-action", "game-b", {}, objective, witness, category, 4173)
        verify_case(case)
        self.assertEqual(objective["witness_chosen_actions"], 2)
        self.assertEqual(witness, ["c5d5", "e3e4xe5"])
        self.assertFalse(judge_case(case, [witness[0]])["complete"])

    def test_hq_defenses_resolve_real_threats_and_keep_variants_in_one_family(self):
        cases = defense_cases()
        self.assertEqual(len(cases), 24)
        self.assertEqual(len({case["canonical_state_id"] for case in cases}), 24)
        self.assertEqual(len({case["family_id"] for case in cases}), 3)
        for case in cases:
            board = verify_case(case)
            skipped, complete, _ = play_plan(board, ["skip"])
            self.assertTrue(complete)
            self.assertTrue(immediate_hq_loss(skipped, board.turn))
            self.assertIsNone(immediate_hq_loss(skipped, board.turn, max_states=0))
            self.assertFalse(judge_case(case, ["skip"])["passed"])
        for family in {case["family_id"] for case in cases}:
            self.assertEqual(len({case["split"] for case in cases if case["family_id"] == family}), 1)

    def test_frozen_roundtrip_rejects_duplicates_split_leaks_and_bad_witnesses(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "suite.jsonl"
            cases = defense_cases(count=2) + [capture_case()]
            write_suite(path, cases, usage="evaluation_only")
            header, loaded = read_suite(path)
            self.assertEqual(loaded, cases)
            self.assertEqual(header["summary"]["unique_states"], 3)
            with self.assertRaises(ValueError):
                write_suite(path, cases)
            with self.assertRaises(ValueError):
                write_suite(Path(folder) / "duplicate.jsonl", [cases[0], cases[0]])
            leaked = deepcopy(cases)
            leaked[1]["split"] = "dev" if leaked[0]["split"] != "dev" else "test"
            with self.assertRaises(ValueError):
                write_suite(Path(folder) / "leak.jsonl", leaked)
            bad = deepcopy(cases[0]); bad["witness"] = ["skip"]
            with self.assertRaises(ValueError):
                verify_case(bad)

    def test_mining_validates_trajectory_and_does_not_turn_losing_labels_into_truth(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "game.jsonl"
            board = position({"a1": "Q", "e5": "q", "c5": "I", "e3": "I"})
            moves = ["c5d5", "e3e4xe5"]
            samples = []
            for action in moves:
                legal = {move.uci(): move for move in board.generate_legal_moves()}
                samples.append(Sample(board.serialize(), list(legal), [float(name == action) for name in legal],
                                      outcome=2, game_id="example", source="mcts"))
                board.push(legal[action])
            game = {"moves": moves, "final_state": board.serialize(), "terminal": cpu._terminal(board)}
            write_replay(path, samples, game=game)
            cases, candidates, exclusions = mine_replays([path], count=2, defenses=0)
            self.assertEqual(len(cases), 2)
            self.assertEqual(candidates, [])
            self.assertEqual(len(exclusions["canonical_state_ids"]), 2)
            self.assertTrue(all(case["source"]["observed_game_outcome_for_mover"] == 2 for case in cases))
            self.assertTrue(all(judge_case(case, case["witness"])["passed"] for case in cases))
            lines = path.read_text().splitlines()
            header = json.loads(lines[0]); header["metadata"]["game"]["moves"][0] = "skip"
            path.write_text(json.dumps(header) + "\n" + "\n".join(lines[1:]) + "\n")
            with self.assertRaises(ValueError):
                mine_replays([path], count=2, defenses=0)

    def test_candidate_policy_is_unscored_and_mirrors_are_not_independent_successes(self):
        case = capture_case()
        candidate = deepcopy(case)
        candidate.update(evidence="teacher_approximate", objective=None, witness=None)
        verify_case(candidate)
        self.assertFalse(judge_case(candidate, case["witness"])["passed"])
        rows = [{"case_id": "one", "family_id": "family", "category": "capture", "player": color, "passed": passed}
                for color, passed in (("RED", True), ("BLUE", False))]
        report = frozen_results_summary(rows)
        self.assertEqual(report["completed_unique_cases"], 1)
        self.assertEqual(report["passed_both_colors"], 0)
        self.assertEqual(report["categories"]["capture"]["passed_color_runs"], 1)

    def test_empty_mixed_replay_is_explicitly_unscored_without_aborting_collection(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "opponent-won-before-learner.jsonl"
            board = e.BaseBoard()
            board.push(e.Move.skip()); board.push(e.Move.skip())
            write_replay(path, [], game={"moves": ["skip", "skip"], "sample_action_indices": [],
                                         "final_state": board.serialize(), "terminal": cpu._terminal(board)})
            diagnostics = {}
            cases, candidates, exclusions = mine_replays([path], count=1, defenses=0, diagnostics=diagnostics)
            self.assertEqual(cases, [])
            self.assertEqual(candidates, [])
            self.assertEqual(exclusions["canonical_state_ids"], [])
            self.assertEqual(diagnostics["unscored_empty_replays"], [str(path)])


if __name__ == "__main__":
    unittest.main()
