"""Value filtering must not fabricate labels or infer self-play from row source."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
import tempfile
import unittest

try:
    import numpy
    import torch
except ImportError:
    raise unittest.SkipTest("Neural provenance tests require optional training dependencies")

from ghqnet.data import DATA_VERSION, read_replay, write_replay
from ghqnet.model import rules_source_hash
from ghqnet.rules import e
from ghqnet.selfplay import play_game
from ghqnet.value_targets import audit_value_targets, value_eligible, value_eligibility_reason


class SkipEvaluator:
    def predict(self, board, moves):
        return [float(move.name == "Skip") for move in moves], 0.0


def position():
    board = e.BaseBoard()
    board.clear_board()
    board.set_piece_at(e.parse_square("a1"), e.Piece.from_symbol("Q"))
    board.set_piece_at(e.parse_square("h8"), e.Piece.from_symbol("q"))
    return board


def collect(**kwargs):
    return play_game(SkipEvaluator(), start_board=position(), simulations=1,
                     add_noise=False, temperature_turns=0, max_turns=kwargs.pop("max_turns", 2),
                     model_id=kwargs.pop("model_id", "fixed-checkpoint"), **kwargs)


class ValueProvenanceTests(unittest.TestCase):
    def test_same_model_terminal_records_both_players_and_actual_search(self):
        samples, report = collect()
        provenance = report["value_provenance"]
        self.assertEqual(provenance["collection_kind"], "self")
        self.assertEqual(provenance["terminal"], {"winner": None, "reason": "double-skip"})
        self.assertEqual(provenance["final_state"], report["final_state"])
        self.assertEqual(provenance["players"]["RED"], provenance["players"]["BLUE"])
        self.assertEqual(provenance["players"]["RED"]["search"]["simulations"], 1)
        self.assertFalse(provenance["players"]["RED"]["search"]["root_noise"])
        for sample in samples:
            self.assertEqual(sample.outcome, 1)
            self.assertTrue(value_eligible(sample))
            self.assertTrue(value_eligible(sample, "selfplay"))
            sample.validate()
        samples[0].value_provenance["players"]["RED"]["model_id"] = "mutated"
        self.assertEqual(samples[1].value_provenance, report["value_provenance"])

    def test_legacy_rows_stay_labeled_but_are_unknown_not_inferred_from_source(self):
        samples, _ = collect()
        sample = replace(samples[0], value_provenance=None, source="mcts:fixed-checkpoint")
        original = asdict(sample)
        self.assertTrue(value_eligible(sample, "all"))
        self.assertFalse(value_eligible(sample, "selfplay"))
        self.assertEqual(value_eligibility_reason(sample, "selfplay"), "unknown-provenance")
        self.assertEqual(asdict(sample), original)

    def test_recent_identity_filter_and_unknown_ids_are_explicit(self):
        samples, _ = collect()
        sample = samples[0]
        self.assertTrue(value_eligible(sample, "selfplay", {"fixed-checkpoint"}))
        self.assertFalse(value_eligible(sample, "selfplay", {"older-checkpoint"}))
        self.assertFalse(value_eligible(sample, "selfplay", set()))
        sample.value_provenance["players"]["RED"]["model_id"] = ""
        sample.value_provenance["players"]["BLUE"]["model_id"] = ""
        self.assertEqual(value_eligibility_reason(sample, "selfplay"), "unknown-model")
        with self.assertRaisesRegex(ValueError, "collection"):
            value_eligible(sample, "selfplay", "fixed-checkpoint")

    def test_different_players_or_search_settings_do_not_qualify(self):
        samples, _ = collect()
        sample = samples[0]
        sample.value_provenance["players"]["BLUE"]["model_id"] = "other"
        self.assertEqual(value_eligibility_reason(sample, "selfplay"), "different-models")
        sample.value_provenance["players"]["BLUE"]["model_id"] = "fixed-checkpoint"
        for field, altered in (("simulations", 2), ("root_noise", True), ("temperature_turns", 12),
                               ("cpuct", 2), ("max_depth", 12)):
            variant = deepcopy(sample)
            variant.value_provenance["players"]["BLUE"]["search"][field] = altered
            self.assertEqual(value_eligibility_reason(variant, "selfplay"), "different-search-settings")

    def test_historical_even_same_identity_is_not_same_policy_selfplay(self):
        samples, report = collect(opponent="historical", learner_color="BLUE",
                                  opponent_evaluator=SkipEvaluator(), opponent_model_id="fixed-checkpoint",
                                  opponent_simulations=2)
        self.assertEqual(len(samples), 1)
        provenance = samples[0].value_provenance
        self.assertEqual(provenance["collection_kind"], "historical")
        self.assertEqual(provenance["players"]["RED"]["search"]["simulations"], 2)
        self.assertEqual(provenance["players"]["BLUE"]["search"]["simulations"], 1)
        self.assertTrue(value_eligible(samples[0]))
        self.assertFalse(value_eligible(samples[0], "selfplay"))
        self.assertEqual(provenance, report["value_provenance"])

    def test_cutoff_does_not_become_a_draw_or_eligible_value_target(self):
        samples, report = collect(max_turns=1)
        self.assertIsNone(samples[0].outcome)
        self.assertIsNone(report["value_provenance"]["terminal"])
        self.assertEqual(report["value_provenance"]["termination_reason"], "max-turns")
        self.assertFalse(value_eligible(samples[0], "all"))
        self.assertFalse(value_eligible(samples[0], "selfplay"))
        variant = deepcopy(samples[0])
        variant.outcome = 1
        with self.assertRaisesRegex(ValueError, "terminal and outcome"):
            value_eligible(variant)

    def test_malformed_or_false_provenance_raises_even_in_default_mode(self):
        samples, _ = collect()
        mutations = [
            lambda p: p.update(version=True),
            lambda p: p.update(collection_kind="unknown"),
            lambda p: p.update(termination_reason="max-turns"),
            lambda p: p.update(final_state=position().serialize()),
            lambda p: p["players"]["RED"]["search"].update(simulations=-1),
            lambda p: p["players"]["RED"]["search"].update(root_noise=1),
            lambda p: p["players"]["RED"]["search"].update(cpuct=float("nan")),
            lambda p: p.update(extra="not in version1"),
            lambda p: p["players"]["RED"].update(model_id=3),
        ]
        for mutate in mutations:
            with self.subTest(mutation=mutate):
                sample = deepcopy(samples[0])
                mutate(sample.value_provenance)
                with self.assertRaises(ValueError):
                    value_eligible(sample, "all")
        mislabeled = replace(samples[0], outcome=0)
        with self.assertRaisesRegex(ValueError, "sample mover"):
            value_eligible(mislabeled)

    def test_winning_partial_turn_preserves_exact_state_and_label_perspective(self):
        board = position()
        board.set_piece_at(e.parse_square("g8"), e.Piece.from_symbol("I"))
        board.set_piece_at(e.parse_square("g6"), e.Piece.from_symbol("I"))
        board.turn_moves = 2
        state = board.serialize()
        samples, report = play_game(SkipEvaluator(), start_state=state, simulations=3,
                                    max_turns=1, model_id="winner")
        self.assertEqual(samples[0].state, state)
        self.assertEqual(samples[0].outcome, 0)
        self.assertEqual(report["terminal"]["winner"], "RED")
        self.assertTrue(value_eligible(samples[0], "selfplay"))
        flipped = e.BaseBoard.deserialize(samples[0].state)
        flipped.turn = e.BLUE
        with self.assertRaisesRegex(ValueError, "sample mover"):
            value_eligible(replace(samples[0], state=flipped.serialize()))

    def test_read_versions_one_two_three_preserve_legacy_and_new_records(self):
        samples, _ = collect()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "replay.jsonl"
            write_replay(path, samples)
            self.assertEqual(json.loads(path.read_text().splitlines()[0])["version"], DATA_VERSION)
            self.assertEqual(read_replay(path), samples)
            for version in (1, 2):
                old = asdict(samples[0])
                old.pop("value_provenance")
                path.write_text(json.dumps({"version": version, "rules_sha256": rules_source_hash()}) + "\n" + json.dumps(old) + "\n")
                restored = read_replay(path)[0]
                self.assertIsNone(restored.value_provenance)
                self.assertEqual(restored.outcome, 1)
                self.assertTrue(value_eligible(restored))
                self.assertFalse(value_eligible(restored, "selfplay"))

    def test_audit_distinguishes_raw_labels_and_filtered_labels_per_source(self):
        samples, _ = collect()
        unknown = replace(samples[0], source="mcts:legacy", value_provenance=None)
        cutoff, _ = collect(max_turns=1)
        report = audit_value_targets([samples[0], unknown, cutoff[0]], "selfplay")
        self.assertEqual(report["totals"]["rows"], 3)
        self.assertEqual(report["totals"]["labeled_rows"], 2)
        self.assertEqual(report["totals"]["eligible_rows"], 1)
        self.assertEqual(report["totals"]["unknown-provenance"], 1)
        self.assertEqual(report["totals"]["unlabeled"], 1)
        self.assertEqual(report["by_source"]["mcts:legacy"]["eligible_rows"], 0)

    def test_empty_audit_still_checks_options_and_reports_zero_rows(self):
        self.assertEqual(audit_value_targets([], "selfplay")["totals"]["eligible_rows"], 0)
        with self.assertRaisesRegex(ValueError, "mode"):
            audit_value_targets([], "guess")
        with self.assertRaisesRegex(ValueError, "collection"):
            audit_value_targets([], "selfplay", "not-a-list")


if __name__ == "__main__":
    unittest.main()
