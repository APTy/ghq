"""CPU search regressions. Run: python3 -m unittest discover -s tests -v"""

import json
from pathlib import Path
import random
import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "public"))
import engine as e
import search


def position(pieces, turn=e.RED):
    board = e.BaseBoard()
    board.clear_board()
    for square, symbol in pieces.items():
        board.set_piece_at(e.parse_square(square), e.Piece.from_symbol(symbol))
    board.turn = turn
    return board


def analyze(board, **options):
    return json.loads(search.analyze_json(board.serialize(), **options))


class SearchTests(unittest.TestCase):
    def test_capture_setups_find_reported_recapture_with_old_values(self):
        board = e.BaseBoard()
        turns = (Path(__file__).parent / "fixtures" / "paratrooper-recapture.pgn").read_text().splitlines()
        for turn in turns[:20]:
            before_drop = board.copy()
            board = self.assert_legal_line(board, turn.split()[1:])
        with patch.dict(search.VALUES, {e.AIRBORNE_INFANTRY: 4.0}), \
                patch.object(search, "AIRBORNE_HOME_BONUS", 0.0), \
                patch.object(search, "AIRBORNE_POSITION_SCALE", 1.0):
            for probe in (board, board.mirror()):
                before = probe.serialize()
                runner = search._Search(probe, 2400, 4)
                paths = runner.capture_setups(probe)
                victim = e.A1 if probe.turn == e.RED else e.H8
                captures = [path for path in paths if e.Move.from_uci(path[-1]).capture_preference == victim]
                self.assertTrue(captures)
                for path in paths:
                    self.assert_legal_line(probe, path)
                if probe.turn == e.RED:
                    self.assertIn(("rib1", "b3a2xa1"), captures)
                # Even the narrow opponent beam must retain a completed reply
                # that captures the jumper, not merely discover a setup.
                plans = runner.plans(probe)
                self.assertTrue(any(not plan.board.occupied_co[not probe.turn] & e.BB_SQUARES[victim]
                                    for plan in plans))
                self.assertEqual(before, probe.serialize())
            result = analyze(before_drop, time_ms=2400, max_depth=2)
            self.assertEqual(result["depth"], 2)
            after_reply = self.assert_legal_line(before_drop, result["lines"][0]["moves"])
            if "c8a1xa2" in result["turnMoves"]:
                # Choosing the trade is allowed; assuming the jumper survives
                # Red's reply is the tactical regression being tested.
                self.assertFalse(after_reply.airborne_infantry & after_reply.occupied_co[e.BLUE] & e.BB_SQUARES[e.A1])

    def test_capture_setups_can_coordinate_three_actions(self):
        board = position({"a1": "Q", "h8": "q", "d4": "i", "e5": "i", "c5": "i",
                          "d5": "I", "c3": "I", "e3": "I", "b1": "P"})
        for probe in (board, board.mirror()):
            paths = search._Search(probe, 2400, 4).capture_setups(probe)
            victim = e.D4 if probe.turn == e.RED else 63 - e.D4
            self.assertTrue(any(len(path) == 3 and e.Move.from_uci(path[-1]).capture_preference == victim
                                for path in paths))
            for path in paths:
                self.assert_legal_line(probe, path)

    def test_topology_cache_preserves_scores_across_legal_successors(self):
        board = e.BaseBoard()
        rng = random.Random(4173)
        cache = {}
        for _ in range(60):
            if search._terminal(board):
                break
            moves = list(board.generate_legal_moves())
            for move in moves:
                child = board.copy()
                child.push(move)
                before = child.serialize()
                self.assertEqual(search.evaluate(child), search.evaluate(child, cache))
                self.assertEqual(before, child.serialize())
                self.assertLessEqual(len(cache), search.CLUSTER_CACHE_LIMIT)
            board.push(rng.choice(moves))

    def test_topology_cache_eviction_preserves_score(self):
        board = e.BaseBoard()
        cache = {i: () for i in range(search.CLUSTER_CACHE_LIMIT)}
        self.assertEqual(search.evaluate(board), search.evaluate(board, cache))
        self.assertEqual(len(cache), 1)

    def assert_legal_line(self, board, moves):
        result = board.copy()
        for index, uci in enumerate(moves):
            self.assertIsNone(search._terminal(result), "line continued after terminal position")
            move = e.Move.from_uci(uci)
            self.assertTrue(result.is_legal(move), (index, uci, result.board_fen()))
            result.push(move)
        return result

    def test_evaluation_is_pure_and_color_symmetric(self):
        board = e.BaseBoard()
        rng = random.Random(17)
        for _ in range(40):
            before = board.serialize()
            stack = list(board.move_stack)
            value = search.evaluate(board)
            self.assertEqual(before, board.serialize())
            self.assertEqual(stack, board.move_stack)
            self.assertAlmostEqual(value, -search.evaluate(board.mirror()))
            if search._terminal(board):
                break
            board.push(rng.choice(list(board.generate_legal_moves())))

    def test_safe_destinations_match_rules_across_seeded_positions(self):
        board = e.BaseBoard()
        rng = random.Random(260121)
        for _ in range(40):
            if search._terminal(board):
                break
            for color in (e.RED, e.BLUE):
                probe = board.copy()
                probe.turn = color
                probe.turn_moves = 1
                probe.turn_pieces = 0
                legal = list(probe.generate_legal_moves())
                mobility = search._mobility_masks(probe, color)
                for square in e.scan_forward(probe.occupied_co[color]):
                    destinations = {move.to_square for move in legal
                                    if move.from_square == square and move.to_square != square}
                    self.assertEqual(destinations, set(e.scan_forward(search._safe_destinations(probe, square, color))))
                    self.assertEqual(destinations, set(e.scan_forward(
                        search._safe_destinations(probe, square, color, mobility))))
            board.push(rng.choice(list(board.generate_legal_moves())))

    def test_reserves_are_material_before_deployment(self):
        board = position({"a1": "Q", "h8": "q"})
        before = search.evaluate(board)
        board.add_to_reserve(e.ARTILLERY, e.RED)
        reserve_value = search.evaluate(board) - before
        self.assertAlmostEqual(reserve_value, 2.55)
        board.push(e.Move.reinforce(e.ARTILLERY, e.D1))
        self.assertLess(search.evaluate(board) - before - reserve_value, 1.0)

    def test_key_distinguishes_complete_rule_state(self):
        board = e.BaseBoard()
        original = search.state_key(board)
        changes = (
            lambda b: b.add_to_reserve(e.INFANTRY, e.RED),
            lambda b: setattr(b, "turn_moves", 1),
            lambda b: setattr(b, "turn_auto_moves", 1),
            lambda b: setattr(b, "turn_pieces", e.BB_SQUARES[e.G1]),
            lambda b: setattr(b, "free_capture_clusters", 1),
            lambda b: setattr(b, "free_capture_enemies", 1),
            lambda b: setattr(b, "free_capture_num_allowed", 1),
            lambda b: setattr(b, "did_offer_draw", True),
            lambda b: setattr(b, "did_accept_draw", True),
        )
        for change in changes:
            child = board.copy()
            change(child)
            self.assertNotEqual(original, search.state_key(child))
        board.move_stack.append(e.Move.skip())
        self.assertEqual(original, search.state_key(board))

    def test_immediate_hq_capture_for_both_colors(self):
        red = position({"a1": "Q", "h8": "q", "g8": "I", "g6": "I"})
        for board, score in ((red, search.WIN), (red.mirror(), -search.WIN)):
            result = analyze(board, time_ms=200)
            self.assertEqual(result["score"], score)
            self.assertEqual(len(result["turnMoves"]), 1)
            final = self.assert_legal_line(board, result["lines"][0]["moves"])
            self.assertIsNotNone(search._terminal(final))

    def test_two_action_hq_combination_keeps_same_player(self):
        red = position({"a1": "Q", "e5": "q", "c4": "I", "e3": "I"})
        for board, score in ((red, search.WIN), (red.mirror(), -search.WIN)):
            self.assertFalse(any(m.capture_preference is not None and
                                 board.hq & e.BB_SQUARES[m.capture_preference]
                                 for m in board.generate_legal_moves()))
            result = analyze(board, time_ms=300)
            self.assertEqual(result["score"], score)
            self.assertGreaterEqual(len(result["turnMoves"]), 2)
            self.assertIsNotNone(search._terminal(self.assert_legal_line(board, result["turnMoves"])))

    def test_hq_evades_opponents_coordinated_capture(self):
        board = position({"e5": "Q", "a1": "q", "c4": "i", "e3": "i"})
        result = analyze(board, time_ms=1000, max_depth=2)
        self.assertGreater(result["score"], -search.WIN)
        after = self.assert_legal_line(board, result["turnMoves"])
        self.assertFalse(after.hq & e.BB_SQUARES[e.E5])
        reply = analyze(after, time_ms=1000, max_depth=1)
        self.assertGreater(reply["score"], -search.WIN)

    def test_live_paratrooper_sacrifice_regression(self):
        # Reported live game: Blue played c8g1 a7b6 rrf8 and lost its
        # paratrooper to Red's mandatory sfg1 before Red spent an action.
        board = e.BaseBoard()
        for turn in ("rhe1 rtd1 rrc1", "rhe8 rpc8 rtg8", "rpa1 f2f1 g1f2↑"):
            board = self.assert_legal_line(board, turn.split())
        unsafe = self.assert_legal_line(board, "c8g1 a7b6 rrf8".split())
        self.assertIn("sfg1", [m.uci() for m in unsafe.generate_legal_moves()])
        # The prefix itself must reflect free-capture danger, otherwise every
        # surviving beam entry can share the sacrifice before turn completion.
        prefix = self.assert_legal_line(board, ["c8g1"])
        self.assertGreater(search.evaluate(prefix), search.evaluate(board) + 1)
        result = analyze(board, time_ms=2500, max_depth=2)
        self.assertEqual(result["depth"], 2)
        self.assertEqual(len(result["turnMoves"]), 3)
        self.assertNotIn("skip", result["turnMoves"])
        self.assertNotIn("c8g1", result["turnMoves"])
        after = self.assert_legal_line(board, result["turnMoves"])
        self.assertFalse(any(m.name == "AutoCapture" and
                             after.piece_type_at(m.capture_preference) == e.AIRBORNE_INFANTRY
                             for m in after.generate_legal_moves()))

    def test_capture_reordering_does_not_eliminate_quiet_turns(self):
        board = e.BaseBoard()
        for turn in ("rhe1 rtd1 rrc1", "rhd8 rpc8 rte8", "rrf1 e1e2↑ f2e1"):
            board = self.assert_legal_line(board, turn.split())
        for probe in (board, board.mirror()):
            result = analyze(probe, time_ms=3000, max_depth=2)
            self.assertEqual(result["depth"], 2)
            after = self.assert_legal_line(probe, result["turnMoves"])
            # Keep the airborne unit on its own back rank instead of making
            # the same losing capture first, second, or third in the turn.
            home = e.BB_RANK_1 if probe.turn == e.RED else e.BB_RANK_8
            self.assertTrue(after.airborne_infantry & after.occupied_co[probe.turn] & home)
            scores = [line["score"] for line in result["lines"]]
            self.assertEqual(scores, sorted(scores, reverse=probe.turn == e.RED))

    def test_paradrop_compares_heavy_and_armored_capture_targets(self):
        board = e.BaseBoard()
        for turn in ("rhe1 rtd1 rrf1", "rhd8 rpc8 rte8", "rrc1 rpb1 e1e2↑"):
            board = self.assert_legal_line(board, turn.split())
        # Different deployments formerly filled the beam with the same d1
        # capture. Both victims must survive even the initial narrow pass.
        for width in (4, 8):
            plans = search._Search(board, 10000, width).plans(board, root=True)
            targets = {capture[2] for plan in plans for capture in plan.captures}
            self.assertTrue({e.D1, e.E2}.issubset(targets))
        for probe in (board, board.mirror()):
            result = analyze(probe, time_ms=3000, max_depth=2)
            self.assertEqual(result["depth"], 2)
            after = self.assert_legal_line(probe, result["turnMoves"])
            self.assertFalse(after.heavy_artillery & after.occupied_co[not probe.turn])
            self.assertTrue(after.armored_artillery & after.occupied_co[not probe.turn])

    def test_future_free_capture_allows_victim_to_escape(self):
        board = e.BaseBoard()
        for turn in ("rhe1 rtd1 rrf1", "rhd8 rpc8 rte8", "rrc1 rpb1 e1e2↑",
                     "rff8 a7b6 c8e1xd1"):
            board = self.assert_legal_line(board, turn.split())
        for probe in (board, board.mirror()):
            square = e.E2 if probe.turn == e.RED else 63 - e.E2
            self.assertTrue(any(m.from_square == square and m.to_square != square
                                for m in probe.generate_legal_moves()))
            movable = search.evaluate(probe)
            locked = probe.copy()
            locked.turn_pieces |= e.BB_SQUARES[square]
            sign = 1 if probe.turn == e.RED else -1
            self.assertGreater(sign * (movable - search.evaluate(locked)), 1.0)

    def test_red_finds_capture_after_reported_bad_paradrop(self):
        board = e.BaseBoard()
        for turn in ("rhe1 rtd1 rrc1", "rhd8 rpc8 rte8", "rrf1 e1e2↑ f2e1", "c8b1xc1 a7b6 rrf8"):
            board = self.assert_legal_line(board, turn.split())
        for probe in (board, board.mirror()):
            result = analyze(probe, time_ms=3000, max_depth=2)
            after = self.assert_legal_line(probe, result["turnMoves"])
            victim = after.airborne_infantry & after.occupied_co[not probe.turn]
            # A direct two-reinforcement capture is even faster than the
            # user's two-battery trap; either tactical solution is sound.
            if victim:
                square = victim.bit_length() - 1
                self.assertTrue(victim & after.bombarded_co[probe.turn])
                self.assertEqual(search._safe_destinations(after, square, not probe.turn), 0)

    def test_two_battery_trap_accounts_for_no_escape(self):
        board = e.BaseBoard()
        for turn in ("rhe1 rtd1 rrc1", "rhd8 rpc8 rte8", "rrf1 e1e2↑ f2e1",
                     "c8b1xc1 a7b6 rrf8", "d1c1← e2d2← skip"):
            board = self.assert_legal_line(board, turn.split())
        self.assertEqual(search._safe_destinations(board, e.B1, e.BLUE), 0)
        self.assertFalse(any(move.from_square == e.B1 for move in board.generate_legal_moves()))
        trapped_score = search.evaluate(board)
        without = board.copy()
        without.set_piece_at(e.B1, None)
        # A doomed four-point piece must not retain most of its material value.
        self.assertLess(abs(search.evaluate(without) - trapped_score), 1.5)
        self.assertAlmostEqual(trapped_score, -search.evaluate(board.mirror()))

    def test_reachable_support_improves_deep_infantry_position(self):
        board = position({"a1": "Q", "h8": "q", "e6": "P", "d7": "i", "f7": "i", "d2": "I", "f2": "I"})
        unsupported = search.evaluate(board)
        board.set_piece_at(e.D2, None)
        board.set_piece_at(e.F2, None)
        board.set_piece_at(e.D5, e.Piece(e.INFANTRY, e.RED))
        board.set_piece_at(e.F5, e.Piece(e.INFANTRY, e.RED))
        self.assertGreater(search.evaluate(board), unsupported + 1)
        self.assertAlmostEqual(search.evaluate(board), -search.evaluate(board.mirror()))

    def test_home_rank_paratrooper_threatens_exposed_battery(self):
        board = position({"a1": "Q", "a8": "q", "c8": "p", "d4": "R↑"})
        self.assertTrue(search._artillery_capture_threats(board, e.BLUE) & e.BB_SQUARES[e.D4])
        board.set_piece_at(e.C8, None)
        board.set_piece_at(e.C7, e.Piece(e.AIRBORNE_INFANTRY, e.BLUE))
        self.assertFalse(search._artillery_capture_threats(board, e.BLUE) & e.BB_SQUARES[e.D4])

    def test_engaged_armor_cannot_threaten_through_forbidden_path(self):
        board = position({"a1": "Q", "h8": "q", "d4": "F", "c4": "i", "c5": "i", "c6": "r↑"})
        board.turn_moves = 1
        for probe in (board, board.mirror()):
            target = probe.artillery & probe.occupied_co[not probe.turn]
            self.assertFalse(any(move.capture_preference is not None for move in probe.generate_legal_moves()))
            self.assertFalse(search._artillery_capture_threats(probe, probe.turn) & target)

    def test_capturable_checking_gun_has_less_pressure_until_attacker_is_used(self):
        board = position({"h1": "Q", "a8": "q", "e4": "h↘", "e2": "I"})
        board.turn_moves = 1
        for probe in (board, board.mirror()):
            attacker = probe.turn
            target = probe.heavy_artillery
            hq = probe.hq & probe.occupied_co[attacker]
            infantry = probe.infantry & probe.occupied_co[attacker]
            self.assertTrue(any(move.capture_preference is not None and
                                e.BB_SQUARES[move.capture_preference] & target
                                for move in probe.generate_legal_moves()))
            threats = [search._artillery_capture_threats(probe, color) for color in (e.RED, e.BLUE)]
            self.assertTrue(threats[attacker] & target)
            self.assertFalse(threats[not attacker] & target)
            self.assertFalse(search._credible_fire(probe, threats)[0][not attacker] & hq)
            vulnerable_score = search.evaluate(probe)
            probe.turn_pieces = infantry
            self.assertFalse(any(move.capture_preference is not None for move in probe.generate_legal_moves()))
            threats = [search._artillery_capture_threats(probe, color) for color in (e.RED, e.BLUE)]
            self.assertTrue(search._credible_fire(probe, threats)[0][not attacker] & hq)
            sign = 1 if attacker == e.RED else -1
            self.assertGreater(sign * vulnerable_score, sign * search.evaluate(probe) + 5)

    def test_reliable_second_battery_preserves_shared_hq_pressure(self):
        board = position({"h1": "Q", "a8": "q", "e4": "h↘", "h3": "r↓", "e2": "I"})
        threats = [search._artillery_capture_threats(board, color) for color in (e.RED, e.BLUE)]
        self.assertTrue(threats[e.RED] & e.BB_SQUARES[e.E4])
        self.assertFalse(threats[e.RED] & e.BB_SQUARES[e.H3])
        self.assertTrue(search._credible_fire(board, threats)[0][e.BLUE] & e.BB_SQUARES[e.H1])

    def test_live_two_artillery_hang_is_rejected_for_both_colors(self):
        # User's turn22: d5e4↘ e7e7↘ ria8 offered both large guns. The
        # heavy's speculative check used to outweigh its certain capture,
        # pruning safe plans before reply search could compare them.
        board = e.BaseBoard("1iffi1r↓r↓/qiiit↓3/r↓1i2R↑2/3h↓2H↑R↑/6R↖1/1FF3I1/4II2/1P5Q IIIIIF ii b")
        hanging = self.assert_legal_line(board, ["d5e4↘"])
        self.assertGreater(search.evaluate(hanging), search.evaluate(board) + 2)
        for probe in (board, board.mirror()):
            result = analyze(probe, time_ms=3000, max_depth=2)
            self.assertEqual(result["depth"], 2)
            after = self.assert_legal_line(probe, result["turnMoves"])
            # Resolve the unrelated gun already lost to bombardment, then
            # inspect authoritative legal captures of the protected batteries.
            while True:
                legal = list(after.generate_legal_moves())
                if not legal or legal[0].name != "AutoCapture":
                    break
                after.push(legal[0])
            large_guns = ((after.heavy_artillery | after.armored_artillery) &
                          after.occupied_co[probe.turn])
            self.assertEqual(large_guns.bit_count(), 2)
            self.assertFalse(any(move.capture_preference is not None and
                                 e.BB_SQUARES[move.capture_preference] & large_guns
                                 for move in legal))

    def test_profitable_paradrop_still_captures_exposed_artillery(self):
        board = position({"h1": "Q", "a8": "q", "a1": "P", "d5": "r↓"})
        result = analyze(board, time_ms=1000, max_depth=2)
        after = self.assert_legal_line(board, result["turnMoves"])
        self.assertFalse(after.occupied_co[e.BLUE] & after.artillery)
        self.assertTrue(after.occupied_co[e.RED] & after.airborne_infantry)
        self.assertTrue(any(e.Move.from_uci(uci).capture_preference == e.D5 for uci in result["turnMoves"]))

    def test_paratrooper_prefers_preserving_jump_over_quiet_centralization(self):
        home = position({"a1": "Q", "h8": "q", "b1": "P"})
        central = position({"a1": "Q", "h8": "q", "d4": "P"})
        for at_home, advanced in ((home, central), (home.mirror(), central.mirror())):
            sign = 1 if at_home.turn == e.RED else -1
            self.assertGreater(sign * search.evaluate(at_home), sign * search.evaluate(advanced))
            result = analyze(at_home, time_ms=1000, max_depth=1)
            after = self.assert_legal_line(at_home, result["turnMoves"])
            home_rank = e.BB_RANK_1 if at_home.turn == e.RED else e.BB_RANK_8
            self.assertTrue(after.airborne_infantry & after.occupied_co[at_home.turn] & home_rank)

    def test_airborne_preserves_jump_instead_of_equal_material_sacrifice(self):
        board = position({"a1": "Q", "h8": "q", "b1": "P", "e5": "h↑", "d5": "i", "f5": "i"})
        result = analyze(board, time_ms=1000, max_depth=2)
        after = self.assert_legal_line(board, result["turnMoves"])
        # Heavy artillery and airborne infantry now both have base value 5.
        # Giving up the home-rank jump option makes this sacrifice unattractive.
        self.assertTrue(after.heavy_artillery & after.occupied_co[e.BLUE])
        self.assertTrue(after.airborne_infantry & after.occupied_co[e.RED] & e.BB_RANK_1)

    def test_overlapping_threats_do_not_make_material_negative(self):
        board = position({"a1": "Q", "h8": "q", "d4": "R↑", "c4": "i", "e4": "i"})
        board.turn_moves = 1
        with_battery = search.evaluate(board)
        board.set_piece_at(e.D4, None)
        self.assertGreater(with_battery, search.evaluate(board))

    def test_equal_free_capture_choices_remain_color_symmetric(self):
        board = position({"a1": "Q", "h8": "q", "c3": "I", "e3": "I", "d3": "i",
                          "c4": "r↑", "e4": "r↑", "c6": "R↓"})
        board.turn_moves = 2
        board.turn_pieces = e.BB_SQUARES[e.C3] | e.BB_SQUARES[e.E3]
        self.assertAlmostEqual(search.evaluate(board), -search.evaluate(board.mirror()))

    def test_artillery_values_forward_board_control(self):
        board = position({"a1": "Q", "h8": "q", "d4": "R↑"})
        forward = search.evaluate(board)
        board.set_piece_at(e.D4, e.Piece(e.ARTILLERY, e.RED, e.ORIENT_S))
        self.assertGreater(forward, search.evaluate(board))

    def test_mandatory_bombardment_wins_without_extra_action(self):
        board = position({"a1": "Q", "h8": "q", "h6": "R↑"})
        result = analyze(board)
        self.assertEqual(result["turnMoves"], ["sbh8"])
        self.assertEqual(result["score"], search.WIN)
        self.assert_legal_line(board, result["turnMoves"])

    def test_due_hq_bombardment_wins_before_vulnerable_battery_can_be_captured(self):
        board = position({"a8": "Q", "h1": "q", "e4": "H↘", "e2": "i"})
        for probe, score in ((board, search.WIN), (board.mirror(), -search.WIN)):
            self.assertTrue(search._artillery_capture_threats(probe, not probe.turn) & probe.heavy_artillery)
            result = analyze(probe)
            self.assertEqual(result["score"], score)
            self.assertEqual(len(result["turnMoves"]), 1)
            self.assertEqual(e.Move.from_uci(result["turnMoves"][0]).name, "AutoCapture")
            self.assertIsNotNone(search._terminal(self.assert_legal_line(probe, result["turnMoves"])))

    def test_automatic_captures_do_not_consume_three_actions(self):
        board = position({"a1": "Q", "h8": "q", "d6": "i", "d4": "R↑", "b2": "I", "c2": "I"})
        result = analyze(board, time_ms=300, max_depth=1)
        self.assertEqual(result["turnMoves"][0], "sbd6")
        final = self.assert_legal_line(board, result["turnMoves"])
        self.assertNotEqual(final.turn, board.turn)
        self.assertLessEqual(sum(e.Move.from_uci(m).name != "AutoCapture" for m in result["turnMoves"]), 3)

    def test_midturn_plan_respects_used_pieces_and_remaining_actions(self):
        board = e.BaseBoard()
        board.push(e.Move.from_uci("f2f3"))
        board.push(e.Move.from_uci("h2h3"))
        result = analyze(board, time_ms=300, max_depth=2)
        self.assertEqual(len(result["turnMoves"]), 1)
        final = self.assert_legal_line(board, result["turnMoves"])
        self.assertEqual(final.turn, e.BLUE)
        for line in result["lines"]:
            self.assert_legal_line(board, line["moves"])

    def test_search_replies_and_streams_completed_turn_depths(self):
        board = e.BaseBoard()
        updates = []
        result = analyze(board, time_ms=3000, max_depth=2, on_update=lambda s: updates.append(json.loads(s)))
        self.assertEqual(result["depth"], 2)
        self.assertEqual([update["depth"] for update in updates], [1, 2, 2])
        self.assertGreater(len(result["lines"][0]["moves"]), len(result["turnMoves"]))
        for line in result["lines"]:
            self.assert_legal_line(board, line["moves"])
        self.assertEqual(result["state"], board.serialize())
        self.assertTrue(result["selective"])

    def test_zero_budget_returns_legal_complete_emergency_plan(self):
        board = e.BaseBoard()
        result = analyze(board, time_ms=0)
        self.assertTrue(result["timedOut"])
        self.assertEqual(result["depth"], 0)
        self.assertEqual(result["turnMoves"], ["skip"])
        self.assertEqual(self.assert_legal_line(board, result["turnMoves"]).turn, e.BLUE)

    def test_zero_budget_still_resolves_mandatory_captures(self):
        board = position({"a1": "Q", "h8": "q", "d6": "i", "d4": "R↑"})
        result = analyze(board, time_ms=0)
        self.assertEqual(result["turnMoves"], ["sbd6", "skip"])
        self.assert_legal_line(board, result["turnMoves"])

    def test_terminal_results_have_no_moves(self):
        for pieces, winner, score in (({"a1": "Q"}, "RED", search.WIN), ({"h8": "q"}, "BLUE", -search.WIN)):
            board = position(pieces)
            result = analyze(board)
            self.assertEqual(result["terminal"]["winner"], winner)
            self.assertEqual(result["score"], score)
            self.assertIsNone(result["bestMove"])
            self.assertEqual(result["turnMoves"], [])
            self.assertEqual(result["lines"], [])

    def test_arena_trapped_hq_terminal_matches_rules(self):
        # Final position from the paired arena: Blue's HQ still exists but
        # every exit is occupied or bombarded and Skip is its only action.
        board = e.BaseBoard("1q6/IFF5/3R↖R↖3/3H↖4/6P1/3R↖2I1/5II1/3IF1IQ - - b")
        for probe, winner, score in ((board, "RED", search.WIN), (board.mirror(), "BLUE", -search.WIN)):
            before = probe.serialize()
            self.assertEqual([move.uci() for move in probe.copy().generate_legal_moves()], ["skip"])
            self.assertIsNotNone(probe.copy().outcome())
            result = analyze(probe)
            self.assertEqual(result["terminal"], {"winner": winner, "reason": "hq-capture"})
            self.assertEqual(result["score"], score)
            self.assertIsNone(result["bestMove"])
            self.assertEqual(result["lines"], [])
            self.assertEqual(search.evaluate(probe), score)
            self.assertEqual(probe.serialize(), before)

    def test_only_skip_stalemate_matches_rules_in_both_colors(self):
        board = position({"a1": "Q", "h8": "q", "a2": "i", "b1": "i", "b2": "i"})
        for probe in (board, board.mirror()):
            self.assertEqual(probe.copy().outcome().termination, "stalemate")
            before = probe.serialize()
            result = analyze(probe)
            self.assertEqual(result["terminal"], {"winner": None, "reason": "stalemate"})
            self.assertEqual(result["score"], 0)
            self.assertIsNone(result["bestMove"])
            self.assertEqual(probe.serialize(), before)

    def test_trapped_hq_with_available_reinforcement_is_not_terminal(self):
        board = position({"a1": "Q", "h8": "q", "a2": "i", "b1": "i", "b2": "i"})
        board.add_to_reserve(e.INFANTRY, e.RED)
        self.assertIsNone(board.copy().outcome())
        self.assertIsNone(search._terminal(board))

    def test_accepting_draw_is_terminal(self):
        board = e.BaseBoard()
        board.push(e.Move.skip())
        board.push(e.Move.skip())
        result = analyze(board)
        self.assertEqual(result["terminal"], {"winner": None, "reason": "double-skip"})
        self.assertEqual(result["score"], 0)
        self.assertIsNone(result["bestMove"])

    def test_analysis_does_not_mutate_supplied_board_state(self):
        board = e.BaseBoard()
        before = board.serialize()
        result = analyze(board, time_ms=50)
        self.assertEqual(before, board.serialize())
        self.assertEqual(before, result["state"])
        self.assertEqual(board.move_stack, [])
        self.assert_legal_line(board, result["turnMoves"])


if __name__ == "__main__":
    unittest.main()
