"""Rule-state invariants relied on by the CPU search and replay analysis."""

import base64
from pathlib import Path
import random
import struct
import sys
import unittest
import zlib

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "public"))
import engine as e


def position(*pieces, turn=e.RED):
    board = e.BaseBoard()
    board.clear_board()
    for square, piece_type, color, *orientation in pieces:
        board.set_piece_at(square, e.Piece(piece_type, color, orientation[0] if orientation else None))
    board.turn = turn
    list(board._generate_free_captures(turn))
    return board


def mirror_move(move):
    result = move.copy()
    for field in ("from_square", "to_square", "capture_preference"):
        square = getattr(result, field)
        if square is not None:
            setattr(result, field, 63 - square)
    if result.orientation is not None:
        result.orientation = (result.orientation + 4) % 8
    return result


def legal(board):
    return {move.uci() for move in board.generate_legal_moves()}


def free_capture_position():
    return position(
        (e.A1, e.HQ, e.RED), (e.H8, e.HQ, e.BLUE),
        (e.C3, e.INFANTRY, e.RED), (e.E3, e.INFANTRY, e.RED),
        (e.D2, e.INFANTRY, e.RED), (e.D3, e.INFANTRY, e.BLUE),
        (e.C2, e.ARTILLERY, e.BLUE, e.ORIENT_W),
    )


class EngineIntegrityTests(unittest.TestCase):
    def assert_mirror_equivalent(self, board):
        expected = {mirror_move(move).uci() for move in board.generate_legal_moves()}
        mirrored = board.mirror()
        self.assertEqual(expected, legal(mirrored))
        self.assertEqual(board._transposition_key(), mirrored.mirror()._transposition_key())
        artillery = mirrored._all_artillery() & mirrored.occupied
        self.assertEqual(0, (mirrored.orientation_bit0 | mirrored.orientation_bit1 |
                             mirrored.orientation_bit2) & ~artillery)

    def test_key_distinguishes_reserves_and_used_pieces(self):
        board = e.BaseBoard()
        no_infantry_reserve = board.copy()
        no_infantry_reserve.reserves[e.RED].remove(e.INFANTRY, board.get_reserve_count(e.INFANTRY, e.RED))
        self.assertNotEqual(board._transposition_key(), no_infantry_reserve._transposition_key())
        self.assertNotEqual(legal(board), legal(no_infantry_reserve))

        used = board.copy()
        used.turn_pieces = e.BB_SQUARES[e.G1]
        self.assertNotEqual(board._transposition_key(), used._transposition_key())
        self.assertNotEqual(legal(board), legal(used))

    def test_key_distinguishes_turn_and_pending_capture_state(self):
        board = e.BaseBoard()
        for field, value in (
            ("turn", e.BLUE), ("turn_moves", 1), ("turn_auto_moves", 1),
            ("free_capture_clusters", e.BB_SQUARES[e.C3]),
            ("free_capture_enemies", e.BB_SQUARES[e.D3]),
            ("free_capture_num_allowed", e.BB_SQUARES[e.E3]),
            ("did_offer_draw", True), ("did_accept_draw", True),
        ):
            with self.subTest(field=field):
                changed = board.copy()
                setattr(changed, field, value)
                self.assertNotEqual(board._transposition_key(), changed._transposition_key())

        pending = free_capture_position()
        exhausted = pending.copy()
        exhausted.free_capture_num_allowed = 0
        self.assertNotEqual(pending._transposition_key(), exhausted._transposition_key())
        self.assertTrue(all(move.name == "AutoCapture" for move in pending.generate_legal_moves()))
        self.assertTrue(any(move.name != "AutoCapture" for move in exhausted.generate_legal_moves()))

    def test_key_distinguishes_cached_legality_masks(self):
        board = e.BaseBoard()
        for field in ("bombarded_co", "adjacent_infantry_squares_co"):
            changed = board.copy()
            getattr(changed, field)[e.BLUE] ^= e.BB_SQUARES[e.A1]
            self.assertNotEqual(board._transposition_key(), changed._transposition_key())

    def test_key_ignores_history_and_irrelevant_old_orientation_bits(self):
        board = e.BaseBoard()
        changed = board.copy()
        changed.move_stack.append(e.Move.skip())
        changed.orientation_bit2 |= e.BB_SQUARES[e.A3]
        self.assertEqual(board._transposition_key(), changed._transposition_key())

    def test_three_actions_keep_same_player_until_turn_ends(self):
        board = e.BaseBoard()
        for index, uci in enumerate(("f2f3", "g2g3", "h2h3")):
            move = e.Move.from_uci(uci)
            self.assertTrue(board.is_legal(move))
            board.push(move)
            self.assertEqual(e.BLUE if index == 2 else e.RED, board.turn)
            self.assertEqual(0 if index == 2 else index + 1, board.turn_moves)
            if index < 2:
                self.assertTrue(board.turn_pieces & e.BB_SQUARES[move.to_square])
                self.assertFalse(any(m.from_square == move.to_square for m in board.generate_legal_moves()))
        self.assertEqual(0, board.turn_pieces)

    def test_bombardments_are_mandatory_and_do_not_consume_actions(self):
        board = position(
            (e.A1, e.HQ, e.RED), (e.H8, e.HQ, e.BLUE),
            (e.D4, e.ARTILLERY, e.RED, e.ORIENT_N),
            (e.D5, e.INFANTRY, e.BLUE), (e.D6, e.INFANTRY, e.BLUE),
        )
        self.assertEqual({"sbd5", "sbd6"}, legal(board))
        for uci in ("sbd5", "sbd6"):
            board.push(e.Move.from_uci(uci))
            self.assertEqual(e.RED, board.turn)
            self.assertEqual(0, board.turn_moves)
            self.assertEqual(0, board.turn_pieces)
        self.assertEqual(2, board.turn_auto_moves)
        self.assertTrue(any(move.name != "AutoCapture" for move in board.generate_legal_moves()))

    def test_free_captures_preserve_turn_and_spend_capture_allowance(self):
        board = free_capture_position()
        initial_allowance = e.popcount(board.free_capture_num_allowed)
        move = next(board.generate_legal_moves())
        self.assertEqual("free", move.auto_capture_type)
        board.push(move)
        self.assertEqual(e.RED, board.turn)
        self.assertEqual(0, board.turn_moves)
        self.assertEqual(1, board.turn_auto_moves)
        self.assertEqual(initial_allowance - 1, e.popcount(board.free_capture_num_allowed))
        self.assertIsNone(board.piece_at(move.capture_preference))

    def test_skip_transfers_turn_and_two_empty_turns_draw(self):
        board = e.BaseBoard()
        board.push(e.Move.skip())
        self.assertEqual(e.BLUE, board.turn)
        self.assertTrue(board.did_offer_draw)
        self.assertIsNone(board.outcome())
        board.push(e.Move.skip())
        self.assertEqual("double-skip", board.outcome().termination)
        self.assertIsNone(board.outcome().winner)

    def test_serialization_preserves_search_state_and_wire_format(self):
        initial = e.BaseBoard()
        midturn = initial.copy()
        midturn.push(e.Move.from_uci("f2f3"))
        captures = free_capture_position()
        captures.push(next(captures.generate_legal_moves()))
        offered_draw = initial.copy()
        offered_draw.push(e.Move.skip())
        for board in (initial, midturn, captures, offered_draw):
            with self.subTest(fen=board.board_fen(), turn_moves=board.turn_moves):
                encoded = board.serialize()
                self.assertEqual(struct.calcsize(">21Q3b12I2b"), len(zlib.decompress(base64.b64decode(encoded))))
                restored = e.BaseBoard.deserialize(encoded)
                self.assertEqual(board._transposition_key(), restored._transposition_key())
                self.assertEqual(legal(board), legal(restored))
                self.assertEqual(encoded, restored.serialize())

    def test_mirror_preserves_midturn_and_pending_capture_legality(self):
        board = e.BaseBoard()
        self.assert_mirror_equivalent(board)
        board.push(e.Move.from_uci("f2f3"))
        self.assert_mirror_equivalent(board)
        board.push(e.Move.from_uci("g1g1↗"))
        self.assert_mirror_equivalent(board)
        captures = free_capture_position()
        self.assert_mirror_equivalent(captures)
        captures.push(next(captures.generate_legal_moves()))
        self.assert_mirror_equivalent(captures)

    def test_mirror_preserves_legality_through_deterministic_game(self):
        board = e.BaseBoard()
        random_source = random.Random(19)
        for _ in range(40):
            self.assert_mirror_equivalent(board)
            if board.outcome() is not None:
                break
            board.push(random_source.choice(list(board.generate_legal_moves())))

    def test_orientation_replacement_and_rotation(self):
        for orientation in range(8):
            board = position((e.D4, e.ARTILLERY, e.RED, orientation))
            for replacement in range(8):
                board.set_orientation(e.D4, replacement)
                self.assertEqual(replacement, board.get_orientation(e.D4))
            board.set_orientation(e.D4, orientation)
            mirrored = board.mirror()
            self.assertEqual((orientation + 4) % 8, mirrored.get_orientation(63 - e.D4))
            rotated = board.rotate_90_clockwise()
            target = e.lsb(e.rotate_90_clockwise(e.BB_SQUARES[e.D4]))
            self.assertEqual((orientation + 2) % 8, rotated.get_orientation(target))
            self.assertEqual(0, (rotated.orientation_bit0 | rotated.orientation_bit1 |
                                 rotated.orientation_bit2) & ~rotated.occupied)

    def test_reinforced_infantry_has_no_orientation_bits(self):
        board = e.BaseBoard()
        board.push(e.Move.skip())
        move = e.Move.reinforce(e.INFANTRY, e.H8)
        self.assertTrue(board.is_legal(move))
        board.push(move)
        self.assertEqual(0, (board.orientation_bit0 | board.orientation_bit1 |
                             board.orientation_bit2) & e.BB_SQUARES[e.H8])

    def test_copy_has_independent_mutable_state(self):
        board = e.BaseBoard()
        before = board.serialize()
        copied = board.copy()
        copied.push(e.Move.reinforce(e.INFANTRY, e.A1))
        copied.bombarded_co[e.RED] = e.BB_ALL
        copied.adjacent_infantry_squares_co[e.BLUE] = e.BB_ALL
        self.assertEqual(before, board.serialize())
        self.assertEqual([], board.move_stack)
        self.assertIsNot(board.reserves[e.RED], copied.reserves[e.RED])


if __name__ == "__main__":
    unittest.main()
