"""Run with the optional neural environment: python -m unittest discover -s tests."""
from dataclasses import replace
import random
import unittest

try:
    import numpy as np
except ImportError:
    raise unittest.SkipTest("Neural encoding tests require numpy")

from ghqnet.encoding import (ACTION_KINDS, BOARD_CHANNELS, CHANNEL_NAMES,
                             encode_actions, encode_board)
from ghqnet.rules import e


def mirrored_move(move):
    """Independent transform of the actual move; never execute encoded indices."""
    return replace(move,
                   from_square=None if move.from_square is None else 63 - move.from_square,
                   to_square=None if move.to_square is None else 63 - move.to_square,
                   capture_preference=None if move.capture_preference is None else 63 - move.capture_preference,
                   orientation=None if move.orientation is None else (move.orientation + 4) % 8)


def reference_board_encoding(board):
    """Encoding-v1 specification using authoritative mirror and scalar bits."""
    canonical = board.mirror() if board.turn == e.BLUE else board
    masks = [getattr(canonical, field) & canonical.occupied_co[color]
             for color in (e.RED, e.BLUE)
             for field in ("hq", "infantry", "armored_infantry", "airborne_infantry",
                           "artillery", "armored_artillery", "heavy_artillery")]
    masks.extend((*canonical.bombarded_co, *canonical.adjacent_infantry_squares_co,
                  canonical.orientation_bit0, canonical.orientation_bit1, canonical.orientation_bit2,
                  canonical.turn_pieces, canonical.free_capture_clusters,
                  canonical.free_capture_enemies, canonical.free_capture_num_allowed))
    planes = [[[float(bool(mask & (1 << (row * 8 + column)))) for column in range(8)]
               for row in range(8)] for mask in masks]
    scalars = [canonical.reserves[color].get_count(kind) / 8
               for color in (e.RED, e.BLUE) for kind in e.PIECE_TYPES if kind != e.HQ]
    scalars.extend((canonical.turn_moves / 3, canonical.turn_auto_moves / 64,
                    float(canonical.did_offer_draw), float(canonical.did_accept_draw)))
    planes.extend([[value] * 8 for _ in range(8)] for value in scalars)
    return np.asarray(planes, dtype=np.float32)


class NeuralEncodingTests(unittest.TestCase):
    def test_shapes_dtype_and_empty_action_list(self):
        board = e.BaseBoard()
        encoded = encode_board(board)
        self.assertEqual(encoded.shape, (BOARD_CHANNELS, 8, 8))
        self.assertEqual(BOARD_CHANNELS, 41)
        self.assertEqual(encoded.dtype, np.float32)
        self.assertEqual(len(set(CHANNEL_NAMES)), BOARD_CHANNELS)
        self.assertEqual(encode_actions(board, []).shape, (0, 6))
        self.assertEqual(encode_actions(board, []).dtype, np.int64)

    def test_every_piece_and_orientation_is_visible(self):
        board = e.BaseBoard()
        board.clear_board()
        for color in (e.RED, e.BLUE):
            for kind in e.PIECE_TYPES:
                square = (8 if color == e.RED else 48) + kind - 1
                orientation = kind % 8 if kind >= e.ARTILLERY else None
                board.set_piece_at(square, e.Piece(kind, color, orientation))
        encoded = encode_board(board)
        for color in (e.RED, e.BLUE):
            for kind in e.PIECE_TYPES:
                square = (8 if color == e.RED else 48) + kind - 1
                plane = encoded[int(color) * 7 + kind - 1]
                self.assertEqual(plane.sum(), 1)
                self.assertEqual(plane[square // 8, square % 8], 1)
        original = encode_board(board)
        board.set_piece_at(e.E2, e.Piece(e.ARTILLERY, e.RED, e.ORIENT_N))
        self.assertFalse(np.array_equal(original, encode_board(board)))

    def test_all_midturn_reserve_and_draw_fields_are_represented(self):
        board = e.BaseBoard()
        baseline = encode_board(board)
        mutations = {
            "own_bombardment": lambda b: b.bombarded_co.__setitem__(e.RED, b.bombarded_co[e.RED] ^ e.BB_SQUARES[e.D4]),
            "enemy_bombardment": lambda b: b.bombarded_co.__setitem__(e.BLUE, b.bombarded_co[e.BLUE] ^ e.BB_SQUARES[e.D4]),
            "own_infantry_adjacency": lambda b: b.adjacent_infantry_squares_co.__setitem__(e.RED, b.adjacent_infantry_squares_co[e.RED] ^ e.BB_SQUARES[e.D4]),
            "enemy_infantry_adjacency": lambda b: b.adjacent_infantry_squares_co.__setitem__(e.BLUE, b.adjacent_infantry_squares_co[e.BLUE] ^ e.BB_SQUARES[e.D4]),
            "used_pieces": lambda b: setattr(b, "turn_pieces", e.BB_SQUARES[e.G1]),
            "free_capture_clusters": lambda b: setattr(b, "free_capture_clusters", e.BB_SQUARES[e.D4]),
            "free_capture_enemies": lambda b: setattr(b, "free_capture_enemies", e.BB_SQUARES[e.E4]),
            "free_capture_num_allowed": lambda b: setattr(b, "free_capture_num_allowed", e.BB_SQUARES[e.F4]),
            "actions_used": lambda b: setattr(b, "turn_moves", 2),
            "automatic_actions_used": lambda b: setattr(b, "turn_auto_moves", 7),
            "draw_offered": lambda b: setattr(b, "did_offer_draw", True),
            "draw_accepted": lambda b: setattr(b, "did_accept_draw", True),
        }
        for color, side in ((e.RED, "own"), (e.BLUE, "enemy")):
            for kind in range(e.INFANTRY, e.HEAVY_ARTILLERY + 1):
                mutations[f"{side}_reserve_{e.PIECE_NAMES[kind].lower()}"] = (
                    lambda b, color=color, kind=kind: b.reserves[color].add(kind))
        for name, mutate in mutations.items():
            with self.subTest(channel=name):
                changed = board.copy()
                mutate(changed)
                actual = encode_board(changed)
                changed_channels = np.flatnonzero(np.any(actual != baseline, axis=(1, 2))).tolist()
                self.assertEqual(changed_channels, [CHANNEL_NAMES.index(name)])
        # Counts must not be clipped to a binary presence flag or a maximum.
        board.reserves[e.RED].add(e.INFANTRY, 20)
        self.assertGreater(encode_board(board)[CHANNEL_NAMES.index("own_reserve_infantry"), 0, 0], 1)

    def test_canonical_mirror_and_legal_actions_agree_without_mutation(self):
        board = e.BaseBoard()
        rng = random.Random(903)
        for _ in range(24):
            moves = list(board.generate_legal_moves())
            before = board.serialize()
            stack = list(board.move_stack)
            mirror = board.mirror()
            np.testing.assert_array_equal(encode_board(board), encode_board(mirror))
            encoded = encode_actions(board, moves)
            np.testing.assert_array_equal(encoded, encode_actions(mirror, [mirrored_move(m) for m in moves]))
            self.assertEqual(len(set(map(tuple, encoded))), len(moves))
            self.assertEqual(board.serialize(), before)
            self.assertEqual(board.move_stack, stack)
            # The engine should accept original actions and mirrored original
            # actions; canonical descriptors are never sent into real play.
            chosen = rng.choice(moves)
            self.assertTrue(mirror.is_legal(mirrored_move(chosen)))
            board.push(chosen)

    def test_blue_coordinates_orientation_and_action_identity(self):
        board = e.BaseBoard()
        board.turn = e.BLUE
        move = e.Move.move_and_orient(e.B7, e.C6, e.ORIENT_SW)
        np.testing.assert_array_equal(encode_actions(board, [move]), [[e.G2, e.F3, 64, e.ORIENT_NE, 1, 0]])
        moves = [e.Move.skip(), e.Move.move(e.C3, e.D4), e.Move.move(e.C3, e.D4, e.E4),
                 e.Move.auto_capture_bombard(e.E4), e.Move.auto_capture_free(e.E4)]
        moves.extend(e.Move.move_and_orient(e.C3, e.D4, orientation) for orientation in range(8))
        moves.extend(e.Move.reinforce(kind, e.D1, victim) for kind in range(2, 8) for victim in (None, e.D2))
        encoded = encode_actions(board, moves)
        self.assertEqual(len(set(map(tuple, encoded))), len(moves))
        self.assertEqual(encoded[0].tolist(), [64, 64, 64, 8, ACTION_KINDS["Skip"], 0])
        self.assertEqual(encoded[3, 4], ACTION_KINDS["bombard"])
        self.assertEqual(encoded[4, 4], ACTION_KINDS["free"])

    def test_vectorized_features_match_encoding_v1_across_legal_play(self):
        board, rng = e.BaseBoard(), random.Random(9152)
        for step in range(100):
            moves = list(board.generate_legal_moves())
            for candidate in (board, board.mirror()):
                before = candidate.serialize()
                np.testing.assert_array_equal(encode_board(candidate), reference_board_encoding(candidate))
                self.assertTrue(encode_board(candidate).flags.c_contiguous)
                self.assertEqual(candidate.serialize(), before)
            board.push(rng.choice(moves))
            if board.outcome() is not None:
                board = e.BaseBoard()

    def test_blue_bookkeeping_and_orientation_mask_match_previous_format_exactly(self):
        board = e.BaseBoard()
        board.clear_board()
        for square in range(8):
            board.set_piece_at(square + 8, e.Piece(e.ARTILLERY, e.RED, square))
            board.set_piece_at(square + 48, e.Piece(e.HEAVY_ARTILLERY, e.BLUE, square))
        # Serialized rule state can retain non-artillery orientation bits.
        # Existing v1 Red preserves those bits; authoritative Blue mirror
        # masks them. Preserve that behavior so existing weights stay valid.
        board.orientation_bit0 |= e.BB_SQUARES[e.A4]
        board.orientation_bit1 |= e.BB_SQUARES[e.B4]
        board.orientation_bit2 |= e.BB_SQUARES[e.C4]
        board.turn_pieces = e.BB_SQUARES[e.B2] | e.BB_SQUARES[e.H7]
        board.free_capture_clusters = e.BB_SQUARES[e.A1] | e.BB_SQUARES[e.H8]
        board.free_capture_enemies = e.BB_SQUARES[e.G7]
        board.free_capture_num_allowed = e.BB_SQUARES[e.F6]
        board.turn_moves, board.turn_auto_moves = 2, 7
        board.did_offer_draw = board.did_accept_draw = True
        board.reserves[e.BLUE].add(e.INFANTRY, 21)
        for color in (e.RED, e.BLUE):
            board.turn = color
            before, stack = board.serialize(), list(board.move_stack)
            np.testing.assert_array_equal(encode_board(board), reference_board_encoding(board))
            self.assertEqual(board.serialize(), before)
            self.assertEqual(board.move_stack, stack)

    def test_action_encoding_rejects_out_of_range_original_fields(self):
        board = e.BaseBoard()
        for move in (e.Move(name="Move", from_square=-1, to_square=0),
                     e.Move(name="Move", from_square=64, to_square=0),
                     e.Move(name="Move", from_square=0, to_square=64),
                     e.Move(name="Move", from_square=0, to_square=1, capture_preference=64),
                     e.Move(name="MoveAndOrient", from_square=0, to_square=1, orientation=8),
                     e.Move(name="Reinforce", to_square=0, unit_type=8),
                     e.Move(name="unknown")):
            with self.subTest(move=move.name), self.assertRaises(ValueError):
                encode_actions(board, [move])


if __name__ == "__main__":
    unittest.main()
