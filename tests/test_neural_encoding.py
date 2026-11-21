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


if __name__ == "__main__":
    unittest.main()
