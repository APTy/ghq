"""Lossless rule features and joint legal-action descriptors for the network.

Only model inputs are canonicalized: Blue positions use the engine's 180-degree
rotation plus color swap. Search always applies the original legal Move object.
Rows are ranks (a1 is [0, 0]); channels are listed in CHANNEL_NAMES. Counts use
fixed scaling, never clipping, so reserves and midturn bookkeeping stay visible.
"""
from collections.abc import Sequence

import numpy as np

from .rules import e

ENCODING_VERSION = 1
NO_SQUARE = 64
NO_ORIENTATION = 8
NO_DEPLOYMENT = 0
ACTION_COLUMNS = ("source", "destination", "victim", "orientation", "kind", "deployment")
ACTION_KINDS = {"Move": 0, "MoveAndOrient": 1, "Reinforce": 2,
                "bombard": 3, "free": 4, "Skip": 5}
PIECE_TYPES = tuple(e.PIECE_TYPES)
RESERVE_TYPES = tuple(kind for kind in PIECE_TYPES if kind != e.HQ)
CHANNEL_NAMES = tuple(
    [f"{side}_{e.PIECE_NAMES[kind].lower()}" for side in ("own", "enemy") for kind in PIECE_TYPES]
    + ["own_bombardment", "enemy_bombardment", "own_infantry_adjacency", "enemy_infantry_adjacency"]
    + ["orientation_bit0", "orientation_bit1", "orientation_bit2", "used_pieces",
       "free_capture_clusters", "free_capture_enemies", "free_capture_num_allowed"]
    + [f"{side}_reserve_{e.PIECE_NAMES[kind].lower()}" for side in ("own", "enemy") for kind in RESERVE_TYPES]
    + ["actions_used", "automatic_actions_used", "draw_offered", "draw_accepted"]
)
BOARD_CHANNELS = len(CHANNEL_NAMES)
_SQUARE_INDICES = np.arange(64, dtype=np.uint64)
_MASK_CHANNELS = 25


def encode_board(board) -> np.ndarray:
    """Return float32 [41, 8, 8], with the current player on the Red home rank.

    All rule-relevant serialized fields are represented, including pending free
    captures. Occupancy/color masks are encoded jointly by the 14 piece planes;
    absolute player identity is intentionally removed by canonicalization.
    """
    blue = board.turn == e.BLUE
    colors = (board.turn, not board.turn)
    piece_masks = (board.hq, board.infantry, board.armored_infantry,
                   board.airborne_infantry, board.artillery,
                   board.armored_artillery, board.heavy_artillery)
    masks = [mask & board.occupied_co[color] for color in colors for mask in piece_masks]
    orientations = (board.orientation_bit0, board.orientation_bit1, board.orientation_bit2)
    if blue:
        # Match BaseBoard.mirror exactly: only artillery retains orientation,
        # and adding four modulo eight flips bit2. Reverse square order below
        # instead of copying and transforming a complete rules board.
        oriented = board.occupied & (board.artillery | board.armored_artillery | board.heavy_artillery)
        orientations = (orientations[0] & oriented, orientations[1] & oriented,
                        (orientations[2] ^ e.BB_ALL) & oriented)
    masks.extend((*(board.bombarded_co[color] for color in colors),
                  *(board.adjacent_infantry_squares_co[color] for color in colors),
                  *orientations, board.turn_pieces, board.free_capture_clusters,
                  board.free_capture_enemies, board.free_capture_num_allowed))
    result = np.empty((BOARD_CHANNELS, 64), dtype=np.float32)
    shifts = _SQUARE_INDICES[::-1] if blue else _SQUARE_INDICES
    result[:_MASK_CHANNELS] = (np.asarray(masks, dtype=np.uint64)[:, None] >> shifts) & np.uint64(1)
    scalars = [board.reserves[color].get_count(kind) / 8.0
               for color in colors for kind in RESERVE_TYPES]
    scalars.extend((board.turn_moves / 3.0, board.turn_auto_moves / 64.0,
                    float(board.did_offer_draw), float(board.did_accept_draw)))
    result[_MASK_CHANNELS:] = np.asarray(scalars, dtype=np.float32)[:, None]
    return result.reshape(BOARD_CHANNELS, 8, 8)


def encode_actions(board, moves: Sequence) -> np.ndarray:
    """Return int64 [A, 6] in the supplied legal-action order.

    Columns: source, destination, capture victim, orientation, action kind,
    deployment piece type. Squares 0..63 use 64 for absent; orientations 0..7
    use 8 for absent; deployment is the engine's 1..7 piece type or 0. Kind
    distinguishes bombardment from free captures, so descriptors are injective
    for legal moves. No factorized policy probabilities or global action IDs
    are used. Blue's coordinates/directions are transformed only in this array.
    """
    rows = []
    for move in moves:
        kind = move.auto_capture_type if move.name == "AutoCapture" else move.name
        if kind not in ACTION_KINDS:
            raise ValueError(f"Unsupported action kind: {kind!r}")
        source, destination, victim = move.from_square, move.to_square, move.capture_preference
        if ((source is not None and not 0 <= source < 64) or
                (destination is not None and not 0 <= destination < 64) or
                (victim is not None and not 0 <= victim < 64)):
            raise ValueError("Invalid action square")
        orientation = move.orientation
        if orientation is not None and not 0 <= orientation < 8:
            raise ValueError(f"Invalid action orientation: {orientation}")
        deployment = NO_DEPLOYMENT if move.unit_type is None else move.unit_type
        if not 0 <= deployment <= 7:
            raise ValueError(f"Invalid deployment type: {deployment}")
        rows.append((NO_SQUARE if source is None else source,
                     NO_SQUARE if destination is None else destination,
                     NO_SQUARE if victim is None else victim,
                     NO_ORIENTATION if orientation is None else orientation,
                     ACTION_KINDS[kind], deployment))
    result = np.asarray(rows, dtype=np.int64).reshape(-1, len(ACTION_COLUMNS))
    if board.turn == e.BLUE and len(result):
        squares = result[:, :3]
        np.subtract(63, squares, out=squares, where=squares != NO_SQUARE)
        orientations = result[:, 3]
        np.bitwise_xor(orientations, 4, out=orientations, where=orientations != NO_ORIENTATION)
    return result
