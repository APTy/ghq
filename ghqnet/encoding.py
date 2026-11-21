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


def _plane(mask: int) -> np.ndarray:
    return ((np.uint64(mask) >> _SQUARE_INDICES) & np.uint64(1)).astype(np.float32).reshape(8, 8)


def encode_board(board) -> np.ndarray:
    """Return float32 [41, 8, 8], with the current player on the Red home rank.

    All rule-relevant serialized fields are represented, including pending free
    captures. Occupancy/color masks are encoded jointly by the 14 piece planes;
    absolute player identity is intentionally removed by canonicalization.
    """
    canonical = board.mirror() if board.turn == e.BLUE else board
    piece_masks = (canonical.hq, canonical.infantry, canonical.armored_infantry,
                   canonical.airborne_infantry, canonical.artillery,
                   canonical.armored_artillery, canonical.heavy_artillery)
    masks = [mask & canonical.occupied_co[color]
             for color in (e.RED, e.BLUE) for mask in piece_masks]
    masks.extend((*canonical.bombarded_co, *canonical.adjacent_infantry_squares_co,
                  canonical.orientation_bit0, canonical.orientation_bit1,
                  canonical.orientation_bit2, canonical.turn_pieces,
                  canonical.free_capture_clusters, canonical.free_capture_enemies,
                  canonical.free_capture_num_allowed))
    planes = [_plane(mask) for mask in masks]
    scalars = [canonical.reserves[color].get_count(kind) / 8.0
               for color in (e.RED, e.BLUE) for kind in RESERVE_TYPES]
    scalars.extend((canonical.turn_moves / 3.0, canonical.turn_auto_moves / 64.0,
                    float(canonical.did_offer_draw), float(canonical.did_accept_draw)))
    planes.extend(np.full((8, 8), value, dtype=np.float32) for value in scalars)
    return np.stack(planes)


def encode_actions(board, moves: Sequence) -> np.ndarray:
    """Return int64 [A, 6] in the supplied legal-action order.

    Columns: source, destination, capture victim, orientation, action kind,
    deployment piece type. Squares 0..63 use 64 for absent; orientations 0..7
    use 8 for absent; deployment is the engine's 1..7 piece type or 0. Kind
    distinguishes bombardment from free captures, so descriptors are injective
    for legal moves. No factorized policy probabilities or global action IDs
    are used. Blue's coordinates/directions are transformed only in this array.
    """
    result = np.empty((len(moves), len(ACTION_COLUMNS)), dtype=np.int64)
    blue = board.turn == e.BLUE
    for index, move in enumerate(moves):
        kind = move.auto_capture_type if move.name == "AutoCapture" else move.name
        if kind not in ACTION_KINDS:
            raise ValueError(f"Unsupported action kind: {kind!r}")
        squares = []
        for square in (move.from_square, move.to_square, move.capture_preference):
            if square is not None and not 0 <= square < 64:
                raise ValueError(f"Invalid action square: {square}")
            squares.append(NO_SQUARE if square is None else 63 - square if blue else square)
        orientation = move.orientation
        if orientation is not None and not 0 <= orientation < 8:
            raise ValueError(f"Invalid action orientation: {orientation}")
        orientation = NO_ORIENTATION if orientation is None else (orientation + (4 if blue else 0)) % 8
        deployment = NO_DEPLOYMENT if move.unit_type is None else move.unit_type
        if not 0 <= deployment <= 7:
            raise ValueError(f"Invalid deployment type: {deployment}")
        result[index] = (*squares, orientation, ACTION_KINDS[kind], deployment)
    return result
