from typing import Optional
import numpy as np
from public.engine import BLUE, BaseBoard, Move, Square, square_rank

SKIP_MOVE_SIZE = 1
AUTO_CAPTURE_SIZE = 128
REINFORCE_SIZE = 480
MOVE_AND_ORIENT_SIZE = 8704
STANDARD_MOVE_SIZE = 3840
AIRBORNE_MOVE_SIZE = 5120

# Pre-allocate the action size 1 + 128 + 480 + 8704 + 3840 + 5120 = 18273
ACTION_SIZE = SKIP_MOVE_SIZE + AUTO_CAPTURE_SIZE + REINFORCE_SIZE + MOVE_AND_ORIENT_SIZE + STANDARD_MOVE_SIZE + AIRBORNE_MOVE_SIZE

def encode_move(move: Move) -> int:
    offset = 0

    if move.name == "Skip":
        return offset

    offset += SKIP_MOVE_SIZE

    # Free captures are the first 64 actions
    # Bombard captures are the next 64 actions
    # So 128 total actions (64 + 64)
    if move.name == "AutoCapture":
        if move.capture_preference is None:
            raise ValueError("AutoCapture move must specify capture_preference")
        if move.auto_capture_type is None:
            raise ValueError("AutoCapture move must specify auto_capture_type")

        # first block of 64 actions are free captures
        if move.auto_capture_type == "free":
            return offset + move.capture_preference

        offset += 64

        # second block of 64 actions are bombard captures
        if move.auto_capture_type == "bombard":
            return offset + move.capture_preference

        raise ValueError(f"Unknown auto_capture_type: {move.auto_capture_type}")

    offset += AUTO_CAPTURE_SIZE

    # Each reinforce is 80 possible actions (2 * 8 * 5)
    # Since unit type can be reinforced on one of 8 squares and 5 capture options (4 adjacent + 1 none)
    # There are 6 unit types
    # And two home ranks where reinforcements can happen (red and blue)
    # So 480 total actions (6 * 80)
    if move.name == "Reinforce":
        # the to_square is 0-64, and we only care about the file (0-7), so its the same for red or blue
        to_square = move.to_square % 8

        # blue gets the second set of 8 squares
        if square_rank(move.to_square) == 7:
            to_square += 8

        capture_index = _get_adjacent_capture_index(move.to_square, move.capture_preference)
        square_offset = (to_square * 5) + capture_index
        unit_offset = _unit_type_to_index(move.unit_type) * 80

        return offset + unit_offset + square_offset

    offset += REINFORCE_SIZE

    # Each move and orient is 136 possible actions (17 * 8)
    # Since each unit type has 17 possible moves (2 for any direction or remain in place) and 8 possible orientations
    # This can happen from any of the 64 squares on the board
    # So 8,704 total actions (64 * 136)
    if move.name == "MoveAndOrient":
        if move.from_square is None or move.to_square is None or move.orientation is None:
            raise ValueError("MoveAndOrient must specify from_square, to_square, and orientation")

        move_index = _get_artillery_move_index(move.from_square, move.to_square) # 0-16
        orientation_index = move.orientation # 0-7

        from_square_offset = move.from_square * 136
        to_square_offset = move_index * 8
        orientation_offset = orientation_index

        return offset + from_square_offset + to_square_offset + orientation_offset


    offset += MOVE_AND_ORIENT_SIZE

    # Each move is 80 possible actions (16 * 5)
    # Since each unit type has 16 possible moves (2 for any direction) and 5 capture options (4 adjacent + 1 none)
    # This can happen from any of the 48 squares on the board (any 64 squares minus the 16 squares on the home ranks)
    # So 3,840 total actions (48 * 80)
    if move.name == "Move" and square_rank(move.from_square) > 0 and square_rank(move.from_square) < 7:
        if move.from_square is None or move.to_square is None:
            raise ValueError("Move must specify from_square and to_square")

        move_index = _get_infantry_move_index(move.from_square, move.to_square) # 0-15
        from_square_offset = (move.from_square - 8) * 80 # subtract 8 to account for red's home rank
        to_square_offset = move_index * 5
        capture_index = _get_adjacent_capture_index(move.to_square, move.capture_preference)

        return offset + from_square_offset + to_square_offset + capture_index

    offset += STANDARD_MOVE_SIZE

    # Each airborne move is 320 possible actions (64 * 5)
    # Since each unit type has 64 possible moves (any square except the original square) and 5 capture options (4 adjacent + 1 none)
    # This can happen from the 16 squares on the home ranks
    # So 5,120 total actions (16 * 320)
    if move.name == "Move" and (square_rank(move.from_square) == 0 or square_rank(move.from_square) == 7):
        if move.from_square is None or move.to_square is None:
            raise ValueError("AirborneMove must specify from_square and to_square")

        rank_offset = 0 if square_rank(move.from_square) == 0 else 8 # first 8 squares are red's home rank, last 8 are blue's home rank
        from_square_offset = ((move.from_square % 8) + rank_offset) * 320
        to_square_offset = move.to_square * 5
        capture_index = _get_adjacent_capture_index(move.to_square, move.capture_preference)

        return offset + from_square_offset + to_square_offset + capture_index

    offset += AIRBORNE_MOVE_SIZE

    raise ValueError(f"Unknown move type: {move.name}")

def _unit_type_to_index(unit_type: int) -> int:
    return unit_type - 2

def _get_adjacent_capture_index(from_square: Square, to_square: Optional[Square]) -> int:
    if to_square is None:
        return 0

    diff = to_square - from_square
    if diff == -8:
        return 1  # North
    elif diff == 1:
        return 2  # East
    elif diff == 8:
        return 3  # South
    elif diff == -1:
        return 4  # West

    raise ValueError(f"Invalid adjacent capture from {from_square} to {to_square}: diff {diff}")

def _get_artillery_move_index(from_square: Square, to_square: Square) -> int:
    diff = to_square - from_square
    if diff == 0:
        return 0  # Stay in place
    elif diff == -8:
        return 1  # North 1
    elif diff == -7:
        return 2  # Northeast 1
    elif diff == 1:
        return 3  # East 1
    elif diff == 9:
        return 4  # Southeast 1
    elif diff == 8:
        return 5  # South 1
    elif diff == 7:
        return 6  # Southwest 1
    elif diff == -1:
        return 7  # West 1
    elif diff == -9:
        return 8  # Northwest 1
    elif diff == -16:
        return 9  # North 2
    elif diff == -14:
        return 10  # Northeast 2
    elif diff == 2:
        return 11  # East 2
    elif diff == 18:
        return 12  # Southeast 2
    elif diff == 16:
        return 13  # South 2
    elif diff == 14:
        return 14  # Southwest 2
    elif diff == -2:
        return 15  # West 2
    elif diff == -18:
        return 16  # Northwest 2
    raise ValueError(f"Invalid move index: {diff}")

def _get_infantry_move_index(from_square: Square, to_square: Square) -> int:
    diff = to_square - from_square
    if diff == 0:
        raise ValueError(f"Invalid move index: {diff}")

    # Infantry moves are the same as artillery moves, but without the stay in place option
    return _get_artillery_move_index(from_square, to_square) - 1


def decode_move(action: int) -> Move:
    offset = 0

    if action < offset + SKIP_MOVE_SIZE:
        return Move.skip()

    offset += SKIP_MOVE_SIZE

    if action < offset + AUTO_CAPTURE_SIZE:
        free_capture_squares = offset + 64
        if action < free_capture_squares:
            return Move.auto_capture_free(action - offset)
        return Move.auto_capture_bombard(action - free_capture_squares)

    offset += AUTO_CAPTURE_SIZE

    if action < offset + REINFORCE_SIZE:
        action = action - offset
        unit_type = _index_to_unit_type(action // 80)
        remaining = action % 80
        to_square = remaining // 5
        capture_index = remaining % 5

        # Convert encoded square (0-7 for red, 8-15 for blue) to actual square
        to_square = to_square + (48 if to_square >= 8 else 0)

        if capture_index == 0:
            return Move.reinforce(unit_type, to_square)

        capture_square = _get_adjacent_capture_square(to_square, capture_index)
        return Move.reinforce(unit_type, to_square, capture_square)

    offset += REINFORCE_SIZE

    if action < offset + MOVE_AND_ORIENT_SIZE:
        action = action - offset
        from_square = action // 136
        remaining = action % 136
        move_index = remaining // 8
        orientation_index = remaining % 8

        to_square = from_square + _get_artillery_move_square(move_index)

        return Move.move_and_orient(from_square, to_square, orientation_index)

    offset += MOVE_AND_ORIENT_SIZE

    if action < offset + STANDARD_MOVE_SIZE:
        action = action - offset
        from_square = (action // 80) + 8 # add 8 to account for red's home rank
        remaining = action % 80
        move_index = remaining // 5
        capture_index = remaining % 5

        to_square = from_square + _get_infantry_move_square(move_index)
        capture_square = _get_adjacent_capture_square(to_square, capture_index)

        return Move.move(from_square, to_square, capture_square)

    offset += STANDARD_MOVE_SIZE

    if action < offset + AIRBORNE_MOVE_SIZE:
        action = action - offset
        from_square_index = action // 320
        remaining = action % 320
        to_square = remaining // 5
        capture_index = remaining % 5
        capture_square = _get_adjacent_capture_square(to_square, capture_index)

        from_square = from_square_index if square_rank(from_square_index) == 0 else from_square_index + (56 - 8)

        return Move.move(from_square, to_square, capture_square)

    raise ValueError(f"Unknown action: {action}")

def _index_to_unit_type(index: int) -> int:
    return index + 2

def _get_adjacent_capture_square(square: Square, direction: int) -> Optional[Square]:
    if direction == 0:
        return None

    if direction == 1:
        return square - 8
    elif direction == 2:
        return square + 1
    elif direction == 3:
        return square + 8
    elif direction == 4:
        return square - 1

    raise ValueError(f"Invalid direction: {direction}")

def _get_artillery_move_square(move_index: int) -> int:
    if move_index == 0:
        return 0  # Stay in place
    elif move_index == 1:
        return -8  # North 1
    elif move_index == 2:
        return -7  # Northeast 1
    elif move_index == 3:
        return 1  # East 1
    elif move_index == 4:
        return 9  # Southeast 1
    elif move_index == 5:
        return 8  # South 1
    elif move_index == 6:
        return 7  # Southwest 1
    elif move_index == 7:
        return -1  # West 1
    elif move_index == 8:
        return -9  # Northwest 1
    elif move_index == 9:
        return -16  # North 2
    elif move_index == 10:
        return -14  # Northeast 2
    elif move_index == 11:
        return 2  # East 2
    elif move_index == 12:
        return 18  # Southeast 2
    elif move_index == 13:
        return 16  # South 2
    elif move_index == 14:
        return 14  # Southwest 2
    elif move_index == 15:
        return -2  # West 2
    elif move_index == 16:
        return -18  # Northwest 2
    raise ValueError(f"Invalid move index: {move_index}")

def _get_infantry_move_square(move_index: int) -> int:
    return _get_artillery_move_square(move_index + 1)

def get_action_size() -> int:
    """
    Returns the total number of possible actions in the action space.
    """
    return ACTION_SIZE

def get_valid_moves(board: BaseBoard) -> np.ndarray:
    """
    Returns a binary vector indicating which actions are valid from the current board state.
    Uses numpy for better performance.
    """
    valid_moves = np.zeros(ACTION_SIZE, dtype=np.int8)

    for move in board.generate_legal_moves():
        valid_moves[encode_move(move)] = 1

    return valid_moves
