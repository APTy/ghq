from .test_cases import LEGAL_MOVES_TESTS, MAKE_MOVE_TESTS
import pytest
from public.engine import (
    A1, AIRBORNE_INFANTRY, ARMORED_ARTILLERY, B3, B8, BB_ALL, BB_DIAG_ATTACKS, BB_DIAG_MASKS, BB_SQUARES, C3, D3, D4, D5, G1, ORIENT_E, ORIENT_N, ORIENT_NE, ORIENT_NW, ORIENT_S, ORIENT_SE, ORIENT_SW, ORIENT_W, BaseBoard, Outcome, Piece, HQ, INFANTRY, RED, BLUE, SquareSet, popcount, rotate_orientations_90, square, ARTILLERY, ARMORED_INFANTRY, HEAVY_ARTILLERY, STARTING_FEN, Move
)

@pytest.fixture
def board():
    return BaseBoard()

def test_empty_board(board):
    print(board)
    assert board.board_fen() == STARTING_FEN

def test_setting_pieces(board):
    board.clear_board()

    board.set_piece_at(square(0, 0), Piece(HQ, RED))
    board.set_piece_at(square(1, 0), Piece(INFANTRY, RED))
    board.set_piece_at(square(7, 7), Piece(HQ, BLUE))

def test_piece_operations(board):
    board.set_piece_at(square(3, 3), Piece(HQ, RED))
    piece = board.piece_at(square(3, 3))

    assert piece is not None
    assert piece.piece_type == HQ
    assert piece.color == RED

def test_load_fen():
    fen = "qr↓6/iii5/8/8/8/8/5III/6R↑Q IIIIIFFFPRRTH IIIIIFFFPRRTH r"
    board = BaseBoard(fen)

    # Test turn
    assert board.turn == RED

    # Test blue pieces
    assert board.piece_at(square(0, 7)).color == BLUE
    assert board.piece_at(square(0, 7)).piece_type == HQ

    assert board.piece_at(square(1, 7)).piece_type == ARTILLERY
    assert board.piece_at(square(1, 7)).color == BLUE
    assert board.piece_at(square(1, 7)).orientation == ORIENT_S

    assert board.piece_at(square(0, 6)).piece_type == INFANTRY
    assert board.piece_at(square(1, 7)).color == BLUE

    assert board.piece_at(square(1, 6)).piece_type == INFANTRY
    assert board.piece_at(square(1, 7)).color == BLUE

    assert board.piece_at(square(2, 6)).piece_type == INFANTRY
    assert board.piece_at(square(1, 7)).color == BLUE

    # Test red pieces
    assert board.piece_at(square(7, 0)).color == RED
    assert board.piece_at(square(7, 0)).piece_type == HQ

    assert board.piece_at(square(6, 0)).piece_type == ARTILLERY
    assert board.piece_at(square(6, 0)).color == RED
    assert board.piece_at(square(6, 0)).orientation == ORIENT_N

    assert board.piece_at(square(7, 1)).piece_type == INFANTRY
    assert board.piece_at(square(7, 1)).color == RED

    assert board.piece_at(square(6, 1)).piece_type == INFANTRY
    assert board.piece_at(square(6, 1)).color == RED

    assert board.piece_at(square(5, 1)).piece_type == INFANTRY
    assert board.piece_at(square(5, 1)).color == RED

    # Test reserves
    assert board.get_reserve_count(INFANTRY, RED) == 5
    assert board.get_reserve_count(ARMORED_INFANTRY, RED) == 3
    assert board.get_reserve_count(AIRBORNE_INFANTRY, RED) == 1
    assert board.get_reserve_count(ARTILLERY, RED) == 2
    assert board.get_reserve_count(ARMORED_ARTILLERY, RED) == 1
    assert board.get_reserve_count(HEAVY_ARTILLERY, RED) == 1

    assert board.get_reserve_count(INFANTRY, BLUE) == 5
    assert board.get_reserve_count(ARMORED_INFANTRY, BLUE) == 3

def test_invalid_fen():
    with pytest.raises(ValueError):
        BaseBoard("invalid fen")

    with pytest.raises(ValueError):
        BaseBoard("8/8/8/8/8/8/8/8")  # Missing reserves

def test_reserve_operations(board):
    board.clear_board()

    # Test adding to reserve
    board.add_to_reserve(INFANTRY, RED, 3)
    assert board.get_reserve_count(INFANTRY, RED) == 3

    # Test removing from reserve
    board.remove_from_reserve(INFANTRY, RED, 2)
    assert board.get_reserve_count(INFANTRY, RED) == 1

    # Test removing too many pieces
    with pytest.raises(ValueError):
        board.remove_from_reserve(INFANTRY, RED, 2)

def test_board_copy(board):
    board.clear_board()

    # Set up original board
    board.set_piece_at(square(0, 0), Piece(HQ, RED))
    board.add_to_reserve(INFANTRY, RED, 3)
    assert board.get_reserve_count(INFANTRY, RED) == 3
    print(board)

    # Make a copy
    copy = board.copy()

    # Modify copy
    copy.set_piece_at(square(0, 0), None)
    copy.add_to_reserve(INFANTRY, RED, 1)
    print(copy)

    # Check original is unchanged
    assert board.piece_at(square(0, 0)) is not None
    assert board.get_reserve_count(INFANTRY, RED) == 3

    # Check copy is modified
    assert copy.piece_at(square(0, 0)) is None
    assert copy.get_reserve_count(INFANTRY, RED) == 4

def test_moves():
    assert Move.move(square(0, 0), square(1, 1)).uci() == "a1b2"
    assert Move.move(square(0, 0), square(1, 1), square(2, 1)).uci() == "a1b2xc2"
    assert Move.reinforce(INFANTRY, square(0, 0)).uci() == "ria1"
    assert Move.reinforce(INFANTRY, square(0, 0), square(0, 1)).uci() == "ria1xa2"
    assert Move.move_and_orient(square(0, 0), square(1, 1), ORIENT_E).uci() == "a1b2→"
    assert Move.move_and_orient(square(0, 0), square(0, 0), ORIENT_E).uci() == "a1a1→"
    assert Move.skip().uci() == "skip"


def test_legal_moves():
    only_tests = [test for test in LEGAL_MOVES_TESTS if test.only]
    all_tests = LEGAL_MOVES_TESTS

    tests = only_tests if only_tests else all_tests

    for test in tests:
        fen = test.boardFEN
        board = BaseBoard(fen)
        print(board)

        if test.moves:
            for move_uci in test.moves:
                board.push(Move.from_uci(move_uci))

        if test.act_as_second_move:
            board.turn_moves = 1
            board.turn_auto_moves = 0

        actual_moves = [move.uci() for move in board.generate_legal_moves()]
        expected_moves = test.expectedMovesUCI.split()

        # Print differences between actual and expected moves
        actual_set = set(actual_moves)
        expected_set = set(expected_moves)

        missing_moves = expected_set - actual_set
        extra_moves = actual_set - expected_set

        if missing_moves:
            print(f"Missing moves: {' '.join(sorted(missing_moves))}")
        if extra_moves:
            print(f"Extra moves: {' '.join(sorted(extra_moves))}")

        assert len(actual_moves) == len(expected_moves), f"Incorrect moves for {test.description}"
        assert set(actual_moves) == set(expected_moves), f"Incorrect moves for {test.description}"


def test_make_move():
    for test in MAKE_MOVE_TESTS:
        fen = test.boardFEN
        board = BaseBoard(fen)

        print("Initial board:")
        print(board)

        actual_moves = [move.uci() for move in board.generate_legal_moves()]
        print(actual_moves)

        assert test.moveUCI in actual_moves, f"Move {test.moveUCI} not found in {test.description}"

        move = Move.from_uci(test.moveUCI)
        board.push(move)

        print("Board after move:")
        print(board)
        print(board.board_fen())

        assert board.board_fen() == test.expectedBoardFEN, f"Incorrect board after move {test.moveUCI} in {test.description}"


def test_bombarded_squares():
    board = BaseBoard("q7/8/8/4i3/8/4R↑3/8/7Q - - r")
    print(board)
    print(SquareSet(board.get_bombarded_squares(RED)))
    board._remove_bombarded_pieces()
    assert board.move_stack[-1].uci() == "sbe5"
    assert popcount(board.infantry) == 0
    print(board.move_stack)
    print(board)

def test_outcome():
    board = BaseBoard("q7/8/8/8/8/8/8/8 IIIIIFFFPRRTH iiiiifffprrth r")
    assert board.outcome() == Outcome("hq-capture", BLUE)
    board = BaseBoard("Q7/8/8/8/8/8/8/8 IIIIIFFFPRRTH iiiiifffprrth r")
    assert board.outcome() == Outcome("hq-capture", RED)

def test_engagement_disambiguation():
    # q  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  I  i  ·  ·  ·  ·
    # ·  ·  ·  I  i  I  i  I
    # ·  ·  ·  ·  I  ·  ·  ·
    # I  i  ·  ·  ·  ·  ·  ·
    # ·  I  i  I  r↓ ·  ·  ·
    # ·  ·  I  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  Q
    board = BaseBoard("q7/2I↑i↓4/3I↑i↓I↑i↓I↑/4I↑3/I↑i↓6/1I↑i↓I↑r↓3/2I↑5/7Q - - r -")
    print(board)

    # generate initial captures
    moves = list(board._generate_free_captures(RED))
    expected_moves = [Move.from_uci('sfb4'), Move.from_uci('sfe3'), Move.from_uci('sfc3'), Move.from_uci('sfd7'), Move.from_uci('sfg6'), Move.from_uci('sfe6')]
    assert set(moves) == set(expected_moves)
    assert popcount(board._all_infantry() & board.occupied_co[BLUE]) == 5, "expected 5 infantry"
    assert popcount(board._all_artillery() & board.occupied_co[BLUE]) == 1, "expected 1 artillery"

    # make a capture from the first cluster, should be one more capture left
    board.push(moves[0])
    moves = list(board._generate_free_captures(RED))
    expected_moves = [Move.from_uci('sfe3'), Move.from_uci('sfc3'), Move.from_uci('sfd7'), Move.from_uci('sfg6'), Move.from_uci('sfe6')]
    assert set(moves) == set(expected_moves)
    assert popcount(board._all_infantry() & board.occupied_co[BLUE]) == 4, "expected one infantry to be destroyed"
    assert popcount(board._all_artillery() & board.occupied_co[BLUE]) == 1, "expected same number of artillery"

    print(board)

    # make another capture from the first cluster, should be no more captures left for that cluster
    board.push(moves[0])
    moves = list(board._generate_free_captures(RED))
    expected_moves = [Move.from_uci('sfd7'), Move.from_uci('sfg6'), Move.from_uci('sfe6')]
    assert set(moves) == set(expected_moves)
    assert popcount(board._all_infantry() & board.occupied_co[BLUE]) == 4, "expected same number of infantry"
    assert popcount(board._all_artillery() & board.occupied_co[BLUE]) == 0, "expected one artillery to be destroyed"

    print(board)

    # make a capture from the second cluster, should be one more capture left
    board.push(moves[0])
    moves = list(board._generate_free_captures(RED))
    expected_moves = [Move.from_uci('sfg6'), Move.from_uci('sfe6')]
    assert set(moves) == set(expected_moves)
    assert popcount(board._all_infantry() & board.occupied_co[BLUE]) == 3, "expected one infantry to be destroyed"
    assert popcount(board._all_artillery() & board.occupied_co[BLUE]) == 0, "expected same number of artillery"

    print(board)

    # make another capture from the second cluster, should be no more captures left for that cluster
    board.push(moves[0])
    moves = list(board._generate_free_captures(RED))
    expected_moves = []
    assert set(moves) == set(expected_moves)
    assert popcount(board._all_infantry() & board.occupied_co[BLUE]) == 2, "expected one infantry to be destroyed"
    assert popcount(board._all_artillery() & board.occupied_co[BLUE]) == 0, "expected same number of artillery"

    print(board)

def test_engagement_disambiguation_2():
    # q  ·  i  I  ·  ·  ·  ·
    # ·  i  I  ·  ·  ·  ·  ·
    # i  I  ·  ·  ·  ·  ·  ·
    # I  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  i  I
    # i  I  ·  ·  ·  i  I  ·
    # I  ·  ·  ·  ·  I  ·  Q
    board = BaseBoard("q1i↓I↑4/1i↓I↑5/i↓I↑6/I↑7/8/6i↓I↑/i↓I↑3i↓I↑1/I↑4I↑1Q - - r -")
    print(board)

    # generate initial captures
    moves = list(board._generate_free_captures(RED))
    print(moves)
    expected_moves = [Move.from_uci('sfa2'), Move.from_uci('sfg3'), Move.from_uci('sff2'), Move.from_uci('sfc8'), Move.from_uci('sfb7'), Move.from_uci('sfa6')]
    assert set(moves) == set(expected_moves)
    assert popcount(board._all_infantry() & board.occupied_co[BLUE]) == 6, "expected 6 infantry"

    # make a capture from the first cluster, should be no captures left
    board.push(moves[0])
    moves = list(board._generate_free_captures(RED))
    expected_moves = [Move.from_uci('sfg3'), Move.from_uci('sff2'), Move.from_uci('sfc8'), Move.from_uci('sfb7'), Move.from_uci('sfa6')]
    assert set(moves) == set(expected_moves)
    assert popcount(board._all_infantry() & board.occupied_co[BLUE]) == 5, "expected one infantry to be destroyed"

    # make a capture from second cluster, should be no captures left
    board.push(moves[0])
    moves = list(board._generate_free_captures(RED))
    expected_moves = [Move.from_uci('sfc8'), Move.from_uci('sfb7'), Move.from_uci('sfa6')]
    assert set(moves) == set(expected_moves)
    assert popcount(board._all_infantry() & board.occupied_co[BLUE]) == 4, "expected one infantry to be destroyed"

    # make a capture from third cluster, should be no captures left
    board.push(moves[0])
    moves = list(board._generate_free_captures(RED))
    expected_moves = []
    assert set(moves) == set(expected_moves)
    assert popcount(board._all_infantry() & board.occupied_co[BLUE]) == 3, "expected one infantry to be destroyed"

def test_engagement_disambiguation_3():
    # q  ·  i  I  ·  ·  ·  ·
    # ·  i  I  ·  ·  ·  ·  ·
    # i  I  ·  ·  ·  ·  ·  ·
    # I  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  i  I
    # i  I  I  I  I  i  I  ·
    # I  ·  ·  ·  ·  I  ·  Q
    board = BaseBoard("q1i↓I↑4/1i↓I↑5/i↓I↑6/I↑7/8/6i↓I↑/i↓I↑IIIi↓I↑1/I↑4I↑1Q - - r -")
    print(board)

    # generate initial captures
    moves = list(board._generate_free_captures(RED))
    expected_moves = [Move.from_uci('sfa2'), Move.from_uci('sfg3'), Move.from_uci('sff2'), Move.from_uci('sfc8'), Move.from_uci('sfb7'), Move.from_uci('sfa6')]
    assert set(moves) == set(expected_moves)
    assert popcount(board._all_infantry() & board.occupied_co[BLUE]) == 6, "expected 6 infantry"

    # make a capture from the first cluster, should be no captures left
    board.push(moves[0])
    moves = list(board._generate_free_captures(RED))
    expected_moves = [Move.from_uci('sfg3'), Move.from_uci('sff2'), Move.from_uci('sfc8'), Move.from_uci('sfb7'), Move.from_uci('sfa6')]
    assert set(moves) == set(expected_moves)
    assert popcount(board._all_infantry() & board.occupied_co[BLUE]) == 5, "expected one infantry to be destroyed"

    # make a capture from the second cluster, should be one capture left
    board.push(moves[0])
    moves = list(board._generate_free_captures(RED))
    expected_moves = [Move.from_uci('sff2'), Move.from_uci('sfc8'), Move.from_uci('sfb7'), Move.from_uci('sfa6')]
    assert set(moves) == set(expected_moves)
    assert popcount(board._all_infantry() & board.occupied_co[BLUE]) == 4, "expected one infantry to be destroyed"

    # make another capture from the second cluster, should be no captures left
    board.push(moves[0])
    moves = list(board._generate_free_captures(RED))
    expected_moves = [Move.from_uci('sfc8'), Move.from_uci('sfb7'), Move.from_uci('sfa6')]
    assert set(moves) == set(expected_moves)
    assert popcount(board._all_infantry() & board.occupied_co[BLUE]) == 3, "expected one infantry to be destroyed"

    # make a capture from third cluster, should be no captures left
    board.push(moves[0])
    moves = list(board._generate_free_captures(RED))
    expected_moves = []
    assert set(moves) == set(expected_moves)
    assert popcount(board._all_infantry() & board.occupied_co[BLUE]) == 2, "expected one infantry to be destroyed"

def test_engagement_disambiguation_4():
    # r↑ I  i  ·  ·  ·  ·  ·
    # I  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  Q
    board = BaseBoard("r↑I↑i5/I↑7/8/8/8/8/8/7Q - - r -")
    print(board)

    # artillery should be capturable
    moves = list(board._generate_free_captures(RED))
    expected_moves = [Move.from_uci('sfa8')]
    assert set(moves) == set(expected_moves)
    assert popcount(board._all_infantry() & board.occupied_co[BLUE]) == 1, "expected 1 infantry"
    assert popcount(board._all_artillery() & board.occupied_co[BLUE]) == 1, "expected 1 artillery"

    # make a capture from cluster, should be no captures left
    board.push(moves[0])
    moves = list(board._generate_free_captures(RED))
    expected_moves = []
    assert set(moves) == set(expected_moves)
    assert popcount(board._all_infantry() & board.occupied_co[BLUE]) == 1, "expected 1 infantry"
    assert popcount(board._all_artillery() & board.occupied_co[BLUE]) == 0, "expected one artillery to be destroyed"


def test_engagement_disambiguation_5():
    # q  ·  r↑ ·  ·  ·  ·  ·
    # ·  ·  I  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  Q
    board = BaseBoard("q1r↑5/2I↑5/8/8/8/8/8/7Q - - r -")
    print(board)

    moves = list(board._generate_free_captures(RED))
    print(moves)

    # artillery should be capturable
    moves = list(board._generate_free_captures(RED))
    expected_moves = [Move.from_uci('sfc88')]
    assert set(moves) == set(expected_moves)
    assert popcount(board._all_infantry() & board.occupied_co[BLUE]) == 0, "expected 0 infantry"
    assert popcount(board._all_artillery() & board.occupied_co[BLUE]) == 1, "expected 1 artillery"

    # make a capture from cluster, should be no captures left
    board.push(moves[0])
    moves = list(board._generate_free_captures(RED))
    expected_moves = []
    assert set(moves) == set(expected_moves)
    assert popcount(board._all_infantry() & board.occupied_co[BLUE]) == 0, "expected 0 infantry"
    assert popcount(board._all_artillery() & board.occupied_co[BLUE]) == 0, "expected one artillery to be destroyed"

def test_engagement_disambiguation_6():
    # q  I  i  ·  ·  ·  ·  ·
    # I  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  Q
    board = BaseBoard("qI↑i5/I↑7/8/8/8/8/8/7Q - - r -")
    print(board)

    # hq shouldn't be capturable
    moves = list(board._generate_free_captures(RED))
    print(moves)
    expected_moves = []
    assert set(moves) == set(expected_moves)

def test_engagement_disambiguation_7():
    # q  I  i  I  ·  ·  ·  ·
    # I  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  Q
    board = BaseBoard("qI↑iI4/I↑7/8/8/8/8/8/7Q - - r -")
    print(board)

    # hq should be capturable
    moves = list(board._generate_free_captures(RED))
    expected_moves = [Move.from_uci('sfc8'), Move.from_uci('sfa8')]
    assert set(moves) == set(expected_moves)

def test_engagement_disambiguation_8():
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  I  ·  ·  ·  ·
    # ·  ·  I  q  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  Q
    board = BaseBoard("8/8/3I↑4/2I↑q↓4/8/8/8/7Q - - r -")
    print(board)

    moves = list(board._generate_free_captures(RED))
    expected_moves = [Move.from_uci('sfd5')]
    assert set(moves) == set(expected_moves)

def test_engagement_disambiguation_9():
    # ·  I  i  I  i  ·  ·  ·
    # ·  i  I  i  ·  ·  ·  ·
    # ·  ·  ·  I  ·  ·  ·  ·
    # ·  ·  I  q  I  i  ·  ·
    # ·  ·  ·  I  ·  I  ·  ·
    # ·  ·  ·  i  I  i  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  Q
    board = BaseBoard("1I↑i↓I↑i↓3/1i↓I↑i↓4/3I↑4/2I↑q↓I↑i↓2/3I↑1I↑2/3i↓I↑i↓2/8/7Q - - r -")
    print(board)

    # hq should be capturable
    moves = list(board._generate_free_captures(RED))
    expected_moves = [Move.from_uci('sff5'), Move.from_uci('sfd5'), Move.from_uci('sff3'), Move.from_uci('sfd3')]
    assert set(moves) == set(expected_moves)

def test_engagement_disambiguation_10():
    # ·  I  i  I  i  ·  ·  ·
    # ·  i  I  i  ·  ·  ·  ·
    # ·  ·  ·  I  ·  ·  ·  ·
    # ·  i  I  q  I  i  ·  ·
    # ·  ·  ·  I  ·  I  ·  ·
    # ·  ·  ·  i  I  i  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  ·
    # ·  ·  ·  ·  ·  ·  ·  Q
    board = BaseBoard("1I↑i↓I↑i↓3/1i↓I↑i↓4/3I↑4/1iI↑q↓I↑i↓2/3I↑1I↑2/3i↓I↑i↓2/8/7Q - - r -")
    print(board)

    # hq shouldn't be capturable
    moves = list(board._generate_free_captures(RED))
    expected_moves = [Move.from_uci('sff5'), Move.from_uci('sff3'), Move.from_uci('sfd3')]
    print(moves)
    assert set(moves) == set(expected_moves)


def test_canonical_form():
    board = BaseBoard()
    mirrored = board.mirror()
    mirrored.turn = not mirrored.turn # turn should be opposite
    assert mirrored.board_fen() == board.board_fen()


def test_canonical_form_2():
    board = BaseBoard("q5r←1/6r↙r↓/8/8/8/8/6R↖R↑/6R←Q IIIIIFFFPRRTH iiiiifffprrth r")
    mirrored = board.mirror()
    print(board)
    print(mirrored)
    mirrored.turn = not mirrored.turn # turn should be opposite
    assert mirrored.board_fen() == "qr→6/r↓r↘6/8/8/8/8/R↑R↗6/1R→5Q IIIIIFFFPRRTH iiiiifffprrth r"

    mirrored_2 = mirrored.mirror()
    mirrored_2.turn = not mirrored_2.turn # turn should be opposite
    assert mirrored_2.board_fen() == board.board_fen()

def test_90_degree_rotate_1():
    board = BaseBoard()
    assert board.get_orientation(B8) == ORIENT_S
    assert board.get_orientation(G1) == ORIENT_N
    board.apply_orientation_transform(rotate_orientations_90)
    assert board.get_orientation(B8) == ORIENT_W
    assert board.get_orientation(G1) == ORIENT_E
    board.apply_orientation_transform(rotate_orientations_90)
    assert board.get_orientation(B8) == ORIENT_N
    assert board.get_orientation(G1) == ORIENT_S
    board.apply_orientation_transform(rotate_orientations_90)
    assert board.get_orientation(B8) == ORIENT_E
    assert board.get_orientation(G1) == ORIENT_W
    board.apply_orientation_transform(rotate_orientations_90)
    assert board.get_orientation(B8) == ORIENT_S
    assert board.get_orientation(G1) == ORIENT_N

def test_90_degree_rotate_2():
    board = BaseBoard("q7/8/8/3R↗4/8/8/8/7Q - - r -")
    assert board.get_orientation(D5) == ORIENT_NE
    board.apply_orientation_transform(rotate_orientations_90)
    assert board.get_orientation(D5) == ORIENT_SE
    board.apply_orientation_transform(rotate_orientations_90)
    assert board.get_orientation(D5) == ORIENT_SW
    board.apply_orientation_transform(rotate_orientations_90)
    assert board.get_orientation(D5) == ORIENT_NW
    board.apply_orientation_transform(rotate_orientations_90)
    assert board.get_orientation(D5) == ORIENT_NE

def test_90_degree_rotate_3():
    board = BaseBoard()
    print(board)
    rotated = board.rotate_90_clockwise()
    print(rotated)
    rotated = rotated.rotate_90_clockwise()
    print(rotated)
    rotated = rotated.rotate_90_clockwise()
    print(rotated)
    rotated = rotated.rotate_90_clockwise()
    print(rotated)
    assert rotated.board_fen() == board.board_fen()

def test_draw():
    board = BaseBoard()
    board.push(Move.skip())
    assert board.did_offer_draw, "should offer draw after first skip"
    assert not board.did_accept_draw, "should not accept draw after first skip"
    assert not board.is_game_over(), "should not be game over after first skip"
    board.push(Move.skip())
    assert board.did_offer_draw, "should offer draw after second skip"
    assert board.did_accept_draw, "should accept draw after second skip"
    assert board.outcome() == Outcome("double-skip", None)

def test_draw_2():
    board = BaseBoard()

    board.push(Move.skip())
    assert board.turn == BLUE, "should be blue's turn after first skip"
    assert not board.is_game_over(), "should not be game over after first skip"

    board.push(Move.from_uci('rih8'))
    assert board.turn == BLUE, "should still be blue's turn"
    assert not board.is_game_over(), "should not be game over after"

    board.push(Move.skip())
    assert board.turn == RED, "should be red's turn after second skip"
    assert not board.did_offer_draw, "should not offer draw after second skip"
    assert not board.is_game_over(), "should not be game over after second skip"

def test_draw_by_hq_stalemate():
    board = BaseBoard("qI6/1I6/8/R↑7/8/8/8/7Q - - r")
    assert board.outcome() == None
    board.push(Move.skip())
    assert board.outcome() == Outcome("stalemate", None)

def test_win_by_hq_bombardment():
    board = BaseBoard("qI6/1I6/8/R↑7/8/8/8/7Q - - r")
    board.push(Move.from_uci('a5a6↑'))
    board.push(Move.skip())
    assert board.outcome() == Outcome("hq-capture", RED)
