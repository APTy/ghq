from __future__ import print_function
from typing import List, Tuple
from .action_space import decode_move, get_action_size, get_valid_moves
import numpy as np
import sys
from .Game import Game
from public.engine import COLOR_NAMES, BaseBoard, Move, Square, piece_symbol, piece_name, PIECE_TYPES, PIECE_SYMBOLS, RED, BLUE

sys.path.append('..')

BOARD_SIZE = 8

class GHQGame(Game):
    """
    This class specifies the base Game class. To define your own game, subclass
    this class and implement the functions below. This works when the game is
    two-player, adversarial and turn-based.

    Use 1 for player1 and -1 for player2.
    """
    def __init__(self):
        self._logs = []
        self._move_cache = {}

    def __del__(self):
        with open('game_log.txt', 'a') as f:
            for board, action, move in self._logs[-100:]:
                f.write(f"{board.board_fen()}:{action}:{move}:{board.outcome()}\n")

    def _log_action(self, board, action, move):
        self._logs.append((board.copy(), action, move))

    def getInitBoard(self):
        """
        Returns:
            startBoard: a representation of the board (ideally this is the form
                        that will be the input to your neural network)
        """
        return BaseBoard()

    def getBoardSize(self):
        """
        Returns:
            (x,y): a tuple of board dimensions
        """
        return (BOARD_SIZE, BOARD_SIZE)

    def getActionSize(self):
        """
        Returns:
            actionSize: number of all possible actions
        """
        return get_action_size()

    def getNextState(self, board: BaseBoard, player: int, action: int) -> Tuple[BaseBoard, int]:
        """
        Input:
            board: current board
            player: current player (1 or -1)
            action: action taken by current player

        Returns:
            nextBoard: board after applying action
            nextPlayer: player who plays in the next turn (should be -player)
        """
        current_player = 1 if board.turn == RED else -1
        if current_player != player:
            raise ValueError(f"Current player {current_player} does not match passed player {player}")

        b = board.copy()

        move = decode_move(action)
        self._log_action(b, action, move)
        b.push(move)

        color_to_player = 1 if b.turn == RED else -1
        return b, color_to_player

    def getValidMoves(self, board: BaseBoard, player: int) -> np.ndarray:
        """
        Input:
            board: current board
            player: current player

        Returns:
            validMoves: a binary vector of length self.getActionSize(), 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        key = board._transposition_key()

        if key in self._move_cache:
            return self._move_cache[key]

        color_to_player = 1 if board.turn == RED else -1
        if color_to_player != player:
            raise ValueError(f"Current player {color_to_player} does not match expected player {player}")

        moves = get_valid_moves(board)
        self._move_cache[key] = moves
        return moves

    def getGameEnded(self, board: BaseBoard, player: int) -> float:
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            r: 0 if game has not ended. 1 if player won, -1 if player lost,
               small non-zero value for draw.

        """
        if board.is_game_over():
            player_color = RED if player == 1 else BLUE
            outcome = board.outcome()
            if outcome is None:
                raise ValueError("game is not over")
            if outcome.winner is None:
                return 1e-3
            return 1 if outcome.winner == player_color else -1
        return 0

    def getCanonicalForm(self, board: BaseBoard, player: int) -> BaseBoard:
        """
        Input:
            board: current board
            player: current player (1 or -1)

        Returns:
            canonicalBoard: returns canonical form of board. The canonical form
                            should be independent of player. For e.g. in chess,
                            the canonical form can be chosen to be from the pov
                            of white. When the player is white, we can return
                            board as is. When the player is black, we can invert
                            the colors and return the board.
        """
        b = board.copy()

        if player == 1:
            return b

        b.reserves[RED], b.reserves[BLUE] = b.reserves[BLUE], b.reserves[RED]
        b.occupied_co[RED], b.occupied_co[BLUE] = b.occupied_co[BLUE], b.occupied_co[RED]
        b.bombarded_co[RED], b.bombarded_co[BLUE] = b.bombarded_co[BLUE], b.bombarded_co[RED]
        b.adjacent_infantry_squares_co[RED], b.adjacent_infantry_squares_co[BLUE] = b.adjacent_infantry_squares_co[BLUE], b.adjacent_infantry_squares_co[RED]
        b.turn = RED

        return b

    def getSymmetries(self, board: BaseBoard, pi: np.ndarray) -> List[Tuple[BaseBoard, np.ndarray]]:
        """
        Input:
            board: current board
            pi: policy vector of size self.getActionSize()

        Returns:
            symmForms: a list of [(board,pi)] where each tuple is a symmetrical
                       form of the board and the corresponding pi vector. This
                       is used when training the neural network from examples.
        """
        return [(board, pi)]

        # TODO(tyler): implement this to speed up trainin
        # syms = []

        # # Handle skip move separately since it's not board-position dependent
        # skip_pi = pi[0]

        # # Reshape policy vector into board dimensions for each move type
        # auto_capture_pi = pi[1:129].reshape(2, 8, 8)  # 2 capture types, 8x8 board
        # reinforce_pi = pi[129:369].reshape(6, 8, 5)   # 6 unit types, 8 locations, 5 capture options
        # move_orient_pi = pi[369:9073].reshape(8, 8, 136)  # 8x8 board, 136 move+orient combinations (17 moves * 8 orientations)
        # standard_move_pi = pi[9073:12913].reshape(6, 8, 80)  # 6 ranks (1-6), 8 files, 80 move combinations (16 moves * 5 capture options)
        # airborne_move_pi = pi[12913:].reshape(2, 8, 8, 8, 5)  # 2 players, 8 home rank locations, 8x8 landing squares, 5 capture options

        # # Generate all rotations and flips
        # for i in range(1, 5):  # 90, 180, 270 degree rotations
        #     for j in [True, False]:  # flip
        #         # Rotate board using proper rotation method
        #         newB = b.rotate_90_clockwise()
        #         if i > 1:
        #             for _ in range(i-1):
        #                 newB = newB.rotate_90_clockwise()

        #         # Rotate policy components
        #         new_auto_capture = np.rot90(auto_capture_pi, -i, axes=(1,2))  # Rotate only board dimensions clockwise
        #         new_reinforce = np.rot90(reinforce_pi, -i, axes=(1,2))  # Rotate only location dimension clockwise
        #         new_move_orient = np.rot90(move_orient_pi, -i, axes=(0,1))  # Rotate board positions, moves and orientations are relative
        #         new_standard = np.rot90(standard_move_pi, -i, axes=(0,1))  # Rotate only rank and file dimensions
        #         new_airborne = np.rot90(airborne_move_pi, -i, axes=(2,3))  # Rotate only landing square dimensions

        #         if j:
        #             # Flip board using mirror method
        #             newB = newB.mirror()

        #             # Flip policy components
        #             new_auto_capture = np.fliplr(new_auto_capture)  # Flip only board dimensions
        #             new_reinforce = np.fliplr(new_reinforce)  # Flip only location dimension
        #             new_move_orient = np.fliplr(new_move_orient)  # Flip board positions, moves and orientations are relative
        #             new_standard = np.fliplr(new_standard)  # Flip only rank and file dimensions
        #             new_airborne = np.fliplr(new_airborne, axes=(2,3))  # Flip only landing square dimensions

        #         # Combine policy components back into 1D vector
        #         new_pi = np.concatenate([
        #             [skip_pi],  # Skip move unchanged
        #             new_auto_capture.ravel(),
        #             new_reinforce.ravel(),
        #             new_move_orient.ravel(),  # This will flatten (8,8,8,8,8) to 1D
        #             new_standard.ravel(),
        #             new_airborne.ravel()
        #         ])

        #         syms.append((serialize_board(newB), list(new_pi)))

        # return syms

    def stringRepresentation(self, board):
        """
        Input:
            board: current board

        Returns:
            boardString: a quick conversion of board to a string format.
                         Required by MCTS for hashing.
        """
        return board._transposition_key()
