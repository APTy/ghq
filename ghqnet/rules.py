"""One authoritative rules engine shared with the production CPU bot."""
from pathlib import Path
import sys

PUBLIC = Path(__file__).resolve().parents[1] / "public"
if str(PUBLIC) not in sys.path:
    sys.path.insert(0, str(PUBLIC))
import engine as e
import search as cpu


def copy_board(board):
    result = board.copy()
    result.move_stack = []
    return result


def terminal_value(board):
    outcome = cpu._terminal(board)
    if outcome is None:
        return None
    if outcome["winner"] is None:
        return 0.0
    winner = e.RED if outcome["winner"] == "RED" else e.BLUE
    return 1.0 if winner == board.turn else -1.0
