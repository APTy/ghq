"""CPU-teacher warm starts and neural self-play using the production rules."""
import json
import random
import time

from .data import Sample, label_game
from .mcts import MCTS
from .rules import e, cpu, copy_board


def random_opening(seed, turns=0):
    board = e.BaseBoard()
    rng = random.Random(seed)
    completed = 0
    while completed < turns and cpu._terminal(board) is None:
        player = board.turn
        moves = list(board.generate_legal_moves())
        active = [move for move in moves if move.name != "Skip"]
        board.push(rng.choice(active or moves))
        completed += board.turn != player
    board.move_stack = []
    return board


def play_game(evaluator=None, *, teacher=False, teacher_ms=100, simulations=64,
              max_turns=100, seed=0, opening_turns=0, game_id="", start_board=None):
    board = copy_board(start_board) if start_board is not None else random_opening(seed, opening_turns)
    rng = random.Random(seed)
    search = None if teacher else MCTS(evaluator, simulations=simulations, seed=seed)
    samples, trajectory = [], []
    completed_turns, teacher_plan, repeated = 0, [], {}
    started = time.perf_counter()
    reason = "max-turns"
    while completed_turns < max_turns:
        terminal = cpu._terminal(board)
        if terminal is not None:
            reason = terminal["reason"]
            break
        key = cpu.state_key(board)
        repeated[key] = repeated.get(key, 0) + 1
        if repeated[key] >= 4:
            reason = "repetition-cutoff"
            break
        player = board.turn
        moves = sorted(board.generate_legal_moves(), key=lambda move: move.uci())
        actions = [move.uci() for move in moves]
        if teacher:
            if not teacher_plan:
                analysis = json.loads(cpu.analyze_json(board.serialize(), time_ms=teacher_ms,
                                                       max_depth=2, beam_width=4, multi_pv=1))
                teacher_plan = list(analysis["turnMoves"])
            if not teacher_plan:
                raise RuntimeError("CPU teacher returned no action in a nonterminal state")
            selected = teacher_plan.pop(0)
            policy = [float(action == selected) for action in actions]
        else:
            result = search.search(board, add_noise=True)
            if len(result.actions) != len(actions) or set(result.actions) != set(actions):
                raise RuntimeError("MCTS actions differ from authoritative legal moves")
            actions, policy = result.actions, result.policy
            # Early exploration; late play uses visit maxima. The target always
            # remains the full visit distribution, independent of sampling.
            selected = (rng.choices(actions, weights=policy, k=1)[0] if completed_turns < 12
                        else actions[max(range(len(actions)), key=lambda i: policy[i])])
        if selected not in actions:
            raise RuntimeError(f"Player selected illegal action {selected}")
        sample = Sample(board.serialize(), actions, policy, game_id=game_id,
                        source=cpu.ENGINE_VERSION if teacher else "mcts")
        sample.validate()
        samples.append(sample)
        board.push(e.Move.from_uci(selected))
        trajectory.append(selected)
        if board.turn != player:
            completed_turns += 1
            teacher_plan = []
        # Search needs no history; bound trajectory memory in the rules object.
        board.move_stack = []
    terminal = cpu._terminal(board)
    if terminal is not None:
        reason = terminal["reason"]
    label_game(samples, terminal)
    return samples, {"game_id": game_id, "seed": seed, "teacher": teacher,
                     "turns": completed_turns, "actions": len(trajectory), "moves": trajectory,
                     "terminal": terminal, "truncated": terminal is None, "reason": reason,
                     "seconds": time.perf_counter() - started,
                     "final_state": board.serialize()}
