"""Held-out tactical gates for a checkpoint, with both colors of each position.

These reported mistakes are evaluation fixtures, never training examples. A
failed gate is recorded in the report; this command does not assert that an
untrained or prototype checkpoint has reached production playing strength.
"""
import argparse
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path

import torch

from .mcts import MCTS
from .model import TorchEvaluator, load_checkpoint, rules_source_hash
from .pipeline import write_json
from .rules import e, cpu, copy_board
from .train import device_name


@dataclass
class TacticalCase:
    name: str
    objective: str
    description: str
    board: object
    known_bad_moves: tuple[str, ...]
    forbidden_move: str | None = None


def _mirror_move(action):
    move = e.Move.from_uci(action)
    return replace(move,
                   from_square=None if move.from_square is None else 63 - move.from_square,
                   to_square=None if move.to_square is None else 63 - move.to_square,
                   capture_preference=None if move.capture_preference is None else 63 - move.capture_preference,
                   orientation=None if move.orientation is None else (move.orientation + 4) % 8).uci()


def _opening(turns):
    board = e.BaseBoard()
    for turn in turns:
        for action in turn.split():
            legal = {move.uci(): move for move in board.generate_legal_moves()}
            if action not in legal or cpu._terminal(board) is not None:
                raise ValueError(f"Invalid tactical fixture opening: {action}")
            board.push(legal[action])
    board.move_stack = []
    return board


def tactical_cases(include_hq=True):
    """The exact positions from tests/test_search.py, plus their 180° mirrors."""
    cases = [
        TacticalCase(
            "empty-paradrop", "airborne_safety",
            "Avoid c8g1 and an immediate free capture of the airborne unit.",
            _opening(("rhe1 rtd1 rrc1", "rhe8 rpc8 rtg8", "rpa1 f2f1 g1f2↑")),
            ("c8g1", "a7b6", "rrf8"), "c8g1"),
        TacticalCase(
            "trapped-paradrop", "airborne_home",
            "Keep the airborne unit on its own home rank instead of trading it for a regular gun.",
            _opening(("rhe1 rtd1 rrc1", "rhd8 rpc8 rte8", "rrf1 e1e2↑ f2e1")),
            ("c8b1xc1", "a7b6", "rrf8")),
        TacticalCase(
            "heavy-over-armored", "heavy_capture",
            "Capture the enemy heavy artillery while leaving the armored artillery.",
            _opening(("rhe1 rtd1 rrf1", "rhd8 rpc8 rte8", "rrc1 rpb1 e1e2↑")),
            ("rff8", "a7b6", "c8e1xd1")),
        TacticalCase(
            "turn22-large-guns", "large_gun_safety",
            "Preserve both large guns without an immediate legal enemy capture after bombardments.",
            e.BaseBoard("1iffi1r↓r↓/qiiit↓3/r↓1i2R↑2/3h↓2H↑R↑/6R↖1/1FF3I1/4II2/1P5Q IIIIIF ii b"),
            ("d5e4↘", "e7e7↘", "ria8")),
    ]
    if include_hq:
        board = e.BaseBoard()
        board.clear_board()
        for square, symbol in {"a1": "Q", "h8": "q", "g8": "I", "g6": "I"}.items():
            board.set_piece_at(e.parse_square(square), e.Piece.from_symbol(symbol))
        cases.append(TacticalCase("immediate-hq", "hq_capture", "Take the available immediate HQ capture.",
                                  board, ("skip",)))
    result = []
    for case in cases:
        for mirrored in (False, True):
            board = case.board.mirror() if mirrored else case.board
            color = "red" if board.turn == e.RED else "blue"
            result.append(replace(case, name=f"{case.name}-{color}", board=board,
                                  known_bad_moves=tuple(_mirror_move(m) for m in case.known_bad_moves) if mirrored else case.known_bad_moves,
                                  forbidden_move=_mirror_move(case.forbidden_move) if mirrored and case.forbidden_move else case.forbidden_move))
    return result


def judge_plan(case, moves):
    """Validate and judge a remaining-turn plan using exact moves/material.

    No CPU search or learned evaluator determines whether a gate passes.
    The H+T gate resolves start-of-turn automatic captures, then examines the
    opponent's authoritative legal capture choices. It is an immediate-safety
    gate, not a claim that the position is safe against every multi-action trap.
    """
    board = copy_board(case.board)
    player = board.turn
    result = {"legal": True, "complete": False, "passed": False, "checks": {}, "error": None}
    for index, action in enumerate(moves):
        if board.turn != player or cpu._terminal(board) is not None:
            result.update(legal=False, error=f"Action {index + 1} continues after the turn ended")
            return result
        legal = {move.uci(): move for move in board.generate_legal_moves()}
        if action not in legal:
            result.update(legal=False, error=f"Illegal action {index + 1}: {action}")
            return result
        board.push(legal[action])
    terminal = cpu._terminal(board)
    result["complete"] = terminal is not None or board.turn != player
    result["final_state"] = board.serialize()
    if not result["complete"]:
        result["error"] = "Plan does not finish the current player's turn"
        return result
    checks = result["checks"]
    ours = board.occupied_co[player]
    if case.objective == "airborne_safety":
        checks["reported_sacrifice_avoided"] = case.forbidden_move not in moves
        checks["airborne_preserved"] = bool(board.airborne_infantry & ours)
        checks["no_immediate_airborne_capture"] = not any(
            move.name == "AutoCapture" and move.capture_preference is not None and
            e.BB_SQUARES[move.capture_preference] & board.airborne_infantry & ours
            for move in copy_board(board).generate_legal_moves())
    elif case.objective == "airborne_home":
        home = e.BB_RANK_1 if player == e.RED else e.BB_RANK_8
        checks["airborne_on_own_home_rank"] = bool(board.airborne_infantry & ours & home)
    elif case.objective == "heavy_capture":
        enemy = board.occupied_co[not player]
        checks["enemy_heavy_captured"] = not bool(board.heavy_artillery & enemy)
        checks["enemy_armored_not_taken_instead"] = bool(board.armored_artillery & enemy)
    elif case.objective == "large_gun_safety":
        probe = copy_board(board)
        threatened = set()
        automatic = []
        # This fixture has an unrelated gun lost to bombardment before the
        # opponent's first chosen action. Track any large-gun exposure during
        # automatic captures as well as after the automatic phase finishes.
        for _ in range(probe.occupied.bit_count() + 1):
            if cpu._terminal(probe) is not None:
                legal = []
                break
            legal = sorted(probe.generate_legal_moves(), key=lambda move: move.uci())
            large = (probe.heavy_artillery | probe.armored_artillery) & probe.occupied_co[player]
            threatened.update(move.capture_preference for move in legal
                              if move.capture_preference is not None and
                              e.BB_SQUARES[move.capture_preference] & large)
            if not legal or legal[0].name != "AutoCapture":
                break
            automatic.append(legal[0].uci())
            probe.push(legal[0])
        large = (probe.heavy_artillery | probe.armored_artillery) & probe.occupied_co[player]
        checks["both_large_guns_preserved"] = large.bit_count() == 2
        checks["no_immediate_large_gun_capture"] = not threatened
        result["automatic_reply_captures"] = automatic
        result["exposed_large_gun_squares"] = [e.square_name(s) for s in sorted(threatened)]
    elif case.objective == "hq_capture":
        winner = "RED" if player == e.RED else "BLUE"
        checks["hq_win"] = terminal is not None and terminal["winner"] == winner
    else:
        raise ValueError(f"Unknown tactical objective: {case.objective}")
    result["passed"] = bool(checks) and all(checks.values())
    return result


def run_case(case, evaluator, *, time_ms=2400, simulations=100000, seed=0):
    search = MCTS(evaluator, simulations=simulations, seed=seed)
    plan = search.plan_turn(case.board, time_ms=time_ms)
    judgement = judge_plan(case, plan.moves)
    return {"name": case.name, "objective": case.objective, "description": case.description,
            "player": "RED" if case.board.turn == e.RED else "BLUE",
            "state": case.board.serialize(), "moves": plan.moves, **judgement,
            "elapsed_ms": plan.elapsed_ms, "timed_out": plan.timed_out,
            "value_current_player": plan.value,
            "simulations": sum(item.simulations for item in plan.searches),
            "nodes": sum(item.nodes for item in plan.searches)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--time-ms", type=float, default=2400)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="cpu")
    parser.add_argument("--simulations", type=int, default=100000)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=4173)
    args = parser.parse_args()
    if args.time_ms <= 0 or min(args.simulations, args.threads) < 1:
        parser.error("Time, simulations, and threads must be positive")
    if args.output.exists():
        parser.error("Tactical report exists; select a new output path")
    torch.set_num_threads(args.threads)
    device = device_name(args.device)
    model, _ = load_checkpoint(args.checkpoint, device)
    evaluator = TorchEvaluator(model, device)
    warm = e.BaseBoard()
    evaluator.predict(warm, list(warm.generate_legal_moves()))
    cases = tactical_cases()
    report = {"checkpoint": str(args.checkpoint),
              "checkpoint_sha256": hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              "rules_sha256": rules_source_hash(), "usage": "evaluation_only",
              "device": device, "time_ms_per_turn": args.time_ms,
              "simulations_cap_per_action": args.simulations, "seed": args.seed,
              "expected_cases": len(cases), "complete": False,
              "passed": 0, "failed": 0, "cases": [], "promoted": False}
    for case in cases:
        result = run_case(case, evaluator, time_ms=args.time_ms,
                          simulations=args.simulations, seed=args.seed)
        report["cases"].append(result)
        report["passed"] += int(result["passed"])
        report["failed"] += int(not result["passed"])
        report["complete"] = len(report["cases"]) == len(cases)
        write_json(args.output, report)
        print(json.dumps({"name": case.name, "passed": result["passed"], "moves": result["moves"],
                          "checks": result["checks"], "elapsed_ms": result["elapsed_ms"]}), flush=True)


if __name__ == "__main__":
    main()
