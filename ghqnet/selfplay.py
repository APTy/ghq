"""CPU-teacher warm starts and neural self-play using the production rules."""
import json
import math
import random
import time
from copy import deepcopy

from .data import Sample, label_game
from .mcts import MCTS
from .rules import e, cpu, copy_board
from .value_targets import PROVENANCE_VERSION, validate_value_provenance


def _value_provenance(*, teacher, opponent, learner_color, model_id, opponent_model_id,
                      search, other_search, teacher_ms, teacher_depth, teacher_beam_width,
                      add_noise, temperature_turns, terminal, reason, final_state):
    """Record the actual players and collection policy after the game finishes."""
    cpu_player = {"model_id": cpu.ENGINE_VERSION,
                  "search": {"kind": "cpu", "time_ms": teacher_ms, "max_depth": teacher_depth,
                             "beam_width": teacher_beam_width, "multi_pv": 1}}

    def neural_player(identifier, tree, is_opponent=False):
        return {"model_id": identifier or "", "search": {
            "kind": "neural", "simulations": tree.simulations, "cpuct": tree.cpuct,
            "max_nodes": tree.max_nodes, "max_depth": tree.max_depth,
            "root_noise": bool(add_noise and not is_opponent), "dirichlet_alpha": 0.3,
            "noise_fraction": 0.25, "temperature_turns": 0 if is_opponent else temperature_turns,
            "temperature": 1.0, "late_temperature": 0.0}}

    if teacher:
        players = {"RED": deepcopy(cpu_player), "BLUE": deepcopy(cpu_player)}
    elif opponent == "self":
        learner = neural_player(model_id, search)
        players = {"RED": deepcopy(learner), "BLUE": deepcopy(learner)}
    else:
        learner = neural_player(model_id, search)
        other = cpu_player if opponent == "cpu" else neural_player(opponent_model_id, other_search, True)
        players = {"RED": learner if learner_color == e.RED else other,
                   "BLUE": learner if learner_color == e.BLUE else other}
    result = {"version": PROVENANCE_VERSION, "collection_kind": "teacher" if teacher else opponent,
              "players": players, "terminal": deepcopy(terminal), "termination_reason": reason,
              "final_state": final_state}
    validate_value_provenance(result)
    return result


class _MeasuredEvaluator:
    """Count synchronous predict calls without changing their inputs/outputs.

    Time includes encoding, device transfers, inference, and returning values
    to the host. This is end-to-end evaluator latency, not CUDA kernel time.
    """
    def __init__(self, evaluator):
        self.evaluator = evaluator
        self.calls = 0
        self.seconds = 0.0
        self.root_player = None
        self.opposite_player_calls = 0

    def predict(self, board, legal_moves):
        started = time.perf_counter()
        self.calls += 1
        self.opposite_player_calls += self.root_player is not None and board.turn != self.root_player
        try:
            return self.evaluator.predict(board, legal_moves)
        finally:
            self.seconds += time.perf_counter() - started


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
              max_turns=100, seed=0, opening_turns=0, game_id="", start_board=None,
              start_state=None, opponent="self", learner_color=None,
              opponent_evaluator=None, opponent_simulations=None, model_id=None,
              opponent_model_id=None, teacher_depth=3, teacher_beam_width=8,
              add_noise=True, temperature_turns=12):
    """Collect a game, retaining only learner decisions against fixed opponents.

    ``teacher=True`` collects CPU targets for both sides. Ordinary self-play
    retains both sides. For ``cpu`` or ``historical`` opponents, learner_color
    is required and opponent decisions are played but omitted from Samples.
    Historical opponents use their own fixed evaluator/tree, without noise.
    All values and outcome labels still refer to the state’s actual mover.
    ``start_state`` is the full serialized rules state, including turn budget.
    """
    started = time.perf_counter()
    if opponent == "history":
        opponent = "historical"
    if opponent not in ("self", "cpu", "historical"):
        raise ValueError("opponent must be self, cpu, or historical")
    if teacher and opponent != "self":
        raise ValueError("Teacher games already use CPU players on both sides")
    if isinstance(learner_color, str):
        if learner_color.upper() not in ("RED", "BLUE"):
            raise ValueError("learner_color must be RED or BLUE")
        learner_color = e.RED if learner_color.upper() == "RED" else e.BLUE
    if opponent != "self" and learner_color not in (e.RED, e.BLUE):
        raise ValueError("Mixed games require learner_color")
    if not teacher and evaluator is None:
        raise ValueError("Neural games require an evaluator")
    if opponent == "historical" and opponent_evaluator is None:
        raise ValueError("Historical games require opponent_evaluator")
    if any(identifier is not None and (not isinstance(identifier, str) or not identifier.strip())
           for identifier in (model_id, opponent_model_id)):
        raise ValueError("Model identities must be nonempty strings or absent")
    if start_state is not None and start_board is not None:
        raise ValueError("Specify only one of start_state and start_board")
    if (simulations < 0 or max_turns < 1 or opening_turns < 0 or temperature_turns < 0 or
            teacher_depth < 1 or teacher_beam_width < 1 or not math.isfinite(teacher_ms) or teacher_ms < 0):
        raise ValueError("Invalid collection/search budget")
    if start_state is not None:
        board = e.BaseBoard.deserialize(start_state)
    else:
        board = copy_board(start_board) if start_board is not None else random_opening(seed, opening_turns)
    rng = random.Random(seed)
    measured = None if teacher else _MeasuredEvaluator(evaluator)
    search = None if teacher else MCTS(measured, simulations=simulations, seed=seed)
    other_measured = _MeasuredEvaluator(opponent_evaluator) if opponent == "historical" else None
    other_search = (MCTS(other_measured, simulations=simulations if opponent_simulations is None else opponent_simulations,
                         seed=seed + 1) if other_measured is not None else None)
    counters = {"search_calls": 0, "search_simulations": 0, "search_nodes": 0,
                "search_seconds": 0.0, "teacher_search_calls": 0,
                "teacher_search_nodes": 0, "teacher_search_seconds": 0.0,
                "root_legal_actions": 0, "root_visited_actions": 0,
                "root_policy_entropy_sum": 0.0, "searches_with_opposite_player_evaluation": 0,
                "turn_start_skips": 0, "learner_turn_start_skips": 0,
                "opponent_turn_start_skips": 0}
    samples, trajectory, action_sources, action_roles, sample_indices = [], [], [], [], []
    completed_turns, teacher_plan, repeated = 0, [], {}
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
        is_opponent = not teacher and opponent != "self" and player != learner_color
        is_cpu = teacher or (is_opponent and opponent == "cpu")
        role = "teacher" if teacher else "opponent" if is_opponent else "learner"
        moves = sorted(board.generate_legal_moves(), key=lambda move: move.uci())
        actions = [move.uci() for move in moves]
        if is_cpu:
            if not teacher_plan:
                search_started = time.perf_counter()
                analysis = json.loads(cpu.analyze_json(board.serialize(), time_ms=teacher_ms,
                                                       max_depth=teacher_depth, beam_width=teacher_beam_width, multi_pv=1))
                counters["teacher_search_seconds"] += time.perf_counter() - search_started
                counters["teacher_search_calls"] += 1
                counters["teacher_search_nodes"] += analysis["nodes"]
                teacher_plan = list(analysis["turnMoves"])
            if not teacher_plan:
                raise RuntimeError("CPU teacher returned no action in a nonterminal state")
            selected = teacher_plan.pop(0)
            policy = [float(action == selected) for action in actions]
            source = cpu.ENGINE_VERSION
        else:
            active_search, active_measured = (other_search, other_measured) if is_opponent else (search, measured)
            active_measured.root_player = player
            opposite_before = active_measured.opposite_player_calls
            search_started = time.perf_counter()
            result = active_search.search(board, add_noise=add_noise and not is_opponent)
            counters["search_seconds"] += time.perf_counter() - search_started
            counters["search_calls"] += 1
            counters["search_simulations"] += result.simulations
            counters["search_nodes"] += result.nodes
            if len(result.actions) != len(actions) or set(result.actions) != set(actions):
                raise RuntimeError("MCTS actions differ from authoritative legal moves")
            actions, policy = result.actions, result.policy
            counters["root_legal_actions"] += len(actions)
            counters["root_visited_actions"] += sum(count > 0 for count in result.visits)
            counters["root_policy_entropy_sum"] += -sum(p * math.log(p) for p in policy if p > 0)
            counters["searches_with_opposite_player_evaluation"] += active_measured.opposite_player_calls > opposite_before
            # Early exploration; late play uses visit maxima. The target always
            # remains the full visit distribution, independent of sampling.
            selected = (rng.choices(actions, weights=policy, k=1)[0] if not is_opponent and completed_turns < temperature_turns
                        else actions[max(range(len(actions)), key=lambda i: policy[i])])
            identifier = opponent_model_id if is_opponent else model_id
            source = "mcts" if identifier is None else f"mcts:{identifier}"
        if selected not in actions:
            raise RuntimeError(f"Player selected illegal action {selected}")
        if not is_opponent:
            # Retention priority identifies tactical opportunities/danger, not
            # an oracle action or a corrected target. Keep the exact search
            # distribution (including mistakes) and actual terminal labels.
            valuable = (board.artillery | board.armored_artillery | board.heavy_artillery |
                        board.airborne_infantry | board.hq)
            capture_available = any(move.name != "AutoCapture" and move.capture_preference is not None and
                                    e.BB_SQUARES[move.capture_preference] & valuable for move in moves)
            hq_in_danger = bool(board.hq & board.occupied_co[player] & board.bombarded_co[not player])
            sample = Sample(board.serialize(), actions, policy, game_id=game_id, source=source,
                            priority="hard" if capture_available or hq_in_danger else "normal",
                            model_id=cpu.ENGINE_VERSION if is_cpu else model_id or "")
            sample.validate()
            samples.append(sample)
            sample_indices.append(len(trajectory))
        if selected == "skip" and board.turn_moves == 0:
            counters["turn_start_skips"] += 1
            counters["opponent_turn_start_skips" if is_opponent else "learner_turn_start_skips"] += 1
        board.push(e.Move.from_uci(selected))
        trajectory.append(selected)
        action_sources.append(source)
        action_roles.append(role)
        if board.turn != player:
            completed_turns += 1
            teacher_plan = []
        # Search needs no history; bound trajectory memory in the rules object.
        board.move_stack = []
    terminal = cpu._terminal(board)
    if terminal is not None:
        reason = terminal["reason"]
    label_game(samples, terminal)
    final_state = board.serialize()
    provenance = _value_provenance(
        teacher=teacher, opponent=opponent, learner_color=learner_color, model_id=model_id,
        opponent_model_id=opponent_model_id, search=search, other_search=other_search,
        teacher_ms=teacher_ms, teacher_depth=teacher_depth, teacher_beam_width=teacher_beam_width,
        add_noise=add_noise, temperature_turns=temperature_turns,
        terminal=terminal, reason=reason, final_state=final_state)
    for sample in samples:
        sample.value_provenance = deepcopy(provenance)
    measurements = [item for item in (measured, other_measured) if item is not None]
    return samples, {"game_id": game_id, "seed": seed, "teacher": teacher,
                     "opponent": opponent, "learner_color": None if learner_color is None else
                     "RED" if learner_color == e.RED else "BLUE", "model_id": model_id,
                     "opponent_model_id": opponent_model_id, "action_sources": action_sources,
                     "action_roles": action_roles, "sample_action_indices": sample_indices,
                     "samples": len(samples), "teacher_ms": teacher_ms,
                     "hard_examples": sum(sample.priority == "hard" for sample in samples),
                     "teacher_depth": teacher_depth, "teacher_beam_width": teacher_beam_width,
                     "turns": completed_turns, "actions": len(trajectory), "moves": trajectory,
                     "terminal": terminal, "truncated": terminal is None, "reason": reason,
                     "value_provenance": provenance,
                     "seconds": time.perf_counter() - started,
                     "final_state": final_state, **counters,
                     "evaluator_calls": sum(item.calls for item in measurements),
                     "evaluator_seconds": sum(item.seconds for item in measurements),
                     "opposite_player_evaluator_calls": sum(item.opposite_player_calls for item in measurements),
                     "learner_evaluator_calls": measured.calls if measured else 0,
                     "opponent_evaluator_calls": other_measured.calls if other_measured else 0}
