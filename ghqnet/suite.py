"""Freeze verifiable tactical objectives and separately labelled replay candidates.

Example: python -m ghqnet.suite --replays RUN/replay --output EVAL --count 250

A capture witness proves that a particular positive objective is attainable in
the remaining turn. It does not prove optimal play, a forced game result, or a
good trade after the opponent replies. Defense cases quantify every forced
capture and the opponent's first chosen action, not its entire three-action
turn. Mirrors are one case, and related positions from a game stay in one split.
"""
import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

from .rules import e, cpu, copy_board, PUBLIC

SUITE_VERSION = 1
MATERIAL = {e.INFANTRY: 1, e.ARMORED_INFANTRY: 2, e.AIRBORNE_INFANTRY: 4,
            e.ARTILLERY: 3, e.ARMORED_ARTILLERY: 4, e.HEAVY_ARTILLERY: 5}


def rules_hash():
    return hashlib.sha256((PUBLIC / "engine.py").read_bytes()).hexdigest()


def canonical_state_id(board):
    """Exact model mirror equivalence; includes reserves and all mid-turn fields."""
    return hashlib.sha256(min(board.serialize(), board.mirror().serialize()).encode("ascii")).hexdigest()


def mirror_move(action):
    move = e.Move.from_uci(action)
    return replace(move, from_square=None if move.from_square is None else 63 - move.from_square,
                   to_square=None if move.to_square is None else 63 - move.to_square,
                   capture_preference=None if move.capture_preference is None else 63 - move.capture_preference,
                   orientation=None if move.orientation is None else (move.orientation + 4) % 8).uci()


def material(board, color):
    # Include reserves so deployment alone neither creates nor loses material.
    return sum(weight * (board.pieces_mask(kind, color).bit_count() +
                         dict(board.reserves[color]).get(kind, 0)) for kind, weight in MATERIAL.items())


def play_plan(board, actions):
    """Exact, pure remaining-turn validation. No post-terminal actions allowed."""
    result = copy_board(board)
    player = result.turn
    captures = []
    for action in actions:
        if result.turn != player or cpu._terminal(result) is not None:
            raise ValueError("Action after the player's turn or game ended")
        # Legal generation initializes the engine's pending free-capture state.
        # Keep that state on the private board which will receive the move.
        legal = {move.uci(): move for move in result.generate_legal_moves()}
        if action not in legal:
            raise ValueError(f"Illegal action: {action}")
        move = legal[action]
        if move.capture_preference is not None:
            captures.append({"type": result.piece_type_at(move.capture_preference),
                             "automatic": move.name == "AutoCapture"})
        result.push(move)
    complete = result.turn != player or cpu._terminal(result) is not None
    return result, complete, captures


def finish_with_skip(board, prefix):
    after, complete, _ = play_plan(board, prefix)
    if complete:
        return list(prefix)
    # Generated capture prefixes have finished any initial automatic phase.
    legal = {move.uci() for move in copy_board(after).generate_legal_moves()}
    if "skip" not in legal:
        raise ValueError("Prefix leaves pending automatic captures")
    return [*prefix, "skip"]


def immediate_hq_loss(board, defender, max_states=512):
    """Any forced-capture branch or next chosen enemy action can take the HQ.

    Return None on the explicit enumeration limit; uncertainty never passes a
    defense gate. This is existential danger / universal immediate safety.
    """
    pending, seen = [copy_board(board)], set()
    while pending:
        current = pending.pop()
        key = cpu.state_key(current)
        if key in seen:
            continue
        seen.add(key)
        if len(seen) > max_states:
            return None
        terminal = cpu._terminal(current)
        if terminal is not None:
            if terminal["winner"] == ("BLUE" if defender == e.RED else "RED"):
                return True
            continue
        if current.turn == defender:
            raise ValueError("Defense probe requires the opponent to move")
        moves = list(current.generate_legal_moves())
        hq = current.hq & current.occupied_co[defender]
        if any(move.capture_preference is not None and e.BB_SQUARES[move.capture_preference] & hq
               for move in moves):
            return True
        if moves and moves[0].name == "AutoCapture":
            for move in moves:
                child = copy_board(current)
                child.push(move)
                pending.append(child)
    return False


def judge_case(case, actions, *, mirrored=False):
    board = e.BaseBoard.deserialize(case["state"])
    if mirrored:
        board = board.mirror()
    player = board.turn
    result = {"legal": False, "complete": False, "passed": False, "checks": {}, "error": None}
    try:
        after, complete, captures = play_plan(board, actions)
    except ValueError as error:
        result["error"] = str(error)
        return result
    result.update(legal=True, complete=complete, final_state=after.serialize())
    if not complete:
        result["error"] = "Plan does not complete the current turn"
        return result
    objective = case.get("objective")
    if not objective or case["evidence"] != "verified_witness":
        result["error"] = "Unverified/teacher-approximate candidates have no scored objective"
        return result
    terminal = cpu._terminal(after)
    ours = "RED" if player == e.RED else "BLUE"
    won = terminal is not None and terminal["winner"] == ours
    checks = result["checks"]
    checks["own_hq_preserved"] = bool(after.hq & after.occupied_co[player])
    if objective["kind"] == "capture_material":
        checks["enemy_material_taken_or_hq_won"] = won or material(board, not player) - material(after, not player) >= objective["min_enemy_loss"]
        checks["own_material_preserved_or_hq_won"] = won or material(after, player) >= material(board, player)
        checks["chosen_capture_played"] = any(not item["automatic"] for item in captures)
    elif objective["kind"] == "capture_hq":
        checks["hq_captured"] = terminal is not None and terminal["winner"] == ours
    elif objective["kind"] == "defend_hq_one_action":
        danger = (False if terminal is not None and terminal["winner"] == ours else
                  immediate_hq_loss(after, player))
        checks["positive_defense_played"] = any(e.Move.from_uci(action).name not in ("Skip", "AutoCapture") for action in actions)
        checks["no_forced_or_first_action_hq_loss"] = danger is False
        result["reply_enumeration_complete"] = danger is not None
    else:
        raise ValueError(f"Unknown suite objective: {objective['kind']}")
    result["passed"] = all(checks.values())
    return result


def verify_case(case):
    board = e.BaseBoard.deserialize(case["state"])
    if case["canonical_state_id"] != canonical_state_id(board):
        raise ValueError("Canonical state fingerprint mismatch")
    if case["id"] != "case-" + case["canonical_state_id"][:20]:
        raise ValueError("Case identifier does not match its state")
    if cpu._terminal(board) is not None:
        raise ValueError("Suite case must be nonterminal")
    if case["split"] not in ("train", "dev", "test"):
        raise ValueError("Invalid split")
    if case["evidence"] not in ("verified_witness", "unverified", "teacher_approximate"):
        raise ValueError("Invalid evidence level")
    if case["evidence"] == "verified_witness":
        if not case.get("witness"):
            raise ValueError("Verified case has no witness")
        objective = case.get("objective", {})
        if objective.get("kind") == "capture_material" and (
                not isinstance(objective.get("min_enemy_loss"), int) or objective["min_enemy_loss"] <= 0):
            raise ValueError("Material objectives require a positive integral material gain")
        if "witness_chosen_actions" in objective:
            chosen = sum(e.Move.from_uci(action).name not in ("Skip", "AutoCapture") for action in case["witness"])
            if chosen != objective["witness_chosen_actions"]:
                raise ValueError("Witness action-depth annotation mismatch")
        if objective.get("kind") == "defend_hq_one_action":
            skipped, _, _ = play_plan(board, ["skip"])
            if immediate_hq_loss(skipped, board.turn) is not True:
                raise ValueError("Defense case lacks a verified baseline HQ threat")
        for mirrored in (False, True):
            actions = [mirror_move(move) for move in case["witness"]] if mirrored else case["witness"]
            result = judge_case(case, actions, mirrored=mirrored)
            if not result["passed"]:
                raise ValueError(f"Invalid witness for {case['id']}: {result}")
            # A passive skip must never satisfy a positive generated objective.
            if judge_case(case, ["skip"], mirrored=mirrored)["passed"]:
                raise ValueError("Positive objective can be passed by skipping")
    elif case.get("objective") is not None or case.get("witness"):
        raise ValueError("Unverified candidates cannot carry a proven objective/witness")
    return board


def _split(family_id, seed):
    bucket = int(hashlib.sha256(f"{seed}:{family_id}".encode()).hexdigest()[:8], 16) % 100
    return "train" if bucket < 70 else "dev" if bucket < 85 else "test"


def _case(board, family, game_id, source, objective, witness, category, seed, evidence="verified_witness"):
    identity = canonical_state_id(board)
    return {"id": f"case-{identity[:20]}", "canonical_state_id": identity,
            "family_id": family, "game_id": game_id, "split": _split(family, seed),
            "category": category, "evidence": evidence, "state": board.serialize(),
            "objective": objective, "witness": witness, "source": source}


def _capture_witness(board, continuation):
    """Prefer a witnessed two-action capture when it beats any immediate victim.

    This is a labelled local objective, not an optimal-move oracle. No outcome
    label or CPU score is used to decide that a tactical witness is correct.
    """
    legal = list(copy_board(board).generate_legal_moves())
    choices = [move for move in legal if move.capture_preference is not None and move.name != "AutoCapture"]
    value = lambda move, position: 100 if position.piece_type_at(move.capture_preference) == e.HQ else MATERIAL[position.piece_type_at(move.capture_preference)]
    best = max(choices, key=lambda move: (value(move, board), move.uci()), default=None)
    prefix, victim = ([best.uci()], board.piece_type_at(best.capture_preference)) if best else (None, None)
    if len(continuation) >= 2:
        first = next((move for move in legal if move.uci() == continuation[0]), None)
        if first and first.name != "AutoCapture" and first.capture_preference is None:
            after, complete, _ = play_plan(board, [first.uci()])
            if not complete:
                second = next((move for move in copy_board(after).generate_legal_moves()
                               if move.uci() == continuation[1] and move.capture_preference is not None
                               and move.name != "AutoCapture"), None)
                if second and (best is None or value(second, after) > value(best, board)):
                    prefix, victim = [first.uci(), second.uci()], after.piece_type_at(second.capture_preference)
    if prefix is None:
        return None
    objective = ({"kind": "capture_hq"} if victim == e.HQ else
                 {"kind": "capture_material", "min_enemy_loss": MATERIAL[victim]})
    objective["proof_scope"] = "existence_of_legal_remaining_turn_plan"
    objective["witness_chosen_actions"] = len(prefix)
    witness = finish_with_skip(board, prefix)
    category = ("hq_capture" if victim == e.HQ else f"capture_{e.piece_name(victim).lower().replace(' ', '_')}")
    return objective, witness, category


def _read_game(path):
    from .data import read_replay
    samples = read_replay(path)
    header = json.loads(path.read_text().splitlines()[0])
    game = header.get("metadata", {}).get("game", {})
    moves = game.get("moves")
    indices = game.get("sample_action_indices", list(range(len(samples))))
    if not isinstance(moves, list) or len(indices) != len(samples) or indices != sorted(set(indices)):
        raise ValueError(f"{path}: expected a game replay with an action trajectory")
    if not samples:
        # A fixed opponent can end a mixed game before the learner acts. There
        # is no retained state from which to verify a trajectory or mine a case.
        return [], moves, []
    if indices[0] < 0 or indices[-1] >= len(moves):
        raise ValueError("Replay sample indices out of range")
    # Mixed-opponent records may start after an opponent move; validate the
    # complete available suffix and every retained state, never invent a prefix.
    board = e.BaseBoard.deserialize(samples[0].state)
    by_index = dict(zip(indices, samples))
    for index in range(indices[0], len(moves)):
        if cpu._terminal(board) is not None:
            raise ValueError("Replay contains post-terminal action")
        legal = {move.uci(): move for move in board.generate_legal_moves()}
        if index in by_index and board.serialize() != by_index[index].state:
            raise ValueError(f"{path}: replay trajectory/state mismatch at action {index}")
        if moves[index] not in legal:
            raise ValueError(f"Illegal replay trajectory action {moves[index]}")
        board.push(legal[moves[index]])
    if board.serialize() != game.get("final_state") or cpu._terminal(board) != game.get("terminal"):
        raise ValueError("Replay final state/outcome mismatch")
    return samples, moves, indices


def defense_cases(seed=4173, count=24):
    """Small constructed HQ-evacuation families, with exact immediate defenses.

    These boards have engine-valid inventories/geometry; they are explicitly
    constructed, not claimed to have an initial-position reachability proof.
    All translations and variants of an attacker pair remain one family.
    """
    result = []
    for attackers in ((e.INFANTRY, e.INFANTRY), (e.INFANTRY, e.ARMORED_INFANTRY),
                      (e.ARMORED_INFANTRY, e.INFANTRY)):
        for rank in (2, 3, 4):
            for file in (2, 3, 4):
                board = e.BaseBoard()
                board.clear_board()
                hq = e.square(file, rank)
                for square, kind, color in ((hq, e.HQ, e.RED), (e.H8, e.HQ, e.BLUE),
                                            (e.square(file - 1, rank), attackers[0], e.BLUE),
                                            (e.square(file + 1, rank + 2), attackers[1], e.BLUE)):
                    board.set_piece_at(square, e.Piece(kind, color))
                skipped, _, _ = play_plan(board, ["skip"])
                if immediate_hq_loss(skipped, e.RED) is not True:
                    continue
                objective = {"kind": "defend_hq_one_action", "proof_scope": "all_forced_captures_then_one_opponent_action"}
                family = "constructed:hq-evacuation:" + "-".join(map(str, attackers))
                case = _case(board, family, None, {"kind": "constructed", "template": family,
                             "reachability": "engine_valid_constructed_position"}, objective, [], "hq_defense", seed)
                for move in copy_board(board).generate_legal_moves():
                    if move.name == "Skip":
                        continue
                    witness = finish_with_skip(board, [move.uci()])
                    if judge_case(case, witness)["passed"]:
                        case["witness"] = witness
                        verify_case(case)
                        result.append(case)
                        break
                if len(result) >= count:
                    return result
    return result


def summarize(cases):
    return {"cases": len(cases), "unique_states": len({case["canonical_state_id"] for case in cases}),
            "families": len({case["family_id"] for case in cases}),
            "games": len({case["game_id"] for case in cases if case["game_id"] is not None}),
            "categories": dict(sorted(Counter(case["category"] for case in cases).items())),
            "evidence": dict(sorted(Counter(case["evidence"] for case in cases).items())),
            "splits": {split: {"cases": sum(case["split"] == split for case in cases),
                                "families": len({case["family_id"] for case in cases if case["split"] == split})}
                       for split in ("train", "dev", "test")}}


def mine_replays(paths, *, count=250, candidate_count=200, seed=4173, source_usage="unknown", defenses=24,
                 diagnostics=None):
    verified = defense_cases(seed, min(defenses, count)) if defenses else []
    potential, candidates = [], []
    seen = {case["canonical_state_id"] for case in verified}
    excluded_games, excluded_hashes, excluded_families = set(), set(), set()
    for path in sorted(map(Path, paths)):
        samples, moves, indices = _read_game(path)
        if not samples:
            if diagnostics is not None:
                diagnostics.setdefault("unscored_empty_replays", []).append(str(path))
            continue
        identities = [canonical_state_id(e.BaseBoard.deserialize(sample.state)) for sample in samples]
        family = "game:" + hashlib.sha256(json.dumps(identities, separators=(",", ":")).encode("ascii")).hexdigest()[:20]
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        excluded_games.update(sample.game_id for sample in samples if sample.game_id)
        excluded_hashes.add(digest)
        excluded_families.add(family)
        for sample, index, identity in zip(samples, indices, identities):
            if identity in seen:
                continue
            seen.add(identity)
            board = e.BaseBoard.deserialize(sample.state)
            if cpu._terminal(board) is not None:
                continue
            source = {"kind": "game_replay", "path": str(path), "sha256": digest,
                      "original_game_id": sample.game_id,
                      "action_index": index, "source_usage": source_usage,
                      "reachability": "trajectory_verified_from_first_retained_state",
                      "observed_game_outcome_for_mover": sample.outcome,
                      "outcome_is_not_objective_label": True}
            solution = _capture_witness(board, moves[index:index + 2])
            if solution:
                objective, witness, category = solution
                case = _case(board, family, family, source, objective, witness, category, seed)
                verify_case(case)
                potential.append(case)
            else:
                evidence = "teacher_approximate" if sample.source.startswith("ghq-cpu-") else "unverified"
                case = _case(board, family, family, source, None, None, "positional_candidate", seed, evidence)
                case["proposal"] = {"action": sample.actions[max(range(len(sample.actions)), key=sample.policy.__getitem__)],
                                    "source": sample.source, "is_ground_truth": False}
                candidates.append(case)
    # Balance across families before taking a second state from any game. A
    # deterministic shuffle avoids selecting only early/easy replay positions.
    rng = random.Random(seed)
    rng.shuffle(potential)
    groups = {}
    for case in potential:
        groups.setdefault(case["family_id"], []).append(case)
    while groups and len(verified) < count:
        for family in list(groups):
            verified.append(groups[family].pop())
            if not groups[family]:
                del groups[family]
            if len(verified) >= count:
                break
    rng.shuffle(candidates)
    candidates = candidates[:candidate_count]
    excluded_families.update(case["family_id"] for case in verified)
    return verified, candidates, {"game_ids": sorted(excluded_games), "replay_sha256": sorted(excluded_hashes),
                                  "canonical_state_ids": sorted(case["canonical_state_id"] for case in verified),
                                  "family_ids": sorted(excluded_families)}


def write_suite(path, cases, **metadata):
    path = Path(path)
    if path.exists():
        raise ValueError(f"Refusing to overwrite frozen suite {path}")
    for case in cases:
        verify_case(case)
    if len({case["canonical_state_id"] for case in cases}) != len(cases):
        raise ValueError("Duplicate or mirrored suite state")
    family_splits, game_splits = {}, {}
    for case in cases:
        for key, mapping in ((case["family_id"], family_splits), (case["game_id"], game_splits)):
            if key is not None and mapping.setdefault(key, case["split"]) != case["split"]:
                raise ValueError("Family/game leaks across suite splits")
    header = {"version": SUITE_VERSION, "rules_sha256": rules_hash(), "summary": summarize(cases), **metadata}
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as output:
        output.write(json.dumps({"header": header}, sort_keys=True) + "\n")
        for case in cases:
            output.write(json.dumps(case, sort_keys=True) + "\n")
    return header


def read_suite(path, split="all"):
    if split not in ("all", "train", "dev", "test"):
        raise ValueError("Invalid split selection")
    with Path(path).open() as source:
        header = json.loads(next(source))["header"]
        cases = [json.loads(line) for line in source if line.strip()]
    if header.get("version") != SUITE_VERSION or header.get("rules_sha256") != rules_hash():
        raise ValueError("Frozen suite schema/rules mismatch")
    if header["summary"] != summarize(cases):
        raise ValueError("Frozen suite counts mismatch")
    if len(cases) != len({case["canonical_state_id"] for case in cases}):
        raise ValueError("Duplicate/mirrored frozen case")
    groups = {}
    for case in cases:
        verify_case(case)
        for group in (case["family_id"], case["game_id"]):
            if group is not None and groups.setdefault(group, case["split"]) != case["split"]:
                raise ValueError("Frozen suite family/game split leakage")
    return header, [case for case in cases if split == "all" or case["split"] == split]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replays", type=Path, action="append", required=True, help="Game JSONL file or directory; repeatable")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--count", type=int, default=250, help="Distinct states; mirrored runs are not additional cases")
    parser.add_argument("--candidate-count", type=int, default=200)
    parser.add_argument("--defenses", type=int, default=24)
    parser.add_argument("--seed", type=int, default=4173)
    parser.add_argument("--source-usage", choices=("unknown", "known_training", "held_out"), default="unknown")
    args = parser.parse_args()
    if args.count < 1 or min(args.candidate_count, args.defenses) < 0:
        parser.error("Invalid case counts")
    if args.output.exists():
        parser.error("Output exists; frozen collections require a new directory")
    paths = sorted({path for item in args.replays for path in (item.glob("*.jsonl") if item.is_dir() else [item])})
    if not paths:
        parser.error("No replay files found")
    diagnostics = {}
    cases, candidates, exclusions = mine_replays(paths, count=args.count, candidate_count=args.candidate_count,
                                                seed=args.seed, source_usage=args.source_usage, defenses=args.defenses,
                                                diagnostics=diagnostics)
    metadata = {"usage": "evaluation_only", "source_usage": args.source_usage,
                "seed": args.seed, "mirrors_are_independent_cases": False,
                "strategic_optimality_claim": False, "requested_cases": args.count,
                **({"diagnostics": diagnostics} if diagnostics else {})}
    write_suite(args.output / "cases.jsonl", cases, **metadata)
    write_suite(args.output / "candidates.jsonl", candidates, **metadata)
    digest = hashlib.sha256((args.output / "cases.jsonl").read_bytes()).hexdigest()
    manifest = {"version": SUITE_VERSION, "rules_sha256": rules_hash(), "cases_sha256": digest,
                **metadata, "summary": summarize(cases), "candidate_summary": summarize(candidates),
                "exclusions": exclusions,
                "splits": {split: {"case_ids": [case["id"] for case in cases if case["split"] == split],
                                    "family_ids": sorted({case["family_id"] for case in cases if case["split"] == split})}
                           for split in ("train", "dev", "test")}}
    (args.output / "reserved.manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    for split, selection in manifest["splits"].items():
        (args.output / f"{split}.manifest.json").write_text(json.dumps({"cases_sha256": digest, "split": split,
                                                                       **selection}, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "summary": manifest["summary"], "requested": args.count,
                      "shortfall": max(0, args.count - len(cases)), "reserved_case_states": len(exclusions["canonical_state_ids"]),
                      "diagnostics": diagnostics}))


if __name__ == "__main__":
    main()
