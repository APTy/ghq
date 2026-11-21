"""Time-bounded, selective GHQ analysis, shared by the bot and analysis UI.

Scores are in infantry units, positive for Red. Search depth counts *turns*,
not actions: all three actions belong to the same maximizing/minimizing player.
This is a beam search, not a proof of the best move. Every legal action at an
expanded position is ranked, but only diverse promising continuations survive.
No training, third-party packages, or browser APIs are required.
"""

import json
import time
from dataclasses import dataclass

from engine import (
    BaseBoard, Move, RED, BLUE, HQ, INFANTRY, ARMORED_INFANTRY,
    AIRBORNE_INFANTRY, ARTILLERY, ARMORED_ARTILLERY, HEAVY_ARTILLERY,
    BB_ALL, BB_SQUARES, BB_REGULAR_MOVES, BB_ADJACENT_SQUARES,
    BB_ARMORED_MOVES, BB_FILE_A, BB_FILE_H, BB_RANK_1, BB_RANK_8,
    scan_forward, between, between_inclusive_end,
)

ENGINE_VERSION = "ghq-cpu-4"
WIN = 10000.0
popcount = int.bit_count
VALUES = {HQ: 0.0, INFANTRY: 1.0, ARMORED_INFANTRY: 2.0,
          AIRBORNE_INFANTRY: 4.0, ARTILLERY: 3.0,
          ARMORED_ARTILLERY: 4.0, HEAVY_ARTILLERY: 5.0}
RADIUS_TWO = [sum(BB_SQUARES[target] for target in range(64)
                  if max(abs(square % 8 - target % 8), abs(square // 8 - target // 8)) <= 2)
              for square in range(64)]
RADIUS_THREE = [sum(BB_SQUARES[target] for target in range(64)
                    if max(abs(square % 8 - target % 8), abs(square // 8 - target // 8)) <= 3)
                for square in range(64)]


def _make_fire_lines():
    # Use the rules engine's geometry once, rather than recomputing each
    # battery's ray in every evaluation. Occupied squares do not block fire.
    board = BaseBoard()
    result = []
    for reach in (2, 3):
        squares = []
        for square in range(64):
            lines = []
            for orientation in range(8):
                target = board.get_bombardment_target(square, orientation, reach)
                lines.append(0 if target is None else between_inclusive_end(square, target))
            squares.append(lines)
        result.append(squares)
    return result


FIRE_LINES = _make_fire_lines()


def _credible_fire(board, capture_threats):
    """Reliable pressure, individual rays, and overlap; legal fire is unchanged.

    An immediately capturable checking gun cannot claim the full value of a
    future HQ/material attack. Another battery covering the same square keeps
    that threat credible. This only affects estimates, never legal movement or
    bombardments already due before the defender may choose an action.
    """
    reliable, covered, overlap = [0, 0], [0, 0], [0, 0]
    lines = {}
    artillery = board.artillery | board.armored_artillery | board.heavy_artillery
    for color in (RED, BLUE):
        vulnerable = capture_threats[not color]
        for square in scan_forward(artillery & board.occupied_co[color]):
            line = FIRE_LINES[bool(BB_SQUARES[square] & board.heavy_artillery)][square][board.get_orientation(square)]
            lines[square] = line
            overlap[color] |= covered[color] & line
            covered[color] |= line
            if not BB_SQUARES[square] & vulnerable:
                reliable[color] |= line
    return reliable, lines, overlap


def _safe_destinations(board, square, color):
    """Movement geometry without move generation or changing board.turn."""
    bit = BB_SQUARES[square]
    infantry = bool(bit & (board.infantry | board.armored_infantry | board.airborne_infantry))
    engaged = infantry and bool(bit & board.adjacent_infantry_squares_co[not color])
    enemy_adjacent = board.adjacent_infantry_squares_co[not color]
    empty = ~board.occupied & ~board.bombarded_co[not color] & BB_ALL
    home = BB_RANK_1 if color == RED else BB_RANK_8
    if bit & board.airborne_infantry and bit & home:
        return empty & (~enemy_adjacent if engaged else BB_ALL)
    moves = BB_REGULAR_MOVES[square]
    if bit & (board.armored_infantry | board.armored_artillery):
        impassable = board.occupied | board.bombarded_co[not color]
        if engaged:
            impassable |= enemy_adjacent
            moves &= ~enemy_adjacent
        for target in scan_forward(BB_ARMORED_MOVES[square] & ~BB_REGULAR_MOVES[square] & empty):
            if not between(square, target) & impassable:
                moves |= BB_SQUARES[target]
    elif engaged:
        moves &= ~enemy_adjacent
    return moves & empty


def _terminal(board):
    if not board.hq & board.occupied_co[RED]:
        return {"winner": "BLUE", "reason": "hq-capture"}
    if not board.hq & board.occupied_co[BLUE]:
        return {"winner": "RED", "reason": "hq-capture"}
    if board.turn_moves == 0:
        # Move.__bool__ deliberately makes Skip false. The rules engine ends
        # a turn-start position with no non-Skip action, even if both HQs are
        # still present. Prove an action exists cheaply in normal positions;
        # consult the authoritative outcome on a copy in rare immobile ones.
        # outcome() can initialize free-capture bookkeeping, hence the copy.
        available_pieces = board.occupied_co[board.turn] & ~board.turn_pieces
        has_rotation = bool(available_pieces & (board.artillery | board.armored_artillery | board.heavy_artillery))
        hq = board.hq & available_pieces
        empty_safe = ~board.occupied & ~board.bombarded_co[not board.turn] & BB_ALL
        has_hq_move = bool(hq and BB_REGULAR_MOVES[hq.bit_length() - 1] & empty_safe)
        home = BB_RANK_1 if board.turn == RED else BB_RANK_8
        has_deployment = bool(empty_safe & home and any(board.reserves[board.turn]))
        if not (has_rotation or has_hq_move or has_deployment):
            probe = board.copy()
            probe.move_stack = []
            outcome = probe.outcome()
            if outcome is not None:
                winner = None if outcome.winner is None else "RED" if outcome.winner == RED else "BLUE"
                return {"winner": winner, "reason": outcome.termination}
    if board.did_offer_draw and board.did_accept_draw:
        return {"winner": None, "reason": "double-skip"}
    return None


def _terminal_score(terminal):
    return WIN if terminal["winner"] == "RED" else -WIN if terminal["winner"] == "BLUE" else 0.0


def state_key(board):
    """All serialized rule state, without compression or irrelevant history.

    In particular reserves, actions used, pending free captures, and draw offers
    cannot be omitted: identical piece placement need not have identical moves.
    """
    return (
        board.occupied, board.infantry, board.armored_infantry,
        board.airborne_infantry, board.artillery, board.armored_artillery,
        board.heavy_artillery, board.hq, *board.occupied_co,
        *board.bombarded_co, *board.adjacent_infantry_squares_co,
        board.orientation_bit0, board.orientation_bit1, board.orientation_bit2,
        board.turn_pieces, board.free_capture_clusters,
        board.free_capture_enemies, board.free_capture_num_allowed,
        board.turn, board.turn_moves, board.turn_auto_moves,
        *board.reserves[RED].to_ints(), *board.reserves[BLUE].to_ints(),
        bool(board.did_offer_draw), bool(board.did_accept_draw),
    )


def _artillery_capture_threats(board, attacker):
    """Squares next to an unengaged legal infantry landing, including paradrops.

    Landing beside a battery with no defending infantry is a one-action
    capture. Checking this geometry is much cheaper than generating every
    artillery orientation and executing every enemy move inside evaluation.
    """
    enemy = not attacker
    infantry = board.infantry | board.armored_infantry | board.airborne_infantry
    friendly = board.occupied_co[attacker] & infantry
    if board.turn == attacker:
        friendly &= ~board.turn_pieces
    # Unengaged landings cannot be adjacent to defending infantry. Enemy
    # bombardment also prevents entry, including a battery firing at its front.
    available = (~board.occupied & ~board.bombarded_co[enemy] &
                 ~board.adjacent_infantry_squares_co[enemy] & BB_ALL)
    back_rank = BB_RANK_1 if attacker == RED else BB_RANK_8
    landings = 0
    for square in scan_forward(friendly):
        bit = BB_SQUARES[square]
        if bit & board.airborne_infantry and bit & back_rank:
            landings |= available
            break
        moves = BB_REGULAR_MOVES[square]
        if bit & board.armored_infantry:
            impassable = board.occupied | board.bombarded_co[enemy]
            if bit & board.adjacent_infantry_squares_co[enemy]:
                # Engaged armor must also avoid enemy-adjacent intermediate
                # squares; a safe destination alone does not make the path legal.
                impassable |= board.adjacent_infantry_squares_co[enemy]
            for destination in scan_forward(BB_ARMORED_MOVES[square] & ~moves & available):
                if not between(square, destination) & impassable:
                    moves |= BB_SQUARES[destination]
        landings |= moves & available
    if any(board.reserves[attacker].get_count(kind) for kind in (INFANTRY, ARMORED_INFANTRY, AIRBORNE_INFANTRY)):
        landings |= available & back_rank
    return ((landings << 8) | (landings >> 8) |
            ((landings & ~BB_FILE_H) << 1) | ((landings & ~BB_FILE_A) >> 1)) & BB_ALL


def evaluate(board):
    """Pure, color-symmetric static evaluation; never push or generate moves.

    Reserves already have material value. Deployment earns a modest tempo
    premium, not the value of an entirely new piece. Mobility and infantry
    pressure are inexpensive approximations; search determines actual captures.
    """
    terminal = _terminal(board)
    if terminal:
        return _terminal_score(terminal)
    infantry = board.infantry | board.armored_infantry | board.airborne_infantry
    masks = ((INFANTRY, board.infantry), (ARMORED_INFANTRY, board.armored_infantry),
             (AIRBORNE_INFANTRY, board.airborne_infantry), (ARTILLERY, board.artillery),
             (ARMORED_ARTILLERY, board.armored_artillery), (HEAVY_ARTILLERY, board.heavy_artillery))
    scores = [0.0, 0.0]
    # Different tactical reasons can threaten the same unit. Track the largest
    # exposure per square rather than repeatedly subtracting its full value.
    exposure = [0.0] * 64
    capture_threats = [_artillery_capture_threats(board, color) for color in (RED, BLUE)]
    reliable_fire, fire_lines, overlapping_fire = _credible_fire(board, capture_threats)
    for color in (RED, BLUE):
        ours, theirs = board.occupied_co[color], board.occupied_co[not color]
        friendly_infantry = ours & infantry
        enemy_infantry = theirs & infantry
        danger = board.bombarded_co[not color]
        enemy_adjacent = board.adjacent_infantry_squares_co[not color]
        empty_safe = ~board.occupied & ~danger & BB_ALL
        artillery_threats = capture_threats[not color]
        enemy_reserve_infantry = sum(board.reserves[not color].get_count(kind)
                                    for kind in (INFANTRY, ARMORED_INFANTRY, AIRBORNE_INFANTRY))
        enemy_reinforcement_rank = BB_RANK_8 if color == RED else BB_RANK_1
        enemy_reinforcement_landings = (enemy_reinforcement_rank & ~board.occupied &
                                       ~board.bombarded_co[color])
        enemy_hq = (board.hq & theirs).bit_length() - 1
        hq_square = (board.hq & ours).bit_length() - 1
        score = 0.0
        for piece_type, mask in masks:
            value = VALUES[piece_type]
            score += 0.85 * value * board.reserves[color].get_count(piece_type)
            for square in scan_forward(mask & ours):
                bit = BB_SQUARES[square]
                rank, file = divmod(square, 8)
                advance = rank if color == RED else 7 - rank
                center = 3.5 - abs(3.5 - file)
                score += value
                exits = _safe_destinations(board, square, color)
                exit_count = popcount(exits)
                if bit & infantry:
                    # Nearby infantry can coordinate a capture during this turn.
                    score += 0.035 * advance + 0.025 * center
                    score += 0.045 * popcount(BB_REGULAR_MOVES[square] & friendly_infantry)
                    distance = abs(file - (enemy_hq % 8)) + abs(rank - (enemy_hq // 8))
                    score += 0.045 * max(0, 7 - distance)
                    score += 0.009 * min(12, exit_count)
                    if bit & enemy_adjacent:
                        score -= 0.07
                else:
                    score += 0.06 * advance + 0.02 * center
                    score += 0.012 * popcount(BB_REGULAR_MOVES[square] & empty_safe)
                    if bit & artillery_threats:
                        can_escape = color == board.turn and not (bit & board.turn_pieces)
                        exposure[square] = value * (0.45 if can_escape else 0.85)
                # Deep pieces need a force that can support them, especially
                # near enemy units and the enemy reinforcement rank. This
                # applies to every combat unit, not only airborne infantry.
                # Existing positional rewards remain for supported attacks.
                if advance >= 4:
                    support = friendly_infantry & ~bit
                    close_support = bool(BB_REGULAR_MOVES[square] & support)
                    reachable_support = bool(RADIUS_TWO[square] & support or
                                             RADIUS_THREE[square] & board.armored_infantry & support)
                    other_fire = ((board.bombarded_co[color] & ~fire_lines.get(square, 0)) |
                                  overlapping_fire[color])
                    artillery_support = popcount(BB_ADJACENT_SQUARES[square] & other_fire) >= 2
                    support_factor = 0.0 if close_support else 0.25 if reachable_support else 0.5 if artillery_support else 1.0
                    nearby_enemies = popcount(RADIUS_TWO[square] & theirs)
                    isolation = value * 0.14 * (advance - 3) * support_factor
                    isolation *= 1.0 + 0.12 * min(3, nearby_enemies)
                    reserve_approaches = popcount(BB_ADJACENT_SQUARES[square] & enemy_reinforcement_landings)
                    if enemy_reserve_infantry and reserve_approaches:
                        isolation *= 1.0 + 0.15 * min(2, enemy_reserve_infantry, reserve_approaches)
                    if exit_count <= 2:
                        isolation *= 1.3
                    exposure[square] = max(exposure[square], min(0.8 * value, isolation))
                if bit & danger:
                    # An already-used unit cannot escape this turn without
                    # another unit removing or redirecting the enemy battery.
                    locked = color == board.turn and bool(bit & board.turn_pieces)
                    immediate = board.turn != color and board.turn_moves == 0
                    # The opponent can coordinate batteries to remove all
                    # escape squares. An unused unit is not necessarily able
                    # to escape; intermediate reductions in exits also help
                    # retain the first move of a multi-battery trapping plan.
                    flight_risk = 0.9 if exit_count == 0 else 0.7 if exit_count == 1 else 0.5 if exit_count == 2 else 0.3
                    pressure = 1.0 if bit & reliable_fire[not color] else 0.2
                    risk = 0.9 if immediate else pressure * (max(0.7, flight_risk) if locked else flight_risk)
                    exposure[square] = max(exposure[square], value * risk)
        coverage = board.bombarded_co[color]
        for square in scan_forward(coverage):
            rank, file = divmod(square, 8)
            advance = rank if color == RED else 7 - rank
            score += 0.02 + 0.018 * advance + 0.008 * (3.5 - abs(3.5 - file))
        score += 0.1 * popcount(coverage & theirs & ~board.hq)
        score += 0.035 * popcount(coverage & BB_REGULAR_MOVES[enemy_hq])
        hq_bit = BB_SQUARES[hq_square]
        exits = BB_REGULAR_MOVES[hq_square] & empty_safe
        safe_exits = exits & ~enemy_adjacent
        attackers = popcount(BB_ADJACENT_SQUARES[hq_square] & enemy_infantry)
        nearby_attackers = popcount(BB_REGULAR_MOVES[hq_square] & enemy_infantry)
        defenders = popcount(BB_REGULAR_MOVES[hq_square] & friendly_infantry)
        score += 0.07 * min(4, popcount(safe_exits))
        score += 0.09 * min(3, defenders)
        score -= 0.65 * attackers + 0.14 * nearby_attackers
        if attackers >= 2:
            score -= 4.0
        if not safe_exits:
            score -= 0.5 + 0.3 * nearby_attackers
        hq_advance = hq_square // 8 if color == RED else 7 - hq_square // 8
        if hq_advance >= 4 and not (RADIUS_TWO[hq_square] & friendly_infantry):
            score -= 0.3 * (hq_advance - 3) + 0.25 * nearby_attackers
        if hq_bit & danger:
            immediate = board.turn != color and board.turn_moves == 0
            locked = color == board.turn and bool(hq_bit & board.turn_pieces)
            pressure = 1.0 if hq_bit & reliable_fire[not color] else 0.2
            score -= 100.0 if immediate else pressure * (35.0 if locked else 8.0)
        scores[color] = score
    # Infantry superiority can make a unit capturable for free next turn.
    # Rewarding an advanced paratrooper before accounting for that fact used to
    # fill the beam with different ways of abandoning the very same unit. This
    # read-only cluster query uses the rules engine's engagement accounting;
    # it neither generates moves nor changes pending-capture bookkeeping.
    for cluster in board._find_adjacency_clusters(board.occupied_co, infantry):
        for attacker in (RED, BLUE):
            victims, allowance = board._find_free_captures_for_cluster(board.occupied_co, cluster, attacker)
            if not victims:
                continue
            count = popcount(allowance)
            values = {}
            for square in scan_forward(victims & ~board.hq):
                piece_type = board.piece_type_at(square)
                values.setdefault(VALUES.get(piece_type, 0.0), []).append(square)
            for value in sorted(values, reverse=True):
                squares = values[value]
                allocation = min(count, len(squares))
                # Equal victims share the estimate. Choosing a square-index
                # tie winner would break color symmetry when other threats
                # already cover one of the alternatives.
                risk = 0.9 * value * allocation / len(squares)
                for square in squares:
                    # A future free capture is not already won material: the
                    # defender may move this victim before the attacker's turn.
                    # Keep full weight for captures due now and locked victims.
                    can_evade = (board.turn != attacker and
                                 not BB_SQUARES[square] & board.turn_pieces and
                                 bool(_safe_destinations(board, square, not attacker)))
                    exposure[square] = max(exposure[square], risk * (0.5 if can_evade else 1.0))
                count -= allocation
                if count <= 0:
                    break
    for color in (RED, BLUE):
        scores[color] -= sum(exposure[square] for square in scan_forward(board.occupied_co[color] & ~board.hq))
    return scores[RED] - scores[BLUE]


def _copy(board):
    result = board.copy()
    result.move_stack = []
    return result


@dataclass
class _Plan:
    board: BaseBoard
    moves: tuple
    score: float
    family: tuple = ()
    first_family: tuple = ()
    captures: tuple = ()


class _Timeout(Exception):
    pass


def _family(move):
    # Do not let eight artillery orientations monopolize the beam.
    return (move.name, move.from_square, move.to_square, move.unit_type)


def _select(plans, width, color, final=False):
    """Keep the best half, then diversify pieces/destinations and first moves.

    This is a safeguard against artillery branching, not exhaustive tactical
    coverage. Terminal winning plans are placed first regardless of diversity.
    """
    sign = 1 if color == RED else -1
    ordered = sorted(plans, key=lambda p: (-sign * p.score, p.moves))
    if len(ordered) <= width:
        return ordered
    selected = ordered[:max(1, width // 2)]
    selected_ids = {id(plan) for plan in selected}
    # A capture can dominate the beam in every action order. Keep one quiet
    # alternative so reply search can reject that tactical commitment instead
    # of merely selecting when to make the same losing capture.
    if any(plan.captures for plan in selected):
        quiet = next((plan for plan in ordered if not plan.captures and id(plan) not in selected_ids), None)
        if quiet is not None:
            selected.append(quiet)
            selected_ids.add(id(quiet))
    # Different preparations for the same capture must not crowd out a
    # different victim. Compare target sets independently of action order,
    # landing square, and the first development move. Use the remaining
    # diversity slots for distinct targets before repeating first decisions.
    targets = {frozenset(c[2] for c in plan.captures) for plan in selected}
    for plan in ordered:
        if len(selected) >= width:
            break
        target = frozenset(c[2] for c in plan.captures)
        if target and target not in targets:
            selected.append(plan)
            selected_ids.add(id(plan))
            targets.add(target)
    # Preserve actual alternative first decisions. Pairing first+last families
    # lets one attractive first action consume every slot merely by changing
    # the second/third action, so deeper search can never reject that action.
    groups = {p.first_family for p in selected}
    for plan in ordered:
        if len(selected) >= width:
            break
        if id(plan) in selected_ids:
            continue
        group = plan.first_family
        if group not in groups:
            selected.append(plan)
            selected_ids.add(id(plan))
            groups.add(group)
            if len(selected) == width:
                break
    if len(selected) < width:
        for plan in ordered:
            if id(plan) not in selected_ids:
                selected.append(plan)
                if len(selected) == width:
                    break
    return sorted(selected, key=lambda p: (-sign * p.score, p.moves))


class _Search:
    def __init__(self, board, time_ms, beam_width):
        self.started = time.perf_counter()
        self.deadline = self.started + max(0, time_ms) / 1000.0
        self.width = max(2, min(24, int(beam_width)))
        self.inner_width = max(2, min(4, self.width // 2))
        self.nodes = 0
        self.evals = {}
        self.turns = {}
        self.table = {}
        self.root = board
        self.root_completed = []

    def check(self):
        if time.perf_counter() >= self.deadline:
            raise _Timeout()

    def score(self, board, key=None):
        self.nodes += 1
        key = state_key(board) if key is None else key
        if key not in self.evals:
            self.evals[key] = evaluate(board)
        return self.evals[key]

    def fallback(self, board):
        """Always legal, complete emergency plan, even with a zero budget.

        Only mandatory captures and an immediate HQ win precede Skip. Normal
        analysis replaces this in the first completed turn-search iteration.
        """
        board = _copy(board)
        color = board.turn
        moves = []
        for _ in range(67):  # Each automatic capture removes a piece.
            if _terminal(board) or board.turn != color:
                break
            legal = list(board.generate_legal_moves())
            if not legal:
                break
            winning = next((m for m in legal if m.capture_preference is not None
                            and board.hq & BB_SQUARES[m.capture_preference]), None)
            if winning:
                move = winning
            elif legal[0].name == "AutoCapture":
                move = max(legal, key=lambda m: (VALUES.get(board.piece_type_at(m.capture_preference), 0), m.uci()))
            else:
                move = next((m for m in legal if m.name == "Skip"), legal[0])
            board.push(move)
            moves.append(move.uci())
        return _Plan(board, tuple(moves), evaluate(board))

    def plans(self, board, root=False):
        self.check()
        width = self.width if root else self.inner_width
        key = (state_key(board), root, width)
        if key in self.turns:
            return self.turns[key]
        color = board.turn
        frontier = [_Plan(board, (), self.score(board))]
        finished = {}
        # Automatic captures use no action. The occupied-piece bound protects
        # against accidental non-progress while allowing every mandatory capture.
        for _ in range(popcount(board.occupied) + 4):
            expanded = {}
            for plan in frontier:
                self.check()
                legal = list(plan.board.generate_legal_moves())
                # Forcing captures are ordered first so an interrupted pass is
                # still likely to discover a terminal win.
                legal.sort(key=lambda move: (move.capture_preference is None,
                           -(WIN if move.capture_preference is not None and
                             plan.board.hq & BB_SQUARES[move.capture_preference]
                             else VALUES.get(plan.board.piece_type_at(move.capture_preference), 0)
                             if move.capture_preference is not None else 0), move.uci()))
                for move in legal:
                    self.check()
                    child = _copy(plan.board)
                    child.push(move)
                    child_key = state_key(child)
                    moves = plan.moves + (move.uci(),)
                    family = _family(move)
                    captures = plan.captures
                    if move.capture_preference is not None and move.name != "AutoCapture":
                        captures += ((move.from_square, move.to_square, move.capture_preference, move.unit_type),)
                    candidate = _Plan(child, moves, self.score(child, child_key), family,
                                      plan.first_family or (family if move.name != "AutoCapture" else ()), captures)
                    terminal = _terminal(child)
                    if terminal or child.turn != color:
                        # Mandatory captures happen before the opponent can
                        # choose an action. Rank completed plans *after* that
                        # settlement; pruning first can discard every safe
                        # alternative to a superficially attractive sacrifice.
                        if not terminal:
                            candidate.score = self.settle(child)[0]
                        previous = finished.get(child_key)
                        if previous is None or len(moves) < len(previous.moves):
                            finished[child_key] = candidate
                        if root:
                            self.root_completed = list(finished.values())
                        if terminal and terminal["winner"] == ("RED" if color == RED else "BLUE"):
                            # A same-player continuation already won. No reply
                            # exists, and later actions must not be appended.
                            self.turns[key] = [candidate]
                            return [candidate]
                    elif child_key not in expanded:
                        expanded[child_key] = candidate
            if not expanded:
                break
            frontier = _select(list(expanded.values()), width, color)
        result = _select(list(finished.values()), width, color, final=True)
        if not result:
            result = [self.fallback(board)]
        self.turns[key] = result
        return result

    def settle(self, board):
        """Extend the horizon through mandatory captures, without skipping.

        Bombardments all happen. For free-capture alternatives choose the best
        resulting static position; this is explicitly selective, like the beam.
        """
        board = _copy(board)
        moves = []
        for _ in range(popcount(board.occupied) + 1):
            self.check()
            if _terminal(board) or board.turn_moves != 0:
                break
            # Avoid generating hundreds of normal actions just to learn that
            # there is no automatic capture at the leaf.
            automatic = list(board._find_bombardment_moves())
            if not automatic:
                automatic = list(board._generate_free_captures(board.turn))
            if not automatic:
                break
            sign = 1 if board.turn == RED else -1
            best = None
            for move in automatic:
                self.check()
                child = _copy(board)
                child.push(move)
                candidate = (sign * self.score(child), move.uci(), child)
                if best is None or candidate[:2] > best[:2]:
                    best = candidate
            moves.append(best[1])
            board = best[2]
        return self.score(board), tuple(moves)

    def minimax(self, board, depth, alpha=-float("inf"), beta=float("inf")):
        self.check()
        terminal = _terminal(board)
        if terminal:
            return _terminal_score(terminal), ()
        if depth == 0:
            return self.settle(board)
        key = (state_key(board), depth, self.inner_width)
        entry = self.table.get(key)
        if entry is not None:
            return entry
        maximizing = board.turn == RED
        original_alpha, original_beta = alpha, beta
        best_score = -float("inf") if maximizing else float("inf")
        best_moves = ()
        exact = True
        for plan in self.plans(board):
            value, continuation = self.minimax(plan.board, depth - 1, alpha, beta)
            if (maximizing and value > best_score) or (not maximizing and value < best_score):
                best_score, best_moves = value, plan.moves + continuation
            if maximizing:
                alpha = max(alpha, best_score)
            else:
                beta = min(beta, best_score)
            if beta <= alpha:
                exact = False
                break
        # Bounds are deliberately not cached as exact multi-PV evaluations.
        if exact and original_alpha < best_score < original_beta:
            self.table[key] = (best_score, best_moves)
        return best_score, best_moves


def analyze_json(serialized, time_ms=1200, max_depth=3, beam_width=8,
                 multi_pv=3, on_update=None):
    """Return JSON with legal root plans and progressively deeper analysis.

    Callbacks receive the same JSON schema after each completed iteration.
    An initial narrow pass prioritizes reaching the opponent's reply; widening
    can publish a second result at the same depth before searching deeper.
    timedOut means the requested next iteration exceeded the time budget. The
    returned depth and lines always belong to a completed iteration (depth 0
    denotes an emergency, still legal plan). No win probability is implied.
    """
    board = BaseBoard.deserialize(serialized)
    search = _Search(board, float(time_ms), beam_width)
    terminal = _terminal(board)
    fallback = search.fallback(board) if terminal is None else _Plan(board, (), _terminal_score(terminal))
    lines = [{"moves": list(fallback.moves), "turnMoves": list(fallback.moves), "score": fallback.score}]
    depth = 0
    timed_out = False
    count = max(1, min(10, int(multi_pv)))

    def result():
        sign = 1 if board.turn == RED else -1
        shown = sorted(lines[:count], key=lambda line: (-sign * line["score"], line["turnMoves"]))
        best = shown[0] if shown else {"score": evaluate(board), "turnMoves": []}
        return json.dumps({
            "engineVersion": ENGINE_VERSION, "state": serialized,
            "score": round(best["score"], 4),
            "bestMove": best["turnMoves"][0] if best["turnMoves"] else None,
            "turnMoves": best["turnMoves"], "lines": shown,
            "depth": depth, "nodes": search.nodes,
            "elapsedMs": round((time.perf_counter() - search.started) * 1000, 1),
            "selective": True, "terminal": terminal, "timedOut": timed_out,
        })

    if terminal:
        lines = []
        return result()
    try:
        requested_width = search.width
        requested_depth = max(1, min(8, int(max_depth)))
        initial_width = min(4, requested_width) if requested_depth >= 2 else requested_width
        iterations = [(1, initial_width)]
        if requested_depth >= 2:
            iterations.append((2, initial_width))
            if initial_width != requested_width:
                iterations.append((2, requested_width))
        iterations.extend((target_depth, requested_width) for target_depth in range(3, requested_depth + 1))
        previous_scores = {}
        for target_depth, width in iterations:
            search.width = width
            search.inner_width = max(2, min(4, width // 2))
            root_plans = search.plans(board, root=True)
            completed = []
            sign = 1 if board.turn == RED else -1
            root_plans.sort(key=lambda plan: -sign * previous_scores.get(plan.moves, plan.score))
            for plan in root_plans:
                value, continuation = search.minimax(plan.board, target_depth - 1)
                completed.append({"moves": list(plan.moves + continuation),
                                  "turnMoves": list(plan.moves), "score": round(value, 4)})
            completed.sort(key=lambda line: (-sign * line["score"], line["turnMoves"]))
            # Prefer distinct first actions for a useful alternatives panel.
            distinct, duplicate, seen = [], [], set()
            for line in completed:
                first = line["turnMoves"][0] if line["turnMoves"] else None
                (duplicate if first in seen else distinct).append(line)
                seen.add(first)
            lines = distinct + duplicate
            previous_scores = {tuple(line["turnMoves"]): line["score"] for line in completed}
            depth = target_depth
            if callable(on_update):
                on_update(result())
            if abs(lines[0]["score"]) == WIN and sign * lines[0]["score"] > 0:
                break
    except _Timeout:
        timed_out = True
        if depth == 0 and search.root_completed:
            best = _select(search.root_completed, count, board.turn, final=True)
            lines = [{"moves": list(plan.moves), "turnMoves": list(plan.moves),
                      "score": round(plan.score, 4)} for plan in best]
    return result()
