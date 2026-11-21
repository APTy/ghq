"""Policy/value MCTS over exact GHQ actions, including same-player actions.

Evaluator.predict(board, legal_moves) returns aligned probabilities and a
value in [-1, 1] from board.turn's perspective. Rules states stay in their
original coordinates; any model canonicalization belongs in the evaluator.
"""
from collections import OrderedDict
from dataclasses import dataclass, field
import math
import random
import time

from .rules import e, cpu, copy_board, terminal_value


@dataclass
class SearchResult:
    actions: list[str]
    policy: list[float]
    value: float
    current_player: bool
    visits: list[int]
    simulations: int
    nodes: int
    cache_size: int
    elapsed_ms: float
    terminal: bool
    timed_out: bool


@dataclass
class TurnPlan:
    moves: list[str]
    complete: bool
    current_player: bool
    value: float
    searches: list[SearchResult]
    elapsed_ms: float
    timed_out: bool
    terminal: bool


@dataclass
class _Node:
    key: tuple
    player: bool
    moves: list
    priors: list[float]
    value: float
    terminal: bool = False
    winning: tuple = ()
    visits: list[int] = field(default_factory=list)
    totals: list[float] = field(default_factory=list)
    total_visits: int = 0


class _Deadline(Exception):
    pass


class MCTS:
    def __init__(self, evaluator, simulations=128, cpuct=1.5, seed=0,
                 max_nodes=2048, max_depth=256):
        self.evaluator = evaluator
        self.simulations = max(0, int(simulations))
        self.cpuct = float(cpuct)
        self.rng = random.Random(seed)
        self.max_nodes = max(1, int(max_nodes))
        self.max_depth = max(1, int(max_depth))
        self._cache = OrderedDict()
        self._active_root = None

    def clear(self):
        """Discard cached predictions/statistics after changing model weights."""
        self._cache.clear()

    @staticmethod
    def _check(deadline):
        if deadline is not None and time.perf_counter() >= deadline:
            raise _Deadline()

    def _cached(self, key):
        node = self._cache.get(key)
        if node is not None:
            self._cache.move_to_end(key)
        return node

    def _remember(self, node):
        self._cache[node.key] = node
        self._cache.move_to_end(node.key)
        # Keep the active root available while bounding all retained nodes.
        if self._active_root in self._cache:
            self._cache.move_to_end(self._active_root)
        while len(self._cache) > self.max_nodes:
            self._cache.popitem(last=False)

    @staticmethod
    def _winning(board, moves):
        enemy_hq = board.hq & board.occupied_co[not board.turn]
        return tuple(i for i, move in enumerate(moves)
                     if move.capture_preference is not None and
                     e.BB_SQUARES[move.capture_preference] & enemy_hq)

    def _node(self, board, deadline, stats):
        key = cpu.state_key(board)
        cached = self._cached(key)
        if cached is not None:
            return cached, False
        value = terminal_value(board)
        if value is not None:
            node = _Node(key, board.turn, [], [], value, terminal=True)
        else:
            self._check(deadline)
            moves = sorted(board.generate_legal_moves(), key=lambda move: move.uci())
            # Legal move generation initializes pending free-capture state.
            # Cache the initialized complete state, not a stale pre-query key.
            key = cpu.state_key(board)
            cached = self._cached(key)
            if cached is not None:
                return cached, False
            if not moves:
                raise RuntimeError("Nonterminal GHQ state has no legal action")
            self._check(deadline)
            priors, value = self.evaluator.predict(copy_board(board), moves)
            priors = [float(p) for p in priors]
            value = float(value)
            if len(priors) != len(moves):
                raise ValueError("Evaluator probabilities must align with every legal move")
            if any(not math.isfinite(p) or p < 0 for p in priors) or not math.isfinite(value):
                raise ValueError("Evaluator returned invalid probabilities or value")
            total = sum(priors)
            priors = [p / total for p in priors] if total else [1.0 / len(moves)] * len(moves)
            winning = self._winning(board, moves)
            node = _Node(key, board.turn, moves, priors,
                         1.0 if winning else max(-1.0, min(1.0, value)),
                         winning=winning, visits=[0] * len(moves), totals=[0.0] * len(moves))
        self._remember(node)
        stats["nodes"] += 1
        return node, True

    def _select(self, node, priors):
        candidates = node.winning or range(len(node.moves))
        scale = self.cpuct * math.sqrt(1 + node.total_visits)
        return max(candidates, key=lambda i: (
            (node.totals[i] / node.visits[i] if node.visits[i] else 0.0) +
            scale * priors[i] / (1 + node.visits[i]), -i))

    def _simulate(self, original, root, root_priors, deadline, stats):
        board = copy_board(original)
        node = root
        path, seen = [], {root.key}
        for _ in range(self.max_depth):
            self._check(deadline)
            index = self._select(node, root_priors if node is root else node.priors)
            board.push(node.moves[index])
            path.append((node, index, board.turn))
            node, fresh = self._node(board, deadline, stats)
            if node.terminal or fresh or node.key in seen:
                break
            seen.add(node.key)
        # Cycles/depth cutoffs bootstrap the network estimate, never invent a
        # draw. Only the authoritative terminal rules produce a draw target.
        value = node.value
        for parent, index, child_player in reversed(path):
            if parent.player != child_player:
                value = -value
            parent.visits[index] += 1
            parent.totals[index] += value
            parent.total_visits += 1

    def _emergency(self, board):
        moves = sorted(board.generate_legal_moves(), key=lambda move: move.uci())
        winning = self._winning(board, moves)
        if winning:
            index = winning[0]
        elif moves and moves[0].name == "AutoCapture":
            index = max(range(len(moves)), key=lambda i: cpu.VALUES.get(
                board.piece_type_at(moves[i].capture_preference), 0))
        else:
            index = next((i for i, move in enumerate(moves) if move.name == "Skip"), 0)
        priors = [float(i == index) for i in range(len(moves))]
        return _Node(cpu.state_key(board), board.turn, moves, priors,
                     1.0 if winning else 0.0, winning=winning,
                     visits=[0] * len(moves), totals=[0.0] * len(moves))

    def search(self, board, time_ms=None, add_noise=False):
        return self._search(board, time_ms, add_noise, self.simulations)

    def _search(self, original, time_ms, add_noise, simulations):
        started = time.perf_counter()
        deadline = None if time_ms is None else started + max(0.0, float(time_ms)) / 1000
        board = copy_board(original)
        stats = {"nodes": 0}
        completed, timed_out = 0, False
        root = None
        self._active_root = None
        try:
            root, _ = self._node(board, deadline, stats)
            self._active_root = root.key
            priors = root.priors
            if add_noise and priors:
                noise = [self.rng.gammavariate(0.3, 1.0) for _ in priors]
                total = sum(noise)
                priors = [0.75 * p + 0.25 * n / total for p, n in zip(priors, noise)]
            if not root.terminal:
                for _ in range(simulations):
                    self._simulate(board, root, priors, deadline, stats)
                    completed += 1
        except _Deadline:
            timed_out = True
            if root is None:
                root = self._emergency(board)
        finally:
            self._active_root = None
        total = sum(root.visits)
        policy = ([n / total for n in root.visits] if total else list(root.priors))
        if not total and root.winning:
            policy = [float(i in root.winning) / len(root.winning) for i in range(len(root.moves))]
        value = sum(root.totals) / total if total else root.value
        return SearchResult(
            [move.uci() for move in root.moves], policy, value, root.player,
            list(root.visits), completed, stats["nodes"], len(self._cache),
            (time.perf_counter() - started) * 1000, root.terminal, timed_out)

    def plan_turn(self, original, time_ms=2400, simulations_per_action=None):
        """Plan the remaining turn under one shared wall-clock deadline.

        Search receives a share of the remaining budget at each decision and
        reuses existing subtree statistics. After expiry cached choices or a
        legal emergency action complete the turn. An evaluator call already
        in flight cannot be preempted, so its latency bounds deadline overshoot.
        """
        started = time.perf_counter()
        deadline = started + max(0.0, float(time_ms)) / 1000
        board = copy_board(original)
        color, moves, searches = board.turn, [], []
        value, timed_out = 0.0, False
        simulations = self.simulations if simulations_per_action is None else max(0, int(simulations_per_action))
        for _ in range(board.occupied.bit_count() + 4):
            outcome = terminal_value(board)
            if outcome is not None or board.turn != color:
                if not searches and outcome is not None:
                    value = outcome if board.turn == color else -outcome
                break
            legal = list(board.generate_legal_moves())
            if len(legal) == 1 and legal[0].name == "AutoCapture":
                move = legal[0]
            else:
                remaining_ms = max(0.0, (deadline - time.perf_counter()) * 1000)
                timed_out |= remaining_ms == 0
                decisions = max(1, 3 - board.turn_moves)
                if legal and legal[0].name == "AutoCapture":
                    decisions += 1
                result = self._search(board, remaining_ms / decisions, False, simulations)
                searches.append(result)
                if len(searches) == 1:
                    value = result.value
                if not result.actions:
                    break
                index = max(range(len(result.actions)), key=lambda i: (result.policy[i], -i))
                move = e.Move.from_uci(result.actions[index])
            board.push(move)
            moves.append(move.uci())
        terminal = terminal_value(board) is not None
        return TurnPlan(moves, terminal or board.turn != color, color, value, searches,
                        (time.perf_counter() - started) * 1000,
                        timed_out or time.perf_counter() >= deadline, terminal)
