"""Deterministic game curricula and persistent, leakage-safe replay partitions.

A family (a game and its continuations/variants) belongs to one partition.
Canonical state fingerprints are retained even after replay eviction; a state
seen in the other partition is omitted, never relabeled or silently reused.
"""
from dataclasses import replace
import hashlib
import math
import random

from .rules import e, cpu

CURRICULUM_VERSION = 1


def fingerprint(sample):
    board = e.BaseBoard.deserialize(sample.state)
    return hashlib.sha256(min(board.serialize(), board.mirror().serialize()).encode()).hexdigest()


class ReplayPartition:
    def __init__(self, seed, validation_fraction=0.1, *, train_states=(), validation_states=()):
        if not 0 <= validation_fraction < 1:
            raise ValueError("Validation fraction must be in [0, 1)")
        self.seed = seed
        self.validation_fraction = validation_fraction
        self.train_states = set(train_states)
        self.validation_states = set(validation_states)
        if self.train_states & self.validation_states:
            raise ValueError("Curriculum state contains overlapping training and validation positions")

    def split_for(self, family):
        if not family:
            raise ValueError("Curriculum samples require a game or family identity")
        digest = hashlib.sha256(f"{self.seed}:holdout:{family}".encode()).digest()
        return "validation" if int.from_bytes(digest[:8], "big") / 2**64 < self.validation_fraction else "train"

    def add_game(self, samples, *, family_id=None):
        training, validation, excluded = [], [], 0
        # All records of a game must share its family. Callers process completed
        # games in task order, so worker completion timing cannot change splits.
        families = {family_id or row.family_id or row.game_id for row in samples}
        if len(families) > 1:
            raise ValueError("A collected game cannot contain multiple holdout families")
        for row in samples:
            family = family_id or row.family_id or row.game_id
            row = replace(row, family_id=family)
            key = fingerprint(row)
            split = self.split_for(family)
            own, other = ((self.train_states, self.validation_states) if split == "train" else
                          (self.validation_states, self.train_states))
            if key in other:
                excluded += 1
                continue
            own.add(key)
            (training if split == "train" else validation).append(row)
        return training, validation, excluded

    def to_dict(self):
        return {"seed": self.seed, "validation_fraction": self.validation_fraction,
                "train_states": sorted(self.train_states), "validation_states": sorted(self.validation_states)}

    @classmethod
    def from_dict(cls, state):
        return cls(**state)


def retain_replay(samples, limit, *, teacher_fraction=0.2, hard_fraction=0.1, seed=0):
    """Reserve quotas when available, then fill remaining slots without duplication.

    Fractions are minimum shares, not oversampling weights. A scarce bucket is
    kept in full and unused capacity is available to every other example.
    Hard teacher rows count toward both quotas but occupy only one slot.
    Targets round up; if rounding makes them jointly impossible in a tiny
    buffer, the larger fractional remainder wins the extra slot (teacher wins
    ties). An available overlap is used before relaxing either rounded target.
    """
    if limit < 1 or min(teacher_fraction, hard_fraction) < 0 or teacher_fraction + hard_fraction > 1:
        raise ValueError("Invalid replay retention limit or quotas")
    if len(samples) <= limit:
        return list(samples)
    rng = random.Random(seed)
    hard = [i for i, row in enumerate(samples) if row.priority == "hard"]
    teacher = [i for i, row in enumerate(samples) if row.source.startswith("ghq-cpu")]
    teacher_set = set(teacher)
    overlap = [i for i in hard if i in teacher_set]
    teacher_raw, hard_raw = limit * teacher_fraction, limit * hard_fraction
    teacher_target = min(len(teacher), math.ceil(teacher_raw))
    hard_target = min(len(hard), math.ceil(hard_raw))
    shared_needed = max(0, teacher_target + hard_target - limit)
    if shared_needed > len(overlap):
        # The fractions sum to at most one, so separate ceilings can create
        # at most one excess slot. There is insufficient overlap to share it.
        if teacher_raw % 1 >= hard_raw % 1:
            hard_target -= 1
        else:
            teacher_target -= 1
        shared_needed = max(0, teacher_target + hard_target - limit)
    selected = set(rng.sample(overlap, shared_needed))
    selected.update(rng.sample([i for i in hard if i not in selected], hard_target - len(selected)))
    teacher_needed = max(0, teacher_target - len(selected & teacher_set))
    selected.update(rng.sample([i for i in teacher if i not in selected], teacher_needed))
    remainder = [i for i in range(len(samples)) if i not in selected]
    selected.update(rng.sample(remainder, limit - len(selected)))
    return [samples[i] for i in sorted(selected)]


def opponent_schedule(count, *, seed, cpu_fraction=0.4, history_fraction=0.2, start_index=0):
    """Return a slice of one seeded schedule, independent of collection size.

    At each assignment the most underrepresented opponent receives the next
    game. Colors alternate within each opponent category across collection and
    resume boundaries. The cursor counts scheduled attempts, not completed
    games: watchdogs can still bias the distribution of completed experience.
    """
    if (count < 0 or start_index < 0 or min(cpu_fraction, history_fraction) < 0 or
            not all(math.isfinite(n) for n in (cpu_fraction, history_fraction)) or
            cpu_fraction + history_fraction > 1):
        raise ValueError("Invalid opponent proportions")
    kinds = ("self", "cpu", "historical")
    weights = (1 - cpu_fraction - history_fraction, cpu_fraction, history_fraction)
    counts = [0, 0, 0]
    rng = random.Random(seed)
    jobs = []
    for number in range(start_index + count):
        index = max(range(3), key=lambda i: (round((number + 1) * weights[i] - counts[i], 12), rng.random()))
        job = {"opponent": kinds[index],
               "learner_color": "RED" if (counts[index] + seed) % 2 == 0 else "BLUE"}
        counts[index] += 1
        if number >= start_index:
            jobs.append(job)
    return jobs


def update_start_pool(pool, samples, *, limit, seed, family_id=None):
    """Keep training-only, legal, turn-boundary middlegames with family identity."""
    if limit <= 0:
        return []
    candidates = []
    for row in samples[len(samples) // 3:]:
        board = e.BaseBoard.deserialize(row.state)
        if (board.turn_moves == 0 and not board.did_offer_draw and not board.did_accept_draw and
                board.occupied.bit_count() > 8 and cpu._terminal(board) is None):
            candidates.append({"state": row.state, "family_id": family_id or row.family_id or row.game_id})
    rng = random.Random(seed)
    candidates = rng.sample(candidates, min(8, len(candidates)))
    unique = {entry["state"]: entry for entry in (*pool, *candidates)}
    entries = list(unique.values())
    return rng.sample(entries, limit) if len(entries) > limit else entries
