"""Explicit collection provenance and opt-in value-label eligibility.

Raw outcomes are observations, never rewritten by these filters. Legacy replay
without provenance remains usable in ``all`` mode and is unknown in ``selfplay``
mode. Model IDs should be immutable checkpoint identities supplied by collection;
callers can additionally require an explicit allowlist of recent identities.
"""
from collections import Counter
from functools import lru_cache
import math

from .rules import e, cpu

PROVENANCE_VERSION = 1
_UNSET = object()


def _keys(value, expected, label):
    if not isinstance(value, dict) or set(value) != set(expected):
        raise ValueError(f"Invalid value provenance {label} fields")


def _integer(value, name, minimum=0):
    if type(value) is not int or value < minimum:
        raise ValueError(f"Invalid value provenance {name}")


def _number(value, name, minimum=0):
    if type(value) not in (int, float) or not math.isfinite(value) or value < minimum:
        raise ValueError(f"Invalid value provenance {name}")


def _search(settings):
    if not isinstance(settings, dict):
        raise ValueError("Invalid value provenance search settings")
    if settings.get("kind") == "neural":
        _keys(settings, ("kind", "simulations", "cpuct", "max_nodes", "max_depth",
                         "root_noise", "dirichlet_alpha", "noise_fraction",
                         "temperature_turns", "temperature", "late_temperature"), "neural search")
        _integer(settings["simulations"], "simulations")
        _integer(settings["max_nodes"], "max_nodes", 1)
        _integer(settings["max_depth"], "max_depth", 1)
        _integer(settings["temperature_turns"], "temperature_turns")
        for name in ("cpuct", "dirichlet_alpha", "noise_fraction", "temperature", "late_temperature"):
            _number(settings[name], name)
        if (type(settings["root_noise"]) is not bool or settings["dirichlet_alpha"] <= 0 or
                settings["noise_fraction"] > 1):
            raise ValueError("Invalid value provenance root noise")
    elif settings.get("kind") == "cpu":
        _keys(settings, ("kind", "time_ms", "max_depth", "beam_width", "multi_pv"), "CPU search")
        _number(settings["time_ms"], "time_ms")
        for name in ("max_depth", "beam_width", "multi_pv"):
            _integer(settings[name], name, 1)
    else:
        raise ValueError("Invalid value provenance search kind")


@lru_cache(maxsize=128)
def _final_terminal(state):
    if not isinstance(state, str) or not state:
        raise ValueError("Invalid value provenance final_state")
    try:
        terminal = cpu._terminal(e.BaseBoard.deserialize(state))
    except Exception as exc:
        raise ValueError("Invalid value provenance final_state") from exc
    return None if terminal is None else (terminal["winner"], terminal["reason"])


def validate_value_provenance(provenance, *, outcome=_UNSET, state=None):
    """Reject malformed or contradictory metadata; absence is explicitly unknown."""
    if provenance is None:
        return
    _keys(provenance, ("version", "collection_kind", "players", "terminal",
                       "termination_reason", "final_state"), "record")
    if type(provenance["version"]) is not int or provenance["version"] != PROVENANCE_VERSION:
        raise ValueError("Unsupported value provenance version")
    kind = provenance["collection_kind"]
    if kind not in ("teacher", "self", "cpu", "historical"):
        raise ValueError("Invalid value provenance collection_kind")
    players = provenance["players"]
    _keys(players, ("RED", "BLUE"), "players")
    for player in players.values():
        _keys(player, ("model_id", "search"), "player")
        if not isinstance(player["model_id"], str) or (player["model_id"] and not player["model_id"].strip()):
            raise ValueError("Invalid value provenance model_id")
        _search(player["search"])
    search_kinds = sorted(player["search"]["kind"] for player in players.values())
    expected = ["cpu", "cpu"] if kind == "teacher" else ["cpu", "neural"] if kind == "cpu" else ["neural", "neural"]
    if search_kinds != expected:
        raise ValueError("Value provenance collection kind and players disagree")
    reason = provenance["termination_reason"]
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("Invalid value provenance termination_reason")
    terminal = provenance["terminal"]
    if terminal is not None:
        _keys(terminal, ("winner", "reason"), "terminal")
        if terminal["winner"] not in (None, "RED", "BLUE") or terminal["reason"] != reason:
            raise ValueError("Invalid value provenance terminal")
        recorded = (terminal["winner"], terminal["reason"])
    else:
        recorded = None
    final_state = provenance["final_state"]
    if not isinstance(final_state, str) or not final_state:
        raise ValueError("Invalid value provenance final_state")
    if _final_terminal(final_state) != recorded:
        raise ValueError("Value provenance terminal does not match final rules state")
    if terminal is None and reason not in ("max-turns", "repetition-cutoff"):
        raise ValueError("Invalid value provenance unfinished reason")
    if outcome is not _UNSET:
        if outcome is not None and (type(outcome) is not int or outcome not in (0, 1, 2)):
            raise ValueError("Invalid value provenance WDL outcome")
        if (outcome is None) != (terminal is None):
            raise ValueError("Value provenance terminal and outcome disagree")
        if terminal is not None:
            if state is None:
                raise ValueError("Value provenance outcome validation requires sample state")
            try:
                color = "RED" if e.BaseBoard.deserialize(state).turn == e.RED else "BLUE"
            except Exception as exc:
                raise ValueError("Invalid value provenance sample state") from exc
            expected_outcome = 1 if terminal["winner"] is None else 0 if terminal["winner"] == color else 2
            if outcome != expected_outcome:
                raise ValueError("Value provenance outcome does not match the sample mover")


def _configuration(mode, model_ids):
    if mode not in ("all", "selfplay"):
        raise ValueError("Value target mode must be all or selfplay")
    if model_ids is not None:
        if isinstance(model_ids, (str, bytes)):
            raise ValueError("Value model IDs must be a collection of nonempty strings")
        try:
            model_ids = frozenset(model_ids)
        except TypeError as exc:
            raise ValueError("Value model IDs must be a collection of nonempty strings") from exc
        if any(not isinstance(item, str) or not item.strip() for item in model_ids):
            raise ValueError("Value model IDs must be nonempty strings")
    return model_ids


def value_eligibility_reason(sample, mode="all", model_ids=None):
    """Return ``eligible`` or a stable exclusion reason, validating first."""
    model_ids = _configuration(mode, model_ids)
    provenance = getattr(sample, "value_provenance", None)
    validate_value_provenance(provenance, outcome=sample.outcome, state=sample.state)
    if sample.outcome is None:
        return "unlabeled"
    if mode == "all":
        return "eligible"
    if provenance is None:
        return "unknown-provenance"
    if provenance["collection_kind"] != "self":
        return "mixed-or-teacher"
    red, blue = provenance["players"]["RED"], provenance["players"]["BLUE"]
    if not red["model_id"] or not blue["model_id"]:
        return "unknown-model"
    if red["model_id"] != blue["model_id"]:
        return "different-models"
    if red["search"] != blue["search"]:
        return "different-search-settings"
    if model_ids is not None and red["model_id"] not in model_ids:
        return "outside-model-allowlist"
    return "eligible"


def value_eligible(sample, mode="all", model_ids=None):
    return value_eligibility_reason(sample, mode, model_ids) == "eligible"


def audit_value_targets(samples, mode="all", model_ids=None):
    """Count raw labels separately from admitted labels, overall and by source."""
    model_ids = _configuration(mode, model_ids)
    overall = Counter(rows=0, labeled_rows=0, eligible_rows=0, excluded_rows=0)
    by_source = {}
    for sample in samples:
        reason = value_eligibility_reason(sample, mode, model_ids)
        source = by_source.setdefault(sample.source, Counter())
        for counter in (overall, source):
            counter["rows"] += 1
            counter["labeled_rows"] += sample.outcome is not None
            counter["eligible_rows"] += reason == "eligible"
            counter["excluded_rows"] += reason != "eligible"
            counter[reason] += 1
    return {"mode": mode, "model_ids": None if model_ids is None else sorted(model_ids),
            "totals": dict(overall), "by_source": {key: dict(value) for key, value in sorted(by_source.items())}}
