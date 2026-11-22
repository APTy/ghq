"""Replay records retain exact rules states, legal actions, and label provenance."""
from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .model import rules_source_hash
from .rules import e, copy_board
from .value_targets import validate_value_provenance

DATA_VERSION = 3
SUPPORTED_DATA_VERSIONS = (1, 2, DATA_VERSION)


@dataclass
class Sample:
    state: str
    actions: list[str]
    policy: list[float]
    outcome: int | None = None  # W=0, D=1, L=2, for THIS state's player
    game_id: str = ""
    source: str = "mcts"
    # Related continuations and mirrored variants share a family for holdouts.
    family_id: str = ""
    priority: str = "normal"
    model_id: str = ""
    # Missing metadata is explicitly unknown, including all legacy v1/v2 rows.
    value_provenance: dict | None = None

    def validate(self):
        board = e.BaseBoard.deserialize(self.state)
        legal = {move.uci() for move in copy_board(board).generate_legal_moves()}
        if len(self.actions) != len(set(self.actions)) or set(self.actions) != legal:
            raise ValueError("Replay action list must contain every exact legal action once")
        policy = np.asarray(self.policy)
        if policy.shape != (len(self.actions),) or not np.isfinite(policy).all() or (policy < 0).any():
            raise ValueError("Invalid replay policy")
        if not np.isclose(policy.sum(), 1.0):
            raise ValueError("Replay policy must sum to one")
        if self.outcome not in (None, 0, 1, 2):
            raise ValueError("Outcome must be W/D/L or absent for unfinished games")
        if self.priority not in ("normal", "hard"):
            raise ValueError("Replay priority must be normal or hard")
        validate_value_provenance(self.value_provenance, outcome=self.outcome, state=self.state)
        return board


def label_game(samples, terminal):
    """Never turn time, length, or repetition cutoffs into fabricated draws."""
    for sample in samples:
        if terminal is None:
            sample.outcome = None
        elif terminal["winner"] is None:
            sample.outcome = 1
        else:
            winner = e.RED if terminal["winner"] == "RED" else e.BLUE
            sample.outcome = 0 if e.BaseBoard.deserialize(sample.state).turn == winner else 2


def write_replay(path, samples, **metadata):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w") as output:
        output.write(json.dumps({"version": DATA_VERSION, "rules_sha256": rules_source_hash(),
                                 "metadata": metadata}) + "\n")
        for sample in samples:
            output.write(json.dumps(asdict(sample)) + "\n")
    temporary.replace(path)


def read_replay(path):
    with Path(path).open() as source:
        header = json.loads(next(source))
        if header.get("version") not in SUPPORTED_DATA_VERSIONS or header.get("rules_sha256") != rules_source_hash():
            raise ValueError("Replay schema or rules do not match this engine")
        samples = [Sample(**json.loads(line)) for line in source if line.strip()]
    for sample in samples:
        sample.validate()
    return samples
