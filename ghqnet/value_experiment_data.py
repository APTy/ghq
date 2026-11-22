"""Prepare, collect, and assemble a frozen value-target experiment cohort.

Preparation performs no inference. Collection never trains. Assembly keeps the
original family split and excludes cross-partition canonical states, including
mirrors. These are development panels, not untouched promotion evidence.
"""
import argparse
from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import random

from .ablate_replay import file_sha, verify_partition, write_json
from .curriculum import fingerprint
from .data import Sample, SUPPORTED_DATA_VERSIONS, read_replay, write_replay
from .model import rules_source_hash
from .rules import e, cpu
from .value_targets import value_eligible


def raw_rows(path):
    """Read identities for preparation without encoding or searching positions."""
    with Path(path).open() as stream:
        header = json.loads(next(stream))
        if (header.get("version") not in SUPPORTED_DATA_VERSIONS or
                header.get("rules_sha256") != rules_source_hash()):
            raise ValueError("Replay schema or rules mismatch")
        rows = [Sample(**json.loads(line)) for line in stream if line.strip()]
    if any(not row.game_id or not row.family_id for row in rows):
        raise ValueError("Experiment rows require game and family identities")
    return rows


def roots(rows, count, seed, excluded):
    candidates = list(rows)
    random.Random(seed).shuffle(candidates)
    selected, families, states = [], set(), set()
    for row in candidates:
        key = fingerprint(row)
        if row.family_id in families or key in states or key in excluded:
            continue
        board = e.BaseBoard.deserialize(row.state)
        if (board.turn_moves or board.did_offer_draw or board.did_accept_draw or
                board.occupied.bit_count() <= 8 or cpu._terminal(board) is not None):
            continue
        selected.append(row)
        families.add(row.family_id)
        states.add(key)
        if len(selected) == count:
            return selected
    raise ValueError(f"Need {count} independent legal root families; found {len(selected)}")


def prepare(config_path, output):
    config_path, output = Path(config_path).resolve(), Path(output).resolve()
    if output.exists():
        raise ValueError("Preparation output must be a new directory")
    config = json.loads(config_path.read_text())
    inputs = {name: Path(path).resolve() for name, path in config["inputs"].items()}
    hashes = {name: file_sha(path) for name, path in inputs.items()}
    if hashes["checkpoint"] != config["checkpoint_sha256"]:
        raise ValueError("Frozen checkpoint hash mismatch")
    training, validation = raw_rows(inputs["training_roots"]), raw_rows(inputs["validation_roots"])
    state = json.loads(inputs["partition"].read_text())
    partition_check = verify_partition(training, validation, state, None)
    exclusions = state.get("exclusions", {})
    excluded = set(exclusions.get("canonical_state_ids", []))
    for name in config.get("excluded_opening_inputs", []):
        manifest = json.loads(inputs[name].read_text())
        if manifest.get("rules_sha256") != rules_source_hash():
            raise ValueError("Opening manifest rules mismatch")
        excluded.update(fingerprint(Sample(row["state"], [], [])) for row in manifest["openings"])
    forbidden_games, forbidden_families = set(exclusions.get("game_ids", [])), set(exclusions.get("family_ids", []))
    tasks = []
    readiness = {}
    for split, rows, count in (("train", training, config["collection"]["training_games"]),
                               ("validation", validation, config["collection"]["validation_games"])):
        rows = [row for row in rows if row.game_id not in forbidden_games and row.family_id not in forbidden_families]
        legacy = balanced_panel([row for row in rows if fingerprint(row) not in excluded],
                                config["panels"][split + "_legacy"], config["seed"], max_per_family=128)
        chosen = roots(rows, count, config["seed"] + (split == "validation"), excluded)
        readiness[split] = {"root_games": len(chosen), "root_families": len({row.family_id for row in chosen}),
                            "legacy_unique_rows_ready": len(legacy), "fresh_rows_ready": 0,
                            "fresh_rows_required": config["panels"][split + "_fresh"]}
        for row in chosen:
            index = len(tasks)
            task = {"opponent": "self", "model_id": hashes["checkpoint"],
                    "game_id": f"value-{config['seed']}-{split}-{index:04d}",
                    "seed": config["seed"] + index * 7919, "start_state": row.state,
                    "simulations": config["collection"]["simulations"],
                    "max_turns": config["collection"]["max_turns"],
                    "add_noise": True, "temperature_turns": 12}
            tasks.append({"split": split, "family_id": row.family_id, "parent_game_id": row.game_id,
                          "canonical_root": fingerprint(row), "task": task})
    # Freeze every input and search/collection setting before collecting outcomes.
    manifest = {"version": 1, "rules_sha256": rules_source_hash(), "config": config,
                "inputs": {name: {"path": str(path), "sha256": hashes[name]} for name, path in inputs.items()},
                "config_sha256": file_sha(config_path), "model_id": hashes["checkpoint"],
                "partition_check": partition_check, "readiness": readiness,
                "excluded_states": sorted(excluded), "tasks": tasks,
                "prepared_only": True, "training_started": False,
                "limitation": "Reused development families; not an untouched strength or promotion test."}
    source_names = ("value_experiment_data", "data", "value_targets", "selfplay", "collector", "model", "mcts",
                    "encoding", "curriculum", "rules", "ablate_replay", "train")
    manifest["source_sha256"] = {f"ghqnet/{name}.py": file_sha(Path(__file__).parent / f"{name}.py") for name in source_names}
    manifest["source_sha256"].update({f"public/{name}.py": file_sha(Path(__file__).parent.parent / "public" / f"{name}.py")
                                      for name in ("engine", "search")})
    output.mkdir(parents=True)
    write_json(output / "collection-plan.json", manifest)
    return manifest


def load_plan(folder, *, check_sources=True):
    folder = Path(folder)
    manifest = json.loads((folder / "collection-plan.json").read_text())
    if manifest["rules_sha256"] != rules_source_hash():
        raise ValueError("Collection rules changed after preparation")
    for entry in manifest["inputs"].values():
        if file_sha(entry["path"]) != entry["sha256"]:
            raise ValueError(f"Frozen input changed: {entry['path']}")
    if check_sources:
        root = Path(__file__).parent.parent
        for name, digest in manifest["source_sha256"].items():
            if file_sha(root / name) != digest:
                raise ValueError(f"Frozen collection source changed: {name}")
    return manifest


def collect(folder, device="cpu"):
    """Explicit execution only; resume unattempted tasks, never retry lost games."""
    import torch
    from .collector import collect_games, CollectionError, CollectionTimeout
    from .model import load_checkpoint, TorchEvaluator
    from .train import device_name
    folder = Path(folder)
    manifest = load_plan(folder)
    settings = manifest["config"]["collection"]
    status_path = folder / "collection-status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {"attempted": [], "runs": []}
    if status.get("active"):
        raise ValueError("Previous collector did not finish cleanup; explicit recovery is required")
    completed = {row["task"]["game_id"] for row in manifest["tasks"]
                 if (folder / "games" / (row["task"]["game_id"] + ".jsonl")).exists()}
    missing = [index for index, row in enumerate(manifest["tasks"])
               if index not in status["attempted"] and row["task"]["game_id"] not in completed]
    if not missing:
        if len(completed) != len(manifest["tasks"]):
            raise ValueError("Attempted collection has missing game reports; explicit recovery is required")
        return status
    remaining = settings["max_seconds"] - sum(run["metrics"].get("seconds", 0) for run in status["runs"])
    if remaining <= 0:
        raise ValueError("Frozen total collection time budget exhausted")
    torch.set_num_threads(settings["threads"])
    device = device_name(device)
    model, _ = load_checkpoint(manifest["inputs"]["checkpoint"]["path"], device)
    evaluators = {manifest["model_id"]: TorchEvaluator(model, device)}
    # Record attempted tasks before dispatch. After a hard interruption, do not
    # silently reroll their outcomes; a separately recorded recovery is required.
    status["attempted"] = sorted(set(status["attempted"]) | set(missing))
    status["active"] = True
    write_json(status_path, status)
    metrics = {}

    def completed_game(index, result):
        rows, report = result
        entry = manifest["tasks"][missing[index]]
        report.update(experiment_split=entry["split"], parent_game_id=entry["parent_game_id"])
        rows = [replace(row, family_id=entry["family_id"]) for row in rows]
        path = folder / "games" / (entry["task"]["game_id"] + ".jsonl")
        if path.exists():
            raise ValueError("Refusing to replace a collected game")
        write_replay(path, rows, game=report, family_id=entry["family_id"])

    try:
        collect_games([manifest["tasks"][index]["task"] for index in missing], evaluators,
                      workers=settings["workers"], batch_size=settings["inference_batch_size"],
                      batch_wait_ms=5, game_timeout_s=settings["game_timeout_seconds"],
                      collection_timeout_s=remaining, metrics=metrics,
                      on_result=completed_game)
    except (CollectionError, CollectionTimeout) as error:
        metrics["error"] = str(error)
        raise
    finally:
        # The collector can identify jobs never dispatched after orderly
        # cancellation. Continue those safely; failed/inflight games are not
        # silently rerolled. A hard process loss stays conservatively attempted.
        unstarted = metrics.get("unstarted_indices", [])
        if any(type(index) is not int or not 0 <= index < len(missing) for index in unstarted):
            raise ValueError("Collector returned invalid unstarted task indices")
        status["attempted"] = sorted(set(status["attempted"]) - {missing[index] for index in unstarted})
        status["active"] = False
        status["runs"].append({"indices": missing, "metrics": metrics})
        write_json(status_path, status)
    return status


def balanced_panel(rows, count, seed, max_per_family=64):
    """Equal family rounds, canonical deduplication; no outcome-class balancing."""
    rng, families, seen = random.Random(seed), defaultdict(list), set()
    for row in rows:
        key = fingerprint(row)
        if key not in seen:
            families[row.family_id].append(row)
            seen.add(key)
    names = sorted(families)
    rng.shuffle(names)
    for name in names:
        rng.shuffle(families[name])
    result = []
    for offset in range(max_per_family):
        for name in names:
            if len(families[name]) > offset:
                result.append(families[name][offset])
                if len(result) == count:
                    return result
    raise ValueError(f"Need {count} distinct panel rows; found {len(result)} within family caps")


def assemble(folder):
    folder = Path(folder)
    manifest = load_plan(folder)
    output = folder / "dataset"
    if output.exists():
        raise ValueError("Dataset output must be new")
    inputs, config = manifest["inputs"], manifest["config"]
    state = json.loads(Path(inputs["partition"]["path"]).read_text())
    forbidden_games = set(state.get("exclusions", {}).get("game_ids", []))
    forbidden_families = set(state.get("exclusions", {}).get("family_ids", []))
    registries = {"train": set(state["partition"]["train_states"]),
                  "validation": set(state["partition"]["validation_states"])}
    old = {"train": read_replay(inputs["training_roots"]["path"]),
           "validation": read_replay(inputs["validation_roots"]["path"])}
    fresh = {"train": [], "validation": []}
    excluded, reports = set(manifest["excluded_states"]), []
    omitted = defaultdict(int)
    # Holdout first so new state collisions consistently remain held out.
    for split in ("validation", "train"):
        other = registries["train" if split == "validation" else "validation"]
        for entry in manifest["tasks"]:
            if entry["split"] != split:
                continue
            path = folder / "games" / (entry["task"]["game_id"] + ".jsonl")
            if not path.exists():
                raise ValueError("Incomplete fixed collection; do not select a successful subset")
            with path.open() as stream:
                report = json.loads(next(stream))["metadata"]["game"]
            rows = read_replay(path)
            if report["game_id"] != entry["task"]["game_id"]:
                raise ValueError("Collected game identity mismatch")
            task = entry["task"]
            provenance = report.get("value_provenance")
            if (report.get("seed") != task["seed"] or report.get("model_id") != task["model_id"] or
                    not isinstance(provenance, dict) or provenance.get("collection_kind") != "self"):
                raise ValueError("Collected report does not match frozen task")
            for player in provenance["players"].values():
                search = player["search"]
                if (player["model_id"] != task["model_id"] or search["simulations"] != task["simulations"] or
                        search["root_noise"] != task["add_noise"] or
                        search["temperature_turns"] != task["temperature_turns"]):
                    raise ValueError("Collected search settings do not match frozen task")
            if report["game_id"] in forbidden_games or entry["family_id"] in forbidden_families:
                raise ValueError("Collected game belongs to an excluded family or game")
            reports.append({"game_id": report["game_id"], "split": split,
                            "terminal": report["terminal"], "reason": report["reason"]})
            for row in rows:
                if row.family_id != entry["family_id"] or row.game_id != entry["task"]["game_id"]:
                    raise ValueError("Collected row family or game identity mismatch")
                if row.value_provenance != provenance:
                    raise ValueError("Collected row provenance disagrees with game report")
                key = fingerprint(row)
                if key in other or key in excluded:
                    omitted[split + "_collision"] += 1
                    continue
                registries[split].add(key)
                if value_eligible(row, "selfplay", [manifest["model_id"]]):
                    fresh[split].append(row)
                else:
                    omitted[split + "_unlabeled_or_ineligible"] += 1
    panels = {}
    for split in ("train", "validation"):
        count = config["panels"][split + "_fresh"]
        old_ids = {fingerprint(row) for row in old[split]}
        # Keep all legacy policy examples available without duplicating their
        # value inputs in the fresh half of the panel.
        novel = [row for row in fresh[split] if fingerprint(row) not in old_ids]
        new = balanced_panel(novel, count, config["seed"] + (split == "validation"))
        old_eligible = [row for row in old[split] if fingerprint(row) not in excluded and
                        row.game_id not in forbidden_games and row.family_id not in forbidden_families]
        legacy_count = config["panels"][split + "_legacy"]
        legacy = balanced_panel(old_eligible, legacy_count, config["seed"], max_per_family=128)
        panels[split] = legacy + new
    state["partition"]["train_states"] = sorted(registries["train"])
    state["partition"]["validation_states"] = sorted(registries["validation"])
    check = verify_partition(panels["train"], panels["validation"], state, None)
    output.mkdir()
    for split, kind in (("train", "training"), ("validation", "validation")):
        write_replay(output / f"{kind}.jsonl", panels[split], kind=kind, purpose="value-target-development")
    write_json(output / "partition.json", state)
    report = {"partition_check": check, "omitted": dict(omitted), "games": reports,
              "model_id": manifest["model_id"], "collection_plan_sha256": file_sha(folder / "collection-plan.json"),
              "panel_rows": {key: len(value) for key, value in panels.items()}, "promoted": False}
    write_json(output / "assembly.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    preparation = sub.add_parser("prepare", help="Freeze roots/settings only; no games or training")
    preparation.add_argument("--config", type=Path, required=True)
    preparation.add_argument("--output", type=Path, required=True)
    collection = sub.add_parser("collect", help="Explicitly execute the fixed game collection; no training")
    collection.add_argument("folder", type=Path)
    collection.add_argument("--device", choices=("cpu", "cuda", "mps"), default="cpu")
    assembly = sub.add_parser("assemble", help="Build bounded panels only after every scheduled game has a report")
    assembly.add_argument("folder", type=Path)
    args = parser.parse_args()
    result = (prepare(args.config, args.output) if args.command == "prepare" else
              collect(args.folder, args.device) if args.command == "collect" else assemble(args.folder))
    print(json.dumps({"command": args.command, "prepared_games": len(result.get("tasks", [])),
                      "panel_rows": result.get("panel_rows"), "training_started": False}))


if __name__ == "__main__":
    main()
