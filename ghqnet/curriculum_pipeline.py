"""Bounded mixed-opponent training with portable curriculum checkpoints."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import numpy as np
import torch

from .curriculum import (CURRICULUM_VERSION, ReplayPartition, fingerprint,
                         opponent_schedule, retain_replay, update_start_pool)
from .data import read_replay, write_replay
from .model import NetworkConfig, PolicyValueNet, TorchEvaluator, load_checkpoint, save_checkpoint, rules_source_hash
from .pipeline import write_json
from .train import device_name, fit
from .training_budget import ReplayExposure


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_reference(parent, name, expected_hash, *, label):
    if not isinstance(name, str) or Path(name).name != name or not expected_hash:
        raise ValueError(f"Checkpoint lacks an exact {label} reference")
    path = parent / name
    if not path.is_file() or file_sha(path) != expected_hash:
        raise ValueError(f"Checkpoint {label} file missing or hash mismatch: {name}")
    return path


def configuration(args):
    def encode(value):
        if isinstance(value, Path):
            return str(value)
        if isinstance(value, list):
            return [encode(item) for item in value]
        return value
    return {key: encode(value) for key, value in vars(args).items()}


def collection_stop_reason(metrics, error=None):
    """Normalize collector metadata, including older exception constructors."""
    reason = metrics.get("stop_reason") or getattr(error, "stop_reason", None)
    if isinstance(reason, str):
        reason = {"kind": reason, **(metrics.get("failure") or getattr(error, "details", {}))}
    if reason is not None and (not isinstance(reason, dict) or not isinstance(reason.get("kind"), str)):
        raise ValueError("Malformed collector stop reason")
    return reason


def load_exclusions(paths):
    result = {name: set() for name in ("game_ids", "family_ids", "canonical_state_ids", "replay_sha256")}
    manifests = []
    for path in paths:
        content = json.loads(path.read_text())
        exclusions = content.get("exclusions")
        if not isinstance(exclusions, dict):
            raise ValueError(f"Missing exclusions in evaluation manifest: {path}")
        for key in result:
            values = exclusions.get(key, [])
            if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
                raise ValueError(f"Malformed {key} in evaluation manifest: {path}")
            result[key].update(values)
        manifests.append(file_sha(path))
    return {key: sorted(values) for key, values in result.items()}, sorted(manifests)


def run_curriculum(args):
    # Import lazily: prototype mode and configuration validation do not start
    # process pools or require the concurrent collection implementation.
    from .collector import CollectionError, CollectionTimeout, collect_games

    started = time.perf_counter()
    deadline = None if args.max_seconds is None else started + args.max_seconds
    device = device_name(args.device)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    optimizer, replay, validation = None, [], []
    next_iteration = 0
    exclusions, manifest_hashes = load_exclusions(args.exclude_manifest)
    partition = ReplayPartition(args.seed, args.validation_fraction)
    state = {"version": CURRICULUM_VERSION, "seed": args.seed, "teacher_games": 0,
             "teacher_positions": 0, "bootstrap_complete": args.teacher_positions == 0,
             "mixed_games": 0, "next_serial": 0, "history": [], "start_pool": [],
             "last_iteration": -1, "excluded_positions": 0, "excluded_games": 0,
             "exclusions": exclusions, "exclusion_manifest_sha256": manifest_hashes,
             "imported_checkpoint_has_reserved_training": False}
    source_parent = None
    if args.resume:
        checkpoint_path = args.resume.resolve()
        source_parent = checkpoint_path.parent
        model, checkpoint = load_checkpoint(checkpoint_path, device)
        metadata = checkpoint["metadata"]
        if metadata.get("seed") != args.seed:
            raise ValueError("Resume seed must match checkpoint seed")
        next_iteration = metadata.get("iteration", 0) + 1
        replay = read_replay(read_reference(source_parent, metadata.get("replay_buffer"),
                                           metadata.get("replay_buffer_sha256"), label="replay-buffer"))
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-4)
        if "optimizer_state" in checkpoint:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        if "curriculum_state" in metadata:
            state_path = read_reference(source_parent, metadata["curriculum_state"],
                                        metadata.get("curriculum_state_sha256"), label="curriculum")
            state = json.loads(state_path.read_text())
            if state.get("version") != CURRICULUM_VERSION or state.get("seed") != args.seed:
                raise ValueError("Curriculum version or seed mismatch")
            partition = ReplayPartition.from_dict(state["partition"])
            if partition.validation_fraction != args.validation_fraction:
                raise ValueError("Resume validation fraction must match the saved curriculum")
            if manifest_hashes and manifest_hashes != state["exclusion_manifest_sha256"]:
                raise ValueError("Resume evaluation exclusion identity differs from the checkpoint")
            validation = read_replay(read_reference(source_parent, metadata.get("validation_buffer"),
                                                   metadata.get("validation_buffer_sha256"), label="validation-buffer"))
            for entry in state["history"]:
                read_reference(source_parent, entry["file"], entry["sha256"], label="history checkpoint")
            if any(fingerprint(row) in partition.validation_states for row in replay):
                raise ValueError("Saved training replay overlaps validation registry")
            if any(fingerprint(row) in partition.train_states for row in validation):
                raise ValueError("Saved validation replay overlaps training registry")
        else:
            # A prototype import has already trained on every replay row. Those
            # positions can never become independent held-out measurements.
            partition.train_states.update(fingerprint(row) for row in replay)
            reserved = set(exclusions["canonical_state_ids"])
            state["imported_checkpoint_has_reserved_training"] = bool(partition.train_states & reserved)
            replay = [row for row in replay if fingerprint(row) not in reserved and
                      row.game_id not in exclusions["game_ids"] and row.family_id not in exclusions["family_ids"]]
            state["last_iteration"] = next_iteration - 1
    else:
        if args.output.exists() and any(args.output.iterdir()):
            raise ValueError("Output directory is nonempty; choose a new run or explicitly --resume")
        model = PolicyValueNet(NetworkConfig(args.blocks, args.channels)).to(device)

    exposure = (ReplayExposure.restore(state["replay_exposure"], args.replay_ratio)
                if "replay_exposure" in state else
                ReplayExposure(args.replay_ratio, legacy_states=len(partition.train_states) if args.resume else 0))
    if args.resume and "replay_exposure" not in state and exposure.migration is None:
        exposure.migration = "Legacy checkpoint: no retroactive exposure credit; no known training states."
    if exposure.unique_states + exposure.legacy_states != len(partition.train_states):
        raise ValueError("Replay exposure ledger does not match the persistent training-state registry")
    # This counter authorizes an actual training update, independently of how
    # often metadata is saved. Pending credit waits for another novel intake.
    new_training_states = 0

    # Older curricula stopped at their first incomplete collection. Recover
    # their scheduled-attempt count without reusing abandoned game identities.
    abandoned = len(state.get("last_collection", {}).get("abandoned_game_ids", []))
    state.setdefault("teacher_attempts", state["teacher_games"] +
                     (abandoned if state.get("last_phase") == "teacher" else 0))
    state.setdefault("mixed_schedule_cursor", state["mixed_games"] +
                     (abandoned if state.get("last_phase") == "mixed" else 0))
    state.setdefault("history_cohort_cursor", 0)
    state.setdefault("game_timeout_counts", {"teacher": 0, "mixed": 0})
    state.setdefault("opponent_schedule_config", {"seed": args.seed,
                     "cpu_fraction": args.cpu_fraction, "history_fraction": args.history_fraction})
    if state["opponent_schedule_config"] != {"seed": args.seed, "cpu_fraction": args.cpu_fraction,
                                             "history_fraction": args.history_fraction}:
        raise ValueError("Resume opponent proportions must match the saved schedule")

    for pattern in ("checkpoint-*.pt", "metrics-*.json", "run-*.json", "curriculum-*.json"):
        for existing in args.output.glob(pattern):
            suffix = existing.stem.rsplit("-", 1)[-1]
            if suffix.isdigit() and int(suffix) >= next_iteration:
                raise ValueError(f"Resume would overwrite existing run output {existing}; use a new output directory")
    args.output.mkdir(parents=True, exist_ok=True)
    if source_parent is not None:
        for entry in state["history"]:
            target = args.output / entry["file"]
            if target.exists():
                if file_sha(target) != entry["sha256"]:
                    raise ValueError(f"Existing historical checkpoint differs: {target}")
            else:
                shutil.copyfile(source_parent / entry["file"], target)

    config = configuration(args)
    config.update({"device_used": device, "rules_sha256": rules_source_hash(), "torch": str(torch.__version__),
                   "parameters": sum(p.numel() for p in model.parameters()),
                   "effective_learning_rates": ([group["lr"] for group in optimizer.param_groups]
                                                if optimizer else [args.learning_rate]),
                   "started_at": time.time(), "status": "curriculum-not-promoted"})
    write_json(args.output / f"run-{next_iteration:04d}.json", config)
    exclusions = {key: set(values) for key, values in state["exclusions"].items()}
    state["bootstrap_complete"] = (state["teacher_positions"] >= args.teacher_positions or
                                    state["teacher_attempts"] >= args.teacher_max_games)
    last_checkpoint = str(args.resume) if args.resume else None
    reports = []
    timed_out = False
    stop_reason = None

    def remaining(reserve=0):
        return None if deadline is None else max(0.0, deadline - time.perf_counter() - reserve)

    def emit(event):
        print(json.dumps(event), flush=True)
        with (args.output / "events.jsonl").open("a") as output:
            output.write(json.dumps(event) + "\n")

    def ingest(samples, *, family=None, teacher=False):
        nonlocal replay, validation, new_training_states
        if not samples:
            return 0
        identifiers = {row.game_id for row in samples}
        families = {family or row.family_id or row.game_id for row in samples}
        full_family = "game:" + hashlib.sha256(json.dumps(
            [fingerprint(row) for row in samples], separators=(",", ":")).encode("ascii")).hexdigest()[:20]
        if (identifiers & exclusions["game_ids"] or families & exclusions["family_ids"] or
                full_family in exclusions["family_ids"]):
            state["excluded_games"] += 1
            state["excluded_positions"] += len(samples)
            return 0
        safe = [row for row in samples if fingerprint(row) not in exclusions["canonical_state_ids"]]
        state["excluded_positions"] += len(samples) - len(safe)
        previous_unique = len(partition.train_states)
        training, heldout, duplicate = partition.add_game(safe, family_id=family)
        admitted_unique = len(partition.train_states) - previous_unique
        exposure.admit(admitted_unique)
        new_training_states += admitted_unique
        state["excluded_positions"] += duplicate
        replay.extend(training)
        validation.extend(heldout)
        if len(validation) > args.validation_size:
            # Persistent registry preserves exclusions even when reported
            # validation examples are bounded to control evaluation overhead.
            validation = random.Random(args.seed).sample(validation, args.validation_size)
        state["start_pool"] = update_start_pool(state["start_pool"], training,
                                                  limit=args.start_pool_size, seed=args.seed + state["next_serial"],
                                                  family_id=family)
        if teacher:
            state["teacher_positions"] += len(training)
        return len(training)

    if args.input_replay:
        if args.resume and "curriculum_state" in checkpoint["metadata"]:
            raise ValueError("Import replay only on a fresh curriculum; resume retains its exact saved buffers")
        for path in args.input_replay:
            if file_sha(path) in exclusions["replay_sha256"]:
                raise ValueError(f"Training input is reserved for evaluation: {path}")
            rows = read_replay(path)
            groups = {}
            for row in rows:
                groups.setdefault(row.game_id, []).append(row)
            for game_rows in groups.values():
                ingest(game_rows, teacher=all(row.source.startswith("ghq-cpu") for row in game_rows))

    def save_update(games, phase, *, incomplete=False, collection_error=None, train=True):
        nonlocal optimizer, replay, next_iteration, last_checkpoint, new_training_states
        iteration = next_iteration
        if replay:
            replay = retain_replay(replay, args.replay_size,
                                   teacher_fraction=args.teacher_replay_fraction,
                                   hard_fraction=args.hard_replay_fraction, seed=args.seed + iteration)
        budget = exposure.available
        newly_admitted = new_training_states
        if collection_error is not None or not train:
            # An inference failure may leave the device unusable. Preserve the
            # last trained weights/optimizer and completed data without any
            # more model execution, so this checkpoint can resume elsewhere.
            metrics = {"samples": len(replay), "validation_samples": len(validation),
                       "steps": 0, "trained_examples": 0, "seconds": 0, "interrupted": True,
                       "reason": ("Collection failed; training was not attempted" if collection_error is not None
                                  else "No completed games; training was not attempted")}
        elif replay and newly_admitted and budget:
            optimizer, metrics = fit(model, replay, epochs=args.epochs, batch_size=args.batch_size,
                                     learning_rate=args.learning_rate, device=device, seed=args.seed + iteration,
                                     optimizer=optimizer, validation_samples=validation,
                                     max_seconds=remaining(args.checkpoint_reserve_seconds), max_examples=budget)
            exposure.debit(metrics["trained_examples"])
        else:
            metrics = {"samples": len(replay), "validation_samples": len(validation), "steps": 0,
                       "trained_examples": 0, "seconds": 0, "interrupted": False,
                       "reason": ("No newly admitted unique training states" if not newly_admitted else
                                  "Replay exposure credit below one row" if not budget else "Empty training replay")}
        metrics.update(new_unique_training_states=newly_admitted,
                       replay_exposure_budget=budget, replay_exposure=exposure.to_dict())
        # Pure saves and failed collection do not spend credit. Credit remains
        # in the ledger; novelty only authorizes this update, never a later save.
        new_training_states = 0
        # Metadata-only saves must not replace useful opponents with repeated
        # snapshots of unchanged weights. An empty history needs one baseline.
        if metrics["steps"] or not state["history"]:
            temporary = args.output / "history-pending.pt"
            save_checkpoint(temporary, model, iteration=iteration, seed=args.seed, purpose="frozen-opponent")
            digest = file_sha(temporary)
            history_name = f"history-{digest[:16]}.pt"
            history_path = args.output / history_name
            if history_path.exists():
                if file_sha(history_path) != digest:
                    raise ValueError("Historical checkpoint name collision")
                temporary.unlink()
            else:
                temporary.replace(history_path)
            state["history"] = (state["history"] + [{"file": history_name, "sha256": digest,
                                                     "iteration": iteration}])[-args.history_size:]
        state["last_iteration"] = iteration
        state["partition"] = partition.to_dict()
        state["replay_exposure"] = exposure.to_dict()
        state["bootstrap_complete"] = (state["teacher_positions"] >= args.teacher_positions or
                                        state["teacher_attempts"] >= args.teacher_max_games)
        state["bootstrap_target_reached"] = state["teacher_positions"] >= args.teacher_positions
        state["last_phase"] = phase
        state["last_collection_incomplete"] = incomplete
        state["last_collection_error"] = collection_error
        state["stop_reason"] = stop_reason
        replay_path = args.output / f"replay-buffer-{iteration:04d}.jsonl"
        validation_path = args.output / f"validation-buffer-{iteration:04d}.jsonl"
        state_path = args.output / f"curriculum-{iteration:04d}.json"
        write_replay(replay_path, replay, kind="training", iteration=iteration)
        write_replay(validation_path, validation, kind="validation", iteration=iteration)
        write_json(state_path, state)
        checkpoint_path = args.output / f"checkpoint-{iteration:04d}.pt"
        save_checkpoint(checkpoint_path, model, optimizer, iteration=iteration, seed=args.seed,
                        metrics=metrics, run_config=config,
                        replay_buffer=replay_path.name, replay_buffer_sha256=file_sha(replay_path),
                        validation_buffer=validation_path.name, validation_buffer_sha256=file_sha(validation_path),
                        curriculum_state=state_path.name, curriculum_state_sha256=file_sha(state_path))
        report = {"iteration": iteration, "phase": phase, "training": metrics, "games": games,
                  "collection_incomplete": incomplete, "teacher_positions": state["teacher_positions"],
                  "teacher_games": state["teacher_games"], "mixed_games": state["mixed_games"],
                  "bootstrap_target_reached": state["bootstrap_target_reached"],
                  "collection_error": collection_error,
                  "collection_stop_reason": state.get("last_collection", {}).get("stop_reason"),
                  "promoted": False}
        write_json(args.output / f"metrics-{iteration:04d}.json", report)
        emit({"event": "trained" if metrics["steps"] else "checkpoint-saved",
              "checkpoint": str(checkpoint_path), "phase": phase, **metrics})
        reports.append(report)
        last_checkpoint = str(checkpoint_path)
        next_iteration += 1

    def collect(count, teacher):
        nonlocal timed_out, stop_reason
        # Checkpoint iteration can advance without changing weights. Exclude
        # the current model by snapshot identity, not checkpoint-save number.
        history = [entry for entry in state["history"]
                   if entry["sha256"] != state["history"][-1]["sha256"]]
        cohort = (history[state["history_cohort_cursor"] % len(history)]
                  if not teacher and history and args.history_fraction > 0 else None)
        current_id = (state["history"][-1]["sha256"] if state["history"] else
                      file_sha(args.resume) if args.resume else "initial-" + hashlib.sha256(
                          json.dumps({"seed": args.seed, "network": model.config.__dict__,
                                      "torch": str(torch.__version__)}, sort_keys=True).encode()).hexdigest())
        evaluators = {} if teacher else {current_id: TorchEvaluator(model, device)}
        if cohort is not None:
            historical, _ = load_checkpoint(args.output / cohort["file"], device)
            evaluators[cohort["sha256"]] = TorchEvaluator(historical, device)
        serial = state["next_serial"]
        schedule = ([{"opponent": "self", "learner_color": "RED"}] * count if teacher else
                    opponent_schedule(count, seed=args.seed, start_index=state["mixed_schedule_cursor"],
                                      cpu_fraction=args.cpu_fraction, history_fraction=args.history_fraction))
        schedule_start = state["mixed_schedule_cursor"] if not teacher else None
        tasks, families = [], {}
        for index, assignment in enumerate(schedule):
            number = serial + index
            # Opening and start-pool randomness does not reset with chunk size.
            rng = random.Random(args.seed + number)
            game_id = f"curr-{args.seed}-{'teacher' if teacher else 'mixed'}-{number:07d}"
            task = {**assignment, "teacher": teacher, "model_id": current_id, "game_id": game_id,
                    "seed": args.seed + number * 7919, "simulations": args.simulations,
                    "max_turns": args.max_turns, "opening_turns": rng.randrange(args.opening_turns + 1),
                    "teacher_ms": args.teacher_ms, "teacher_depth": args.teacher_depth,
                    "teacher_beam_width": args.teacher_beam_width}
            if task["opponent"] == "historical":
                if cohort is not None:
                    task["opponent_model_id"] = cohort["sha256"]
                else:
                    task["opponent"] = "cpu"
            family = game_id
            if not teacher and state["start_pool"] and rng.random() < args.middlegame_fraction:
                start = rng.choice(state["start_pool"])
                task["start_state"], family = start["state"], start["family_id"]
            tasks.append(task)
            families[game_id] = family
        state["next_serial"] += count
        if teacher:
            state["teacher_attempts"] += count
        else:
            state["mixed_schedule_cursor"] += count
            state["history_cohort_cursor"] += 1
        metrics = {}
        emit({"event": "collection-start", "teacher": teacher, "tasks": len(tasks),
              "teacher_games": state["teacher_games"], "teacher_positions": state["teacher_positions"],
              "teacher_attempts": state["teacher_attempts"], "mixed_games": state["mixed_games"],
              "mixed_attempts": state["mixed_schedule_cursor"], "next_serial": state["next_serial"],
              "opponent_schedule_start": schedule_start, "model_id": current_id,
              "historical_cohort": cohort, "evaluator_model_ids": sorted(evaluators),
              "scheduled_opponents": {kind: sum(job["opponent"] == kind for job in schedule)
                                      for kind in ("self", "cpu", "historical")}})

        def completed(index, result):
            samples, report = result
            report["requested_opponent"] = schedule[index]["opponent"]
            game_id = tasks[index]["game_id"]
            path = args.output / "replay" / f"{game_id}.jsonl"
            if path.exists():
                raise ValueError(f"Refusing to overwrite collected game {path}")
            rows = [replace(row, family_id=families[game_id]) for row in samples]
            write_replay(path, rows, game=report, family_id=families[game_id])
            emit({"event": "game", **{key: value for key, value in report.items()
                                      if key not in ("moves", "final_state")}})

        failure = None

        def collection_event(event):
            # Persist watchdog evidence immediately, even if the remaining
            # collection later encounters another failure or the run stops.
            kind = event.get("event", "collector-event").replace("_", "-")
            emit({**event, "event": kind, "teacher": teacher,
                  "model_id": current_id, "historical_cohort": cohort})

        try:
            collection_budget = remaining(args.checkpoint_reserve_seconds)
            if collection_budget == 0:
                metrics.update(unstarted_indices=list(range(len(tasks))), inflight_indices=[],
                               stop_reason={"kind": "global_deadline", "message": "Run budget exhausted before collection"})
                raise CollectionTimeout("Run budget exhausted before collection", stop_reason=metrics["stop_reason"])
            results = collect_games(tasks, evaluators, workers=args.workers,
                                    batch_size=args.inference_batch_size, batch_wait_ms=args.batch_wait_ms,
                                    game_timeout_s=args.game_timeout_seconds,
                                    collection_timeout_s=collection_budget,
                                    inference_protocol=args.inference_protocol, profile=args.profile_inference,
                                    metrics=metrics, on_result=completed, on_event=collection_event)
        except CollectionTimeout as error:
            results = error.results
            detail = collection_stop_reason(metrics, error)
            detail = detail or {"kind": "unknown_timeout", "message": str(error)}
            metrics["stop_reason"] = detail
            if detail["kind"] == "global_deadline":
                timed_out = True
                stop_reason = detail
            else:
                # Startup/request timeouts are infrastructure failures. They
                # must remain distinguishable from an exhausted run budget.
                failure = error
        except CollectionError as error:
            results = error.results
            failure = error
        games = []
        for samples, report in results:
            family = families[report["game_id"]]
            ingest(samples, family=family, teacher=teacher)
            state["teacher_games" if teacher else "mixed_games"] += 1
            games.append(report)
        completed_ids = {game["game_id"] for game in games}
        phase = "teacher" if teacher else "mixed"
        timeout_indices = metrics.get("timed_out_indices", [])
        if any(type(index) is not int or not 0 <= index < len(tasks) for index in timeout_indices):
            raise ValueError("Collector reported an invalid timed-out task index")
        timeout_ids = [tasks[index]["game_id"] for index in timeout_indices]
        if completed_ids & set(timeout_ids):
            raise ValueError("Collector reported a game as both completed and timed out")
        state["game_timeout_counts"][phase] += len(timeout_ids)
        detail = collection_stop_reason(metrics, failure)
        detail = detail or {"kind": ("collection_failure" if failure is not None else
                                    "completed_with_game_timeouts" if timeout_ids else "completed")}
        if failure is not None:
            detail = {**detail, "message": str(failure)}
            stop_reason = detail
        abandoned_ids = [task["game_id"] for task in tasks if task["game_id"] not in completed_ids]
        incomplete = bool(abandoned_ids) or timed_out or failure is not None
        state["last_collection"] = {"scheduled": len(tasks), "completed": len(games),
                                    "abandoned_game_ids": abandoned_ids,
                                    "timed_out_game_ids": timeout_ids,
                                    "game_timeouts": metrics.get("game_timeouts", []),
                                    "unstarted_game_ids": [tasks[i]["game_id"] for i in metrics.get("unstarted_indices", [])],
                                    "inflight_game_ids": [tasks[i]["game_id"] for i in metrics.get("inflight_indices", [])],
                                    "stop_reason": detail, "incomplete": incomplete,
                                    "opponent_schedule_start": schedule_start,
                                    "historical_cohort": cohort}
        emit({"event": "collection", "teacher": teacher,
              **metrics, "incomplete": incomplete, "stop_reason": detail,
              "opponent_schedule_start": schedule_start,
              "historical_cohort": cohort})
        if failure is not None:
            save_update(games, "teacher" if teacher else "mixed", incomplete=True,
                        collection_error=str(failure))
            summary = {"status": "curriculum-collection-failed", "checkpoint": last_checkpoint,
                       "error": str(failure), "updates": len(reports), "promoted": False,
                       "teacher_games": state["teacher_games"], "teacher_positions": state["teacher_positions"],
                       "mixed_games": state["mixed_games"], "stop_reason": detail,
                       "replay_exposure": exposure.to_dict(),
                       "seconds": time.perf_counter() - started}
            write_json(args.output / "summary.json", summary)
            emit({"event": "failed", **summary})
            failure.add_note(f"Completed progress saved to {last_checkpoint}")
            raise failure
        return games

    # Bootstrap updates are bounded in games and positions. Checkpointing every
    # chunk makes long CPU teacher generation resumable before the GPU phase.
    while not state["bootstrap_complete"]:
        if remaining(args.checkpoint_reserve_seconds) == 0:
            timed_out = True
            stop_reason = {"kind": "global_deadline", "phase": "teacher"}
            break
        games = []
        chunk_remaining = min(args.teacher_update_games, args.teacher_max_games - state["teacher_attempts"])
        incomplete = False
        while chunk_remaining > 0 and state["teacher_positions"] < args.teacher_positions:
            wave = min(chunk_remaining, max(args.workers, args.workers * 4))
            completed = collect(wave, teacher=True)
            games.extend(completed)
            chunk_remaining -= wave
            incomplete |= state["last_collection"]["incomplete"]
            if not completed and not timed_out:
                stop_reason = {"kind": "no_completed_games", "phase": "teacher",
                               "collection": state["last_collection"]["stop_reason"]}
            if timed_out or stop_reason is not None:
                break
        save_update(games, "teacher", incomplete=incomplete, train=bool(games))
        if timed_out or stop_reason is not None:
            break
    if not timed_out and stop_reason is None:
        for _ in range(args.iterations):
            if args.total_games and state["mixed_games"] >= args.total_games:
                break
            if remaining(args.checkpoint_reserve_seconds) == 0:
                timed_out = True
                stop_reason = {"kind": "global_deadline", "phase": "mixed"}
                break
            count = min(args.games, args.total_games - state["mixed_games"]) if args.total_games else args.games
            games = collect(count, teacher=False)
            if not games and not timed_out:
                stop_reason = {"kind": "no_completed_games", "phase": "mixed",
                               "collection": state["last_collection"]["stop_reason"]}
            save_update(games, "mixed", incomplete=state["last_collection"]["incomplete"], train=bool(games))
            if timed_out or stop_reason is not None:
                break
    if last_checkpoint is None:
        save_update([], "initial", incomplete=timed_out)
    summary = {"status": ("curriculum-deadline" if timed_out else
                           "curriculum-no-progress" if stop_reason is not None else "curriculum-not-promoted"),
               "checkpoint": last_checkpoint, "device": device, "updates": len(reports),
               "teacher_games": state["teacher_games"], "teacher_positions": state["teacher_positions"],
               "teacher_target_reached": state["teacher_positions"] >= args.teacher_positions,
               "mixed_games": state["mixed_games"], "mixed_target_reached": bool(args.total_games and
                   state["mixed_games"] >= args.total_games), "validation_positions": len(validation),
               "teacher_attempts": state["teacher_attempts"],
               "mixed_attempts": state["mixed_schedule_cursor"],
               "game_timeout_counts": state["game_timeout_counts"], "stop_reason": stop_reason,
               "unique_training_states_seen": len(partition.train_states),
               "unique_validation_states_seen": len(partition.validation_states),
               "replay_exposure": exposure.to_dict(),
               "excluded_positions": state["excluded_positions"],
               "imported_checkpoint_has_reserved_training": state["imported_checkpoint_has_reserved_training"],
               "seconds": time.perf_counter() - started, "promoted": False}
    write_json(args.output / "summary.json", summary)
    emit({"event": "complete", **summary})
    return summary
