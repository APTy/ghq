#!/usr/bin/env python3
"""Fetch one read-only snapshot of a GHQ cloud training run, without polling.

Only `gcloud storage cat` is used. Local fetch times are not remote modification
times or proof that a VM is alive. Worker uploads can lag active training.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import subprocess
import tempfile

OBJECTS = ("job-config.json", "hardware.json", "run/events.jsonl", "run/summary.json",
           "completion.json", "upload-failed.json")
MAX_OBJECT_BYTES = 64 * 1024 ** 2


def utc_now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def validate_location(project, run_uri):
    if not re.fullmatch(r"[a-z][a-z0-9-]{4,28}[a-z0-9]", project):
        raise ValueError("Project must be a Google Cloud project ID (6–30 lowercase letters/digits/hyphens)")
    run_uri = run_uri.rstrip("/")
    if not re.fullmatch(r"gs://[a-z0-9][a-z0-9._-]{1,220}[a-z0-9]/runs/[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?", run_uri):
        raise ValueError("Run URI must be exactly gs://BUCKET/runs/INSTANCE, without wildcards or extra path segments")
    return run_uri


def _number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0


def _sum_known(rows, key):
    values = [row.get(key) for row in rows]
    return sum(values) if all(_number(value) for value in values) else None


def _arguments(config):
    result, values = {}, config.get("pipeline_args", []) if isinstance(config, dict) else []
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        return result
    index = 0
    while index < len(values):
        token = values[index]
        index += 1
        if not token.startswith("--"):
            continue
        key, separator, value = token[2:].partition("=")
        if not separator:
            value = values[index] if index < len(values) and not values[index].startswith("--") else True
            index += value is not True
        result[key.replace("-", "_")] = value
    return result


def parse_events(raw):
    events, malformed = [], []
    lines = raw.splitlines()
    for number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
            if not isinstance(event, dict) or not isinstance(event.get("event"), str):
                raise ValueError("Event must be an object with an event name")
            events.append(event)
        except (ValueError, TypeError):
            malformed.append(number)
    return events, {"parsed_events": len(events), "malformed_lines": malformed,
                    "partial_tail": bool(malformed and malformed[-1] == len(lines)),
                    "counts_are_observed_only": True}


def _checkpoint_metrics_reference(events):
    """Use only the latest saved-checkpoint event, never its directory as a URL."""
    for event in reversed(events or []):
        if event["event"] not in ("trained", "checkpoint-saved"):
            continue
        checkpoint = event.get("checkpoint")
        if not isinstance(checkpoint, str):
            return None
        match = re.fullmatch(r"/?(?:[A-Za-z0-9_.-]+/)*checkpoint-([0-9]{4,9})\.pt", checkpoint)
        if not match or any(part in (".", "..") for part in checkpoint.split("/")):
            return None
        digits = match.group(1)
        iteration = int(digits)
        if digits != f"{iteration:04d}":
            return None
        if "iteration" in event and (type(event["iteration"]) is not int or event["iteration"] != iteration):
            return None
        return {"checkpoint": checkpoint, "iteration": iteration, "object": f"run/metrics-{digits}.json"}
    return None


def _game_counts(games):
    terminal, hq, draws, cutoffs, unknown = 0, 0, 0, 0, 0
    reasons = Counter()
    for game in games:
        outcome = game.get("terminal")
        if isinstance(outcome, dict) and "winner" in outcome and outcome["winner"] in ("RED", "BLUE", None):
            terminal += 1
            hq += outcome.get("reason") == "hq-capture" and outcome["winner"] is not None
            draws += outcome["winner"] is None
            reasons[str(outcome.get("reason", "unknown"))] += 1
        elif game.get("truncated") is True:
            cutoffs += 1
            reasons[str(game.get("reason", "unknown-cutoff"))] += 1
        else:
            unknown += 1
            reasons["unknown"] += 1
    return {"finished_game_tasks": len(games), "sampled_positions": _sum_known(games, "samples"),
            "known_sampled_positions": sum(game["samples"] for game in games if _number(game.get("samples"))),
            "games_missing_sample_counts": sum(not _number(game.get("samples")) for game in games),
            "terminal_games": terminal, "hq_captures": hq, "draws": draws, "cutoffs": cutoffs,
            "unknown_outcomes": unknown, "reasons": dict(sorted(reasons.items()))}


def summarize(parsed, fetches, *, project, run_uri, started_at, finished_at, snapshot_directory):
    config = parsed.get("job-config.json")
    hardware = parsed.get("hardware.json")
    summary = parsed.get("run/summary.json")
    summary_source = {"kind": "pipeline_summary", "object": "run/summary.json"} if isinstance(summary, dict) else None
    completion = parsed.get("completion.json")
    failed_upload = parsed.get("upload-failed.json")
    event_blob = parsed.get("run/events.jsonl")
    events, event_quality = parse_events(event_blob) if isinstance(event_blob, str) else (None, None)
    if events == [] and event_quality["malformed_lines"]:
        events = None
    arguments = _arguments(config)
    targets = {}
    for key in ("teacher_positions", "teacher_max_games", "teacher_games", "total_games", "iterations", "games",
                "workers", "inference_batch_size"):
        value = arguments.get(key)
        try:
            targets[key] = int(value) if value is not None and value is not True and int(value) >= 0 else None
        except (ValueError, TypeError):
            targets[key] = None

    groups, outcomes, batching, latest_checkpoint = None, None, None, None
    collection_start, timeout_events = None, []
    phase, phase_source = "unknown", None
    if events is not None:
        games, duplicates = {}, 0
        for index, event in enumerate(events):
            if event["event"] == "game":
                key = event.get("game_id") or f"unidentified-event-{index}"
                duplicates += key in games
                games[key] = event
            if event["event"] in ("trained", "checkpoint-saved") and isinstance(event.get("checkpoint"), str):
                latest_checkpoint = event["checkpoint"]
        event_quality["duplicate_game_events"] = duplicates
        groups = {name: _game_counts([game for game in games.values() if game.get("teacher") is teacher])
                  for name, teacher in (("teacher", True), ("mixed", False))}
        groups["unknown_phase"] = _game_counts([game for game in games.values() if not isinstance(game.get("teacher"), bool)])
        outcomes = _game_counts(list(games.values()))
        collections = [event for event in events if event["event"] == "collection"]
        starts = [event for event in events if event["event"] == "collection-start"]
        collection_start = starts[-1] if starts else None
        timeout_events = [event for event in events if event["event"] == "game-timeout"]
        if collections:
            rows, batches, seconds = (_sum_known(collections, key) for key in ("rows", "batches", "evaluator_seconds"))
            latest = collections[-1]
            batching = {"observed_collections": len(collections), "rows": rows, "batches": batches,
                        "mean_batch_size": rows / batches if rows is not None and batches else None,
                        "evaluator_rows_per_second": rows / seconds if rows is not None and seconds else None,
                        "latest": {key: latest.get(key) for key in ("teacher", "workers", "requested_batch_size", "completed_games",
                                   "cancelled_games", "timed_out_games", "game_timeouts", "stop_reason", "failure",
                                   "incomplete", "mean_batch_size", "batch_latency_ms", "request_latency_ms")}}
        if events:
            latest = events[-1]
            phase_source = "last_event"
            if latest["event"] in ("game", "collection"):
                prefix = "teacher" if latest.get("teacher") is True else "mixed" if latest.get("teacher") is False else "unknown"
                phase = prefix + ("_game_recorded" if latest["event"] == "game" else "_collection_recorded")
            elif latest["event"] in ("trained", "checkpoint-saved"):
                phase = str(latest.get("phase", "training")) + "_checkpoint_saved"
            else:
                phase = "last_reported_" + latest["event"]
            if latest["event"] in ("complete", "failed") and summary is None:
                summary = latest
                summary_source = {"kind": "pipeline_summary_event", "event": latest["event"]}
    if isinstance(summary, dict):
        # A summary is emitted at pipeline completion, but does not confirm that
        # the final checkpoint upload succeeded or the VM has stopped.
        phase, phase_source = "last_reported_" + str(summary.get("status", "summary")), "pipeline_summary"
        if isinstance(summary.get("checkpoint"), str):
            latest_checkpoint = summary["checkpoint"]

    run_id = run_uri.rsplit("/", 1)[-1]
    identity_conflicts = []
    for name, value in (("job-config.json", config), ("completion.json", completion), ("upload-failed.json", failed_upload)):
        if isinstance(value, dict) and value.get("run_id") not in (None, run_id):
            identity_conflicts.append(name)
    upload = {"status": "unknown", "final_sync_succeeded": None, "exit_code": None, "finished_at_remote": None}
    if isinstance(completion, dict) and "completion.json" not in identity_conflicts:
        confirmed = completion.get("final_sync_succeeded")
        upload.update(status="confirmed" if confirmed is True else "failed" if confirmed is False else "unknown",
                      final_sync_succeeded=confirmed if isinstance(confirmed, bool) else None,
                      exit_code=completion.get("exit_code"), finished_at_remote=completion.get("finished_at"))
        phase = "worker_completed" if completion.get("exit_code") == 0 else "worker_failed" if isinstance(completion.get("exit_code"), int) else "worker_completion_recorded"
        phase_source = "completion_object"
    elif isinstance(failed_upload, dict) and "upload-failed.json" not in identity_conflicts:
        confirmed_failure = failed_upload.get("final_sync_succeeded") is False
        upload.update(status="failed" if confirmed_failure else "unknown", final_sync_succeeded=False if confirmed_failure else None,
                      exit_code=failed_upload.get("exit_code"),
                      finished_at_remote=failed_upload.get("finished_at"))
        phase, phase_source = "final_upload_failed" if confirmed_failure else "upload_failure_recorded_status_unknown", "upload_failed_object"
    cumulative_values, cumulative_source = summary, summary_source
    reference = _checkpoint_metrics_reference(events)
    if not isinstance(summary, dict) and reference is not None:
        metrics = parsed.get(reference["object"])
        if isinstance(metrics, dict) and type(metrics.get("iteration")) is int and metrics["iteration"] == reference["iteration"]:
            cumulative_values = metrics
            cumulative_source = {"kind": "latest_saved_checkpoint_metrics", **reference,
                                 "scope": "At this saved checkpoint only; excludes subsequent in-flight collection and training."}
    cumulative = {key: cumulative_values.get(key) if isinstance(cumulative_values, dict) and _number(cumulative_values.get(key)) else None
                  for key in ("teacher_games", "teacher_positions", "mixed_games", "validation_positions")}
    bootstrap = cumulative_values.get("bootstrap_target_reached") if isinstance(cumulative_values, dict) else None
    cumulative["bootstrap_target_reached"] = bootstrap if isinstance(bootstrap, bool) else None
    return {"project": project, "run_uri": run_uri, "local_fetch_started_at": started_at,
            "local_fetch_finished_at": finished_at, "snapshot_directory": str(snapshot_directory),
            "freshness": "Local fetch time only; remote snapshot age and VM liveness are unknown. Uploads may lag training.",
            "phase": phase, "phase_source": phase_source, "latest_checkpoint_reported": latest_checkpoint,
            "pipeline_status": summary.get("status") if isinstance(summary, dict) else None,
            "pipeline_stop_reason": summary.get("stop_reason") if isinstance(summary, dict) else None,
            "last_collection_start": collection_start,
            "game_timeout_events": timeout_events,
            "event_count_scope": "Events in the current run prefix; cumulative resume totals come from an explicit pipeline summary or saved-checkpoint metrics.",
            "configured_targets": targets, "event_counts": groups, "all_event_outcomes": outcomes,
            "reported_cumulative": cumulative, "reported_cumulative_source": cumulative_source,
            "position_count_note": "Game samples are collected positions, not retained teacher-training positions; cumulative retained progress requires a summary or saved-checkpoint metrics and excludes later in-flight work.",
            "event_quality": event_quality, "batching": batching, "final_upload": upload,
            "hardware": ({"cpu_count": hardware.get("cpu_count"), "gpu": hardware.get("torch_cuda", {}).get("gpu")
                          if isinstance(hardware.get("torch_cuda"), dict) else None,
                          "gpu_query": hardware.get("gpu"), "collected_at_remote": hardware.get("collected_at")}
                         if isinstance(hardware, dict) else None),
            "identity_conflicts": identity_conflicts, "fetches": fetches}


def collect(project, run_uri, output, *, runner=None, timeout=20):
    run_uri = validate_location(project, run_uri)
    if not _number(timeout) or timeout <= 0 or timeout > 60:
        raise ValueError("Fetch timeout must be positive and at most 60 seconds")
    runner = runner or subprocess.run
    output = Path(output)
    (output / "snapshots").mkdir(parents=True, exist_ok=True)
    snapshot = Path(tempfile.mkdtemp(prefix=datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-"), dir=output / "snapshots"))
    started, parsed, fetches = utc_now(), {}, {}
    def fetch(name, expected_iteration=None):
        command = ["gcloud", "storage", "cat", run_uri + "/" + name, "--project=" + project, "--quiet"]
        info = {"status": "unknown", "local_fetched_at": None, "remote_modified_at": None}
        try:
            result = runner(command, capture_output=True, text=True, timeout=timeout, check=False)
            info["local_fetched_at"] = utc_now()
            if result.returncode != 0:
                error = (result.stderr or "").strip()
                missing = bool(re.search(
                    r"no urls matched|the following urls matched no objects or files:|no such object|notfoundexception|\b404\b",
                    error, re.I))
                info.update(status="missing" if missing else "error", error=error[:1200], exit_code=result.returncode)
            else:
                raw = result.stdout
                if len(raw.encode("utf-8")) > MAX_OBJECT_BYTES:
                    raise ValueError("Snapshot exceeds the 64 MiB local-read limit")
                local = snapshot / name
                local.parent.mkdir(parents=True, exist_ok=True)
                local.write_text(raw)
                info.update(status="ok", path=str(local), bytes=local.stat().st_size)
                if name.endswith(".jsonl"):
                    parsed[name] = raw
                else:
                    value = json.loads(raw)
                    if not isinstance(value, dict):
                        raise ValueError("Expected a JSON object")
                    if expected_iteration is not None and (type(value.get("iteration")) is not int or value["iteration"] != expected_iteration):
                        raise ValueError("Metrics iteration does not match the saved-checkpoint event")
                    parsed[name] = value
        except (OSError, subprocess.SubprocessError, ValueError, TypeError) as error:
            info.update(status="error", error=str(error)[:1200], local_fetched_at=utc_now())
        fetches[name] = info

    for name in OBJECTS:
        fetch(name)
    raw_events = parsed.get("run/events.jsonl")
    events = parse_events(raw_events)[0] if isinstance(raw_events, str) else []
    has_final_summary = isinstance(parsed.get("run/summary.json"), dict) or bool(events and events[-1]["event"] in ("complete", "failed"))
    reference = _checkpoint_metrics_reference(events)
    if not has_final_summary and reference is not None:
        fetch(reference["object"], expected_iteration=reference["iteration"])
    report = summarize(parsed, fetches, project=project, run_uri=run_uri, started_at=started,
                       finished_at=utc_now(), snapshot_directory=snapshot)
    raw = json.dumps(report, indent=2) + "\n"
    (snapshot / "status.json").write_text(raw)
    # Never consult an older snapshot on a failed fetch. The latest report
    # explicitly contains unknown values, while older raw files remain archived.
    with tempfile.NamedTemporaryFile(dir=output, prefix=".status-", suffix=".json.tmp", mode="w", delete=False) as target:
        target.write(raw)
        temporary = Path(target.name)
    try:
        temporary.replace(output / "status.json")
    finally:
        temporary.unlink(missing_ok=True)
    return report


def format_summary(report):
    def value(item):
        return "unknown" if item is None else str(item)
    lines = [f"Fetched {report['local_fetch_finished_at']} (local time; remote snapshots may lag).",
             f"Last reported phase: {report['phase']}"]
    if report.get("pipeline_status"):
        lines.append(f"Pipeline result: {report['pipeline_status']}; stop reason: {json.dumps(report.get('pipeline_stop_reason'))}.")
    for name in ("teacher", "mixed"):
        counts = (report["event_counts"] or {}).get(name, {})
        lines.append(f"{name.title()}: {value(counts.get('finished_game_tasks'))} finished tasks; {value(counts.get('sampled_positions'))} sampled positions (current-run events).")
    outcomes = report["all_event_outcomes"] or {}
    lines.append(f"Observed outcomes: HQ captures {value(outcomes.get('hq_captures'))}, draws {value(outcomes.get('draws'))}, cutoffs {value(outcomes.get('cutoffs'))}.")
    actual, targets = report["reported_cumulative"], report["configured_targets"]
    lines.append(f"Cumulative targets: retained teacher positions {value(actual['teacher_positions'])}/{value(targets['teacher_positions'])}; mixed games {value(actual['mixed_games'])}/{value(targets['total_games'])}.")
    source = report["reported_cumulative_source"]
    if source and source["kind"] == "latest_saved_checkpoint_metrics":
        lines.append(f"Cumulative source: latest saved checkpoint metrics (iteration {source['iteration']}); excludes later in-flight work.")
    elif source:
        lines.append("Cumulative source: final pipeline summary.")
    lines.append(f"Latest reported checkpoint: {value(report['latest_checkpoint_reported'])}; final upload: {report['final_upload']['status']}.")
    batching = report["batching"] or {}
    lines.append(f"Batching: {value(batching.get('batches'))} batches, mean size {value(batching.get('mean_batch_size'))}.")
    latest = batching.get("latest", {})
    if latest.get("stop_reason") is not None:
        lines.append(f"Last collection: {json.dumps(latest['stop_reason'])}; individual game timeouts: {value(latest.get('timed_out_games'))}.")
    states = Counter(info["status"] for info in report["fetches"].values())
    lines.append(f"Fetches: {dict(states)}. Snapshots: {report['snapshot_directory']}")
    if report["identity_conflicts"]:
        lines.append("Run identity mismatch: " + ", ".join(report["identity_conflicts"]))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True)
    parser.add_argument("--run-uri", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = collect(args.project, args.run_uri, args.output)
    except (ValueError, OSError) as error:
        parser.exit(1, f"{error}\n")
    print(format_summary(report))
    return 1 if any(info["status"] == "error" for info in report["fetches"].values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
