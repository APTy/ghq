"""Bounded tactical and paired-game checks for one preselected update.

The caller freezes the plan and candidate before these checks. Tactical
development scores are advisory: only legality, complete/finite plans, and
immediate HQ wins gate the arena. Neither command selects or promotes a model.
Outputs are fresh-only; an interrupted experiment is never silently resumed.
"""
import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import signal
import time


ROOT = Path(__file__).resolve().parents[1]
ARMS = (("candidate-vs-baseline", "candidate", "baseline"),
        ("candidate-vs-cpu", "candidate", None),
        ("baseline-vs-cpu", "baseline", None))


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1048576), b""):
            digest.update(block)
    return digest.hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def finite_number(value):
    return type(value) in (int, float) and math.isfinite(value)


def validate_config(plan):
    """Reject expanded or malformed budgets before loading models."""
    if plan.get("threads") != 2:
        raise ValueError("Evaluation requires two CPU Torch threads")
    for phase in ("tactics", "arena"):
        config = plan[phase]
        if config.get("time_ms") != 2400:
            raise ValueError("Both sides must receive 2400 ms per whole turn")
        if not finite_number(config.get("max_seconds")) or config["max_seconds"] <= 0:
            raise ValueError("Phase budget must be finite and positive")
    tactics, arena = plan["tactics"], plan["arena"]
    if (tactics.get("split") != "dev" or tactics.get("verified_cases") != 17 or
            tactics.get("legacy_cases") != 10 or tactics.get("simulations") != 256 or
            tactics["max_seconds"] > 900 or type(tactics.get("seed")) is not int):
        raise ValueError("Unexpected tactical panel, search setting, or budget")
    if (arena.get("opening_count") != 8 or arena.get("max_turns") != 100 or
            arena.get("simulations") != 100000 or arena["max_seconds"] > 15000 or
            not finite_number(arena.get("game_watchdog_seconds")) or
            not 0 < arena["game_watchdog_seconds"] <= 330 or
            type(arena.get("summary_seed")) is not int):
        raise ValueError("Unexpected arena population, search setting, or budget")


def load_context(plan_path, candidate_path):
    plan_path, candidate_path = Path(plan_path).resolve(), Path(candidate_path).resolve()
    plan = json.loads(plan_path.read_text())
    validate_config(plan)
    tracked = {}

    def track(path, expected=None):
        path = Path(path).resolve()
        actual = sha(path)
        if expected is not None and actual != expected:
            raise ValueError(f"Frozen input/source changed: {path}")
        if str(path) in tracked and tracked[str(path)] != actual:
            raise ValueError("Conflicting frozen input identity")
        tracked[str(path)] = actual
        return actual

    plan_sha = track(plan_path)
    candidate_sha = track(candidate_path)
    baseline = Path(plan["baseline"]["path"]).resolve()
    track(baseline, plan["baseline"]["sha256"])
    track(plan["tactics"]["suite"], plan["tactics"]["suite_sha256"])
    track(plan["arena"]["manifest"], plan["arena"]["manifest_sha256"])
    if not plan.get("source_sha256"):
        raise ValueError("Plan must freeze evaluation sources")
    for relative, expected in plan["source_sha256"].items():
        track(ROOT / relative, expected)
    track(__file__)
    identity = {"plan": str(plan_path), "plan_sha256": plan_sha,
                "candidate_checkpoint": str(candidate_path), "candidate_checkpoint_sha256": candidate_sha,
                "baseline_checkpoint": str(baseline), "baseline_checkpoint_sha256": sha(baseline),
                "tracked_sha256": tracked}
    return plan, identity


def assert_unchanged(identity):
    for path, expected in identity["tracked_sha256"].items():
        if sha(path) != expected:
            raise ValueError(f"Frozen input/source changed during evaluation: {path}")


@contextmanager
def watchdog(seconds):
    """Restore the caller's handler/timer even when evaluation raises."""
    if seconds <= 0:
        raise TimeoutError("Evaluation budget exhausted")
    previous_handler = signal.getsignal(signal.SIGALRM)
    previous_timer = signal.getitimer(signal.ITIMER_REAL)
    started = time.monotonic()

    def expired(signum, frame):
        raise TimeoutError("Evaluation watchdog expired; interrupted work is not a draw")

    signal.signal(signal.SIGALRM, expired)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous_handler)
        if previous_timer[0] > 0:
            signal.setitimer(signal.ITIMER_REAL,
                            max(0.000001, previous_timer[0] - (time.monotonic() - started)), previous_timer[1])


def tactical_summary(report):
    rows = report.get("cases", [])
    if (report.get("complete") is not True or len(rows) != 44 or
            len({row["name"] for row in rows}) != 44):
        raise ValueError("Require 44 distinct completed tactical color runs")
    dev = [row for row in rows if "case_id" in row]
    anchors = [row for row in rows if "case_id" not in row]
    if len(dev) != 34 or len(anchors) != 10:
        raise ValueError("Wrong tactical development/anchor populations")
    groups = {}
    for row in dev:
        groups.setdefault(row["case_id"], []).append(row)
    if len(groups) != 17 or any(len(group) != 2 or {row["player"] for row in group} != {"RED", "BLUE"}
                                for group in groups.values()):
        raise ValueError("Development cases must have exactly both colors")
    hq = [row for row in anchors if row["objective"] == "hq_capture"]
    if len(hq) != 2 or {row["player"] for row in hq} != {"RED", "BLUE"}:
        raise ValueError("Require both immediate HQ anchor colors")
    return {
        "all_legal_complete_finite": all(row.get("legal") is True and row.get("complete") is True and
                                          finite_number(row.get("value_current_player")) and
                                          finite_number(row.get("elapsed_ms")) and row["elapsed_ms"] >= 0
                                          for row in rows),
        "immediate_hq_both_pass": all(row.get("passed") is True for row in hq),
        "dev_passed": sum(row.get("passed") is True for row in dev),
        "anchors_passed": sum(row.get("passed") is True for row in anchors),
        "anchor_results": {row["name"]: row.get("passed") is True for row in anchors},
        "dev_results": {row["name"]: row.get("passed") is True for row in dev},
        "timed_out_color_runs": sum(row.get("timed_out") is True for row in rows),
    }


def tactical_decision(models):
    if set(models) != {"baseline", "candidate"}:
        raise ValueError("Require exactly baseline and preselected candidate")
    summaries = {name: tactical_summary(report) for name, report in models.items()}
    baseline, candidate = summaries["baseline"], summaries["candidate"]
    for panel in ("anchor_results", "dev_results"):
        if set(baseline[panel]) != set(candidate[panel]):
            raise ValueError("Models were evaluated on different tactical cases")
    return {"complete": True,
            "eligible_for_arena": all(summary["all_legal_complete_finite"] and summary["immediate_hq_both_pass"]
                                      for summary in summaries.values()),
            "models": summaries,
            "advisory": {"dev_pass_count_delta": candidate["dev_passed"] - baseline["dev_passed"],
                         "anchor_pass_count_delta": candidate["anchors_passed"] - baseline["anchors_passed"],
                         "lost_baseline_anchors": sorted(name for name, passed in baseline["anchor_results"].items()
                                                         if passed and not candidate["anchor_results"][name])},
            "promoted": False,
            "note": "Development/legacy retention is advisory. This gate tests executable plans and immediate HQ wins; matches determine strength."}


def _evaluator(path):
    from .model import TorchEvaluator, load_checkpoint
    from .rules import e
    model, _ = load_checkpoint(Path(path), "cpu")
    evaluator = TorchEvaluator(model, "cpu")
    board = e.BaseBoard()
    evaluator.predict(board, list(board.generate_legal_moves()))
    return evaluator


def run_tactics(plan_path, candidate_path, output):
    plan, identity = load_context(plan_path, candidate_path)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = plan["tactics"]
    started = time.monotonic()
    execution = {"identity": identity, "created_at": now(), "completed_at": None,
                 "status": "running", "active_phase": "tactics", "complete": False,
                 "promoted": False, "models": {}, "failures": []}

    def publish():
        execution["seconds"] = time.monotonic() - started
        save(output / "execution.json", execution)

    publish()
    try:
        with watchdog(config["max_seconds"]):
            import torch
            from .rules import e
            from .suite import read_suite, canonical_state_id
            from .tactics import tactical_cases, run_case, run_frozen_case, frozen_results_summary
            torch.set_num_threads(plan["threads"])
            _, frozen = read_suite(Path(config["suite"]), split=config["split"])
            frozen = [case for case in frozen if case["evidence"] == "verified_witness"]
            legacy = tactical_cases()
            if len(frozen) != config["verified_cases"] or len(legacy) != config["legacy_cases"]:
                raise ValueError("Frozen tactical case counts changed")
            models, model_reports = {}, {}
            kwargs = {"time_ms": config["time_ms"], "simulations": config["simulations"], "seed": config["seed"]}
            for name in ("baseline", "candidate"):
                assert_unchanged(identity)
                execution["active_model"] = name
                publish()
                evaluator = _evaluator(identity[name + "_checkpoint"])
                report = {"identity": identity, "model": name, "cases": [], "complete": False,
                          "created_at": now(), "completed_at": None, "promoted": False}

                def tasks():
                    for case in legacy:
                        row = run_case(case, evaluator, **kwargs)
                        row.update(category="reported_regression", canonical_state_id=canonical_state_id(case.board))
                        yield row
                    for case in frozen:
                        for mirrored in (False, True):
                            yield run_frozen_case(case, evaluator, mirrored=mirrored, **kwargs)

                for row in tasks():
                    # A non-finite result cannot be published as valid JSON or silently accepted.
                    if not finite_number(row.get("value_current_player")) or not finite_number(row.get("elapsed_ms")):
                        raise ValueError(f"Non-finite tactical result: {name}/{row.get('name')}")
                    report["cases"].append(row)
                    report["complete"] = len(report["cases"]) == 44
                    report["passed"] = sum(item.get("passed") is True for item in report["cases"])
                    report["frozen_results"] = frozen_results_summary(report["cases"])
                    if report["complete"]:
                        report["completed_at"] = now()
                    destination = output / (name + ".json")
                    save(destination, report)
                    execution["models"][name] = {"cases": len(report["cases"]), "passed": report["passed"],
                                                  "complete": report["complete"]}
                    publish()
                    print(json.dumps({"phase": "tactics", "model": name, "cases": len(report["cases"]),
                                      "case": row["name"], "passed": row["passed"]}), flush=True)
                assert_unchanged(identity)
                models[name] = report
                model_reports[name] = {"path": str(destination), "sha256": sha(destination)}
                del evaluator
            decision = tactical_decision(models)
            decision.update(identity=identity, model_reports=model_reports, created_at=now())
            save(output / "decision.json", decision)
            execution.update(complete=True, status="complete", eligible_for_arena=decision["eligible_for_arena"],
                             inputs_unchanged=True)
    except Exception as error:
        execution["status"] = "budget-reached" if isinstance(error, TimeoutError) else "failed"
        execution["failures"].append({"error": f"{type(error).__name__}: {error}", "at": now()})
        raise
    finally:
        execution.update(active_phase=None, active_model=None, completed_at=now())
        publish()
    return execution


def validate_gate(gate_path, identity):
    gate_path = Path(gate_path).resolve()
    gate = json.loads(gate_path.read_text())
    if gate.get("identity") != identity or gate.get("eligible_for_arena") is not True:
        raise ValueError("Tactical gate failed or belongs to a different candidate/plan")
    models = {}
    for name, metadata in gate["model_reports"].items():
        if sha(metadata["path"]) != metadata["sha256"]:
            raise ValueError("Tactical model report changed")
        models[name] = json.loads(Path(metadata["path"]).read_text())
        if models[name].get("identity") != identity or models[name].get("model") != name:
            raise ValueError("Tactical model report identity mismatch")
    recomputed = tactical_decision(models)
    if any(gate.get(key) != value for key, value in recomputed.items()):
        raise ValueError("Tactical gate does not match its model reports")
    return gate, sha(gate_path)


def arena_schedule(openings):
    schedule = []
    for index, opening in enumerate(openings):
        colors = ("RED", "BLUE") if index % 2 == 0 else ("BLUE", "RED")
        for arm, player, opponent in ARMS:
            for color in colors:
                schedule.append({"arm": arm, "player": player, "opponent": opponent,
                                 "opening_id": opening["id"], "neural_color": color, "seed": opening["seed"]})
    return schedule


def validate_game(result, task, max_turns):
    if (result.get("neural_color") != task["neural_color"] or result.get("seed") != task["seed"] or
            result.get("outcome") not in ("win", "loss", "draw", None) or
            type(result.get("unfinished")) is not bool or
            result["unfinished"] != (result["outcome"] is None) or
            result["unfinished"] != (result.get("terminal") is None) or
            type(result.get("turns")) is not int or not 0 <= result["turns"] <= max_turns or
            result["turns"] != len(result.get("trajectory", []))):
        raise ValueError("Arena result identity, outcome, or turn count is invalid")
    for side in result.get("latency_ms", {}).values():
        if any(not finite_number(side.get(key)) or side[key] < 0 for key in ("mean", "max", "turns")):
            raise ValueError("Non-finite or negative arena latency")


def run_arena(plan_path, candidate_path, output, gate_report):
    plan, identity = load_context(plan_path, candidate_path)
    gate, gate_sha = validate_gate(gate_report, identity)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    config = plan["arena"]
    started = time.monotonic()
    report = {"identity": identity, "gate_report": str(Path(gate_report).resolve()), "gate_report_sha256": gate_sha,
              "created_at": now(), "completed_at": None, "status": "running", "active_phase": "arena",
              "games": [], "failures": [], "complete": False, "promoted": False,
              "settings": config, "device": "cpu", "threads": plan["threads"], "cpu_depth": 3, "cpu_beam": 8,
              "note": "Bounded 48-game exploratory screen; no automatic extension or promotion."}

    def publish():
        report["seconds"] = time.monotonic() - started
        save(output / "report.json", report)

    def remaining():
        return config["max_seconds"] - (time.monotonic() - started)

    publish()
    active_task = None
    try:
        with watchdog(remaining()):
            import torch
            from .arena import match, summarize_matches, validate_openings
            from .rules import e
            openings = validate_openings(json.loads(Path(config["manifest"]).read_text()))
            if len(openings) != config["opening_count"]:
                raise ValueError("Frozen arena opening count changed")
            by_id = {opening["id"]: opening for opening in openings}
            report["schedule"] = arena_schedule(openings)
            report["expected_games"] = len(report["schedule"])
            torch.set_num_threads(plan["threads"])
            evaluators = {name: _evaluator(identity[name + "_checkpoint"]) for name in ("baseline", "candidate")}
        for task in report["schedule"]:
            active_task = task
            if remaining() < config["game_watchdog_seconds"]:
                report["status"] = "phase-budget-reached"
                return report
            assert_unchanged(identity)
            # Keep the tactical evidence pinned throughout a long-running match phase.
            if sha(gate_report) != gate_sha:
                raise ValueError("Tactical gate changed during arena")
            report["active_game"] = task
            publish()
            opening = by_id[task["opening_id"]]
            with watchdog(config["game_watchdog_seconds"]):
                result = match(evaluators[task["player"]], neural_color=e.RED if task["neural_color"] == "RED" else e.BLUE,
                               seed=task["seed"], time_ms=config["time_ms"], max_turns=config["max_turns"],
                               simulations=config["simulations"], start_board=e.BaseBoard.deserialize(opening["state"]),
                               opponent_evaluator=evaluators[task["opponent"]] if task["opponent"] else None)
            validate_game(result, task, config["max_turns"])
            assert_unchanged(identity)
            result.update(arm=task["arm"], opening_id=task["opening_id"])
            report["games"].append(result)
            report["summaries"] = {arm: summarize_matches([game for game in report["games"] if game["arm"] == arm],
                                                          seed=config["summary_seed"]) for arm, _, _ in ARMS}
            publish()
            print(json.dumps({"phase": "arena", "games": len(report["games"]), "arm": task["arm"],
                              "outcome": result["outcome"], "turns": result["turns"]}), flush=True)
        validate_gate(gate_report, identity)
        assert_unchanged(identity)
        report.update(complete=True, status="complete", inputs_unchanged=True)
    except Exception as error:
        report["status"] = "failed"
        report["failures"].append({"task": active_task, "error": f"{type(error).__name__}: {error}", "at": now()})
        raise
    finally:
        report.update(active_phase=None, active_game=None, completed_at=now())
        publish()
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phase", choices=("tactics", "arena"))
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--gate-report", type=Path)
    args = parser.parse_args(argv)
    if args.phase == "arena" and args.gate_report is None:
        parser.error("Arena requires --gate-report from the fixed candidate's tactics")
    if args.phase == "tactics" and args.gate_report is not None:
        parser.error("Tactics does not consume a gate report")
    if args.phase == "tactics":
        return run_tactics(args.plan, args.candidate, args.output)
    return run_arena(args.plan, args.candidate, args.output, args.gate_report)


if __name__ == "__main__":
    result = main()
    raise SystemExit(0 if result["complete"] else 2)
