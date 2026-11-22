from contextlib import nullcontext, redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from ghqnet.update_evaluation import (
    ARMS, arena_schedule, assert_unchanged, load_context, run_arena, save, sha,
    tactical_decision, tactical_summary, validate_config, validate_game, validate_gate,
)


def config():
    return {"threads": 2,
            "tactics": {"split": "dev", "verified_cases": 17, "legacy_cases": 10,
                        "time_ms": 2400, "simulations": 256, "seed": 4173, "max_seconds": 900},
            "arena": {"opening_count": 8, "time_ms": 2400, "max_turns": 100, "simulations": 100000,
                      "game_watchdog_seconds": 330, "max_seconds": 15000, "summary_seed": 913}}


def model_report():
    rows = []
    for index in range(22):
        for color in ("RED", "BLUE"):
            row = {"name": f"case{index}-{color.lower()}", "player": color,
                   "objective": "hq_capture" if index == 21 else "other", "passed": True,
                   "legal": True, "complete": True, "value_current_player": .25,
                   "elapsed_ms": 12., "timed_out": False}
            if index < 17:
                row["case_id"] = f"case{index}"
            rows.append(row)
    return {"complete": True, "cases": rows}


def game_result(task):
    return {"neural_color": task["neural_color"], "seed": task["seed"], "outcome": None,
            "unfinished": True, "terminal": None, "reason": "max-turns", "turns": 1,
            "trajectory": [{"player": "RED", "moves": ["skip"]}],
            "latency_ms": {"neural": {"mean": 1., "max": 1., "turns": 1}}}


class UpdateEvaluationTests(unittest.TestCase):
    def test_gate_uses_integrity_and_hq_not_advisory_retention(self):
        models = {"baseline": model_report(), "candidate": model_report()}
        for row in models["candidate"]["cases"]:
            if row["objective"] != "hq_capture":
                row["passed"] = False
        result = tactical_decision(models)
        self.assertTrue(result["eligible_for_arena"])
        self.assertEqual(result["advisory"]["dev_pass_count_delta"], -34)
        self.assertEqual(result["advisory"]["anchor_pass_count_delta"], -8)
        self.assertEqual(len(result["advisory"]["lost_baseline_anchors"]), 8)
        self.assertFalse(result["promoted"])

    def test_both_models_require_legal_complete_finite_and_hq_wins(self):
        for model in ("baseline", "candidate"):
            for key, value in (("legal", False), ("complete", False), ("value_current_player", float("nan")),
                               ("value_current_player", float("inf")), ("value_current_player", True),
                               ("elapsed_ms", -1)):
                with self.subTest(model=model, key=key, value=value):
                    models = {"baseline": model_report(), "candidate": model_report()}
                    models[model]["cases"][0][key] = value
                    self.assertFalse(tactical_decision(models)["eligible_for_arena"])
            models = {"baseline": model_report(), "candidate": model_report()}
            models[model]["cases"][-1]["passed"] = False
            self.assertFalse(tactical_decision(models)["eligible_for_arena"])

    def test_gate_rejects_missing_duplicate_and_unpaired_populations(self):
        mutations = [lambda r: r.update(complete=False), lambda r: r["cases"].pop(),
                     lambda r: r["cases"].__setitem__(0, deepcopy(r["cases"][1])),
                     lambda r: r["cases"][0].update(player="BLUE"),
                     lambda r: r["cases"][-1].update(player="RED")]
        for mutation in mutations:
            report = model_report()
            mutation(report)
            with self.assertRaises(ValueError):
                tactical_summary(report)
        models = {"baseline": model_report(), "candidate": model_report()}
        models["candidate"]["cases"][0]["name"] = "different"
        with self.assertRaisesRegex(ValueError, "different tactical cases"):
            tactical_decision(models)

    def test_schedule_freezes_three_arms_and_paired_colors_per_opening(self):
        openings = [{"id": f"opening-{index}", "seed": 100 + index} for index in range(8)]
        schedule = arena_schedule(openings)
        self.assertEqual(schedule, arena_schedule(openings))
        self.assertEqual(len(schedule), 48)
        self.assertEqual(len({(row["arm"], row["opening_id"], row["neural_color"]) for row in schedule}), 48)
        self.assertEqual([row["neural_color"] for row in schedule[:2]], ["RED", "BLUE"])
        self.assertEqual([row["neural_color"] for row in schedule[6:8]], ["BLUE", "RED"])
        for opening in openings:
            for arm, _, _ in ARMS:
                selected = [row for row in schedule if row["arm"] == arm and row["opening_id"] == opening["id"]]
                self.assertEqual({row["neural_color"] for row in selected}, {"RED", "BLUE"})
                self.assertEqual({row["seed"] for row in selected}, {opening["seed"]})

    def test_config_rejects_broadened_budgets_or_test_split(self):
        validate_config(config())
        for phase, key, value in (("arena", "opening_count", 16), ("arena", "max_seconds", 15001),
                                  ("arena", "game_watchdog_seconds", 331), ("arena", "time_ms", 2401),
                                  ("tactics", "split", "test"), ("tactics", "max_seconds", float("inf")),
                                  ("tactics", "verified_cases", 18)):
            plan = config()
            plan[phase][key] = value
            with self.assertRaises(ValueError):
                validate_config(plan)

    def test_frozen_inputs_and_sources_are_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            baseline, candidate, suite, manifest, source = [folder / name for name in
                                                           ("base.pt", "new.pt", "suite.jsonl", "openings.json", "source.py")]
            for path in (baseline, candidate, suite, manifest, source):
                path.write_text(path.name)
            plan = config()
            plan["baseline"] = {"path": str(baseline), "sha256": sha(baseline)}
            plan["source_sha256"] = {str(source): sha(source)}
            plan["tactics"].update(suite=str(suite), suite_sha256=sha(suite))
            plan["arena"].update(manifest=str(manifest), manifest_sha256=sha(manifest))
            plan_path = folder / "plan.json"
            save(plan_path, plan)
            _, identity = load_context(plan_path, candidate)
            assert_unchanged(identity)
            candidate.write_text("modified")
            with self.assertRaisesRegex(ValueError, "changed during evaluation"):
                assert_unchanged(identity)
            source.write_text("modified")
            with self.assertRaisesRegex(ValueError, "Frozen input/source changed"):
                load_context(plan_path, candidate)

    def test_gate_recomputed_from_pinned_model_reports(self):
        with tempfile.TemporaryDirectory() as directory:
            folder = Path(directory)
            identity = {"candidate_checkpoint_sha256": "candidate", "plan_sha256": "plan"}
            models = {name: dict(model_report(), identity=identity, model=name) for name in ("baseline", "candidate")}
            reports = {}
            for name, report in models.items():
                path = folder / f"{name}.json"
                save(path, report)
                reports[name] = {"path": str(path), "sha256": sha(path)}
            gate = dict(tactical_decision(models), identity=identity, model_reports=reports)
            gate_path = folder / "gate.json"
            save(gate_path, gate)
            self.assertEqual(validate_gate(gate_path, identity)[0], gate)
            gate["advisory"]["dev_pass_count_delta"] = 1
            save(gate_path, gate)
            with self.assertRaisesRegex(ValueError, "does not match"):
                validate_gate(gate_path, identity)
            gate["advisory"]["dev_pass_count_delta"] = 0
            save(gate_path, gate)
            Path(reports["candidate"]["path"]).write_text("{}")
            with self.assertRaisesRegex(ValueError, "report changed"):
                validate_gate(gate_path, identity)

    def test_cutoffs_remain_unfinished_and_nonfinite_latencies_fail(self):
        task = {"neural_color": "RED", "seed": 17}
        result = game_result(task)
        validate_game(result, task, 100)
        for key, value in (("outcome", "draw"), ("unfinished", False), ("turns", 101), ("seed", 18)):
            changed = deepcopy(result)
            changed[key] = value
            with self.assertRaises(ValueError):
                validate_game(changed, task, 100)
        result["latency_ms"]["neural"]["mean"] = float("nan")
        with self.assertRaisesRegex(ValueError, "latency"):
            validate_game(result, task, 100)

    def run_mock_arena(self, directory, *, fail=False, max_seconds=15000):
        from ghqnet.rules import e
        folder = Path(directory)
        gate_path = folder / "gate.json"
        gate_path.write_text("{}")
        manifest = folder / "manifest.json"
        manifest.write_text("{}")
        plan = config()
        plan["arena"].update(manifest=str(manifest), max_seconds=max_seconds)
        identity = {"baseline_checkpoint": "baseline", "candidate_checkpoint": "candidate", "tracked_sha256": {}}
        openings = [{"id": f"opening{index}", "seed": index, "state": e.BaseBoard().serialize()} for index in range(8)]

        def play(evaluator, **kwargs):
            if fail:
                raise TimeoutError("game watchdog")
            self.assertEqual(kwargs["time_ms"], 2400)
            self.assertEqual(kwargs["max_turns"], 100)
            self.assertEqual(kwargs["simulations"], 100000)
            return game_result({"neural_color": "RED" if kwargs["neural_color"] == e.RED else "BLUE", "seed": kwargs["seed"]})

        with patch("ghqnet.update_evaluation.load_context", return_value=(plan, identity)), \
                patch("ghqnet.update_evaluation.validate_gate", return_value=({}, sha(gate_path))), \
                patch("ghqnet.update_evaluation._evaluator", side_effect=lambda path: path), \
                patch("ghqnet.update_evaluation.watchdog", side_effect=lambda seconds: nullcontext()), \
                patch("ghqnet.arena.validate_openings", return_value=openings), \
                patch("ghqnet.arena.match", side_effect=play) as match, redirect_stdout(io.StringIO()):
            result = run_arena(folder / "plan.json", folder / "candidate.pt", folder / "arena", gate_path)
            return result, match.call_count

    def test_bounded_arena_executes_exactly_48_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            result, calls = self.run_mock_arena(directory)
            self.assertEqual(calls, 48)
            self.assertTrue(result["complete"])
            self.assertEqual(result["status"], "complete")
            self.assertFalse(result["promoted"])
            self.assertIsNone(result["active_phase"])
            self.assertTrue(result["inputs_unchanged"])
            self.assertEqual(len(result["games"]), 48)
            self.assertEqual(result["summaries"]["candidate-vs-baseline"]["totals"]["unfinished"], 16)
            with self.assertRaises(FileExistsError):
                self.run_mock_arena(directory)

    def test_budget_exhaustion_and_watchdog_preserve_no_fake_game(self):
        with tempfile.TemporaryDirectory() as directory:
            result, calls = self.run_mock_arena(directory, max_seconds=330)
            self.assertEqual(calls, 0)
            self.assertFalse(result["complete"])
            self.assertEqual(result["status"], "phase-budget-reached")
            self.assertEqual(result["games"], [])
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(TimeoutError):
                self.run_mock_arena(directory, fail=True)
            result = json.loads((Path(directory) / "arena/report.json").read_text())
            self.assertEqual(result["status"], "failed")
            self.assertFalse(result["complete"])
            self.assertEqual(result["games"], [])
            self.assertEqual(len(result["failures"]), 1)
            self.assertIsNone(result["active_game"])


if __name__ == "__main__":
    unittest.main()
