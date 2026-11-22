import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from training.gcp import status

URI = "gs://test-ghq-bucket/runs/ghq-train-example"


def mock_runner(payloads, calls):
    def run(command, **kwargs):
        calls.append((command, kwargs))
        name = command[3].removeprefix(URI + "/")
        value = payloads.get(name)
        if isinstance(value, BaseException):
            raise value
        if isinstance(value, subprocess.CompletedProcess):
            return value
        if value is None:
            return subprocess.CompletedProcess(command, 1, "", "ERROR: No URLs matched")
        text = value if isinstance(value, str) else json.dumps(value)
        return subprocess.CompletedProcess(command, 0, text, "")
    return run


def game(identity, teacher, samples, terminal=None, reason="max-turns"):
    return {"event": "game", "game_id": identity, "teacher": teacher, "samples": samples,
            "terminal": terminal, "truncated": terminal is None, "reason": reason}


class StatusTests(unittest.TestCase):
    def test_read_only_commands_snapshot_and_targets_distinguish_collected_from_retained(self):
        events = [game("teacher-a", True, 12), game("mixed-a", False, 7, {"winner": "RED", "reason": "hq-capture"}),
                  game("mixed-b", False, 3, {"winner": None, "reason": "double-skip"}),
                  {"event": "collection", "teacher": False, "rows": 100, "batches": 25, "evaluator_seconds": 2,
                   "mean_batch_size": 4, "completed_games": 2, "workers": 8},
                  {"event": "trained", "phase": "mixed", "checkpoint": "/output/run/checkpoint-0002.pt",
                   "samples": 100, "validation_samples": 12}]
        payloads = {"job-config.json": {"run_id": "ghq-train-example", "pipeline_args":
                    ["--curriculum", "--teacher-positions", "50000", "--total-games=2000", "--workers", "8"]},
                    "hardware.json": {"cpu_count": 16, "torch_cuda": {"gpu": "Tesla T4"}},
                    "run/events.jsonl": "\n".join(json.dumps(event) for event in events) + "\n"}
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            report = status.collect("play-ghq", URI, folder, runner=mock_runner(payloads, calls))
            self.assertEqual(json.loads((Path(folder) / "status.json").read_text()), report)
            self.assertTrue((Path(report["snapshot_directory"]) / "run/events.jsonl").is_file())
            self.assertEqual(len(calls), len(status.OBJECTS) + 1)
            for command, kwargs in calls:
                self.assertEqual(command[:3], ["gcloud", "storage", "cat"])
                self.assertIn("--project=play-ghq", command)
                self.assertFalse(kwargs["check"])
            self.assertEqual(report["event_counts"]["teacher"]["sampled_positions"], 12)
            self.assertEqual(report["event_counts"]["mixed"]["finished_game_tasks"], 2)
            self.assertIsNone(report["reported_cumulative"]["teacher_positions"])
            self.assertIsNone(report["reported_cumulative"]["mixed_games"])
            self.assertEqual(report["configured_targets"]["teacher_positions"], 50000)
            self.assertEqual(report["all_event_outcomes"]["hq_captures"], 1)
            self.assertEqual(report["all_event_outcomes"]["draws"], 1)
            self.assertEqual(report["all_event_outcomes"]["cutoffs"], 1)
            self.assertEqual(report["batching"]["mean_batch_size"], 4)
            self.assertEqual(report["phase"], "mixed_checkpoint_saved")
            self.assertEqual(report["final_upload"]["status"], "unknown")
            self.assertIn("retained teacher positions unknown/50000", status.format_summary(report))

    def test_latest_checkpoint_metrics_reports_saved_totals_without_inflight_samples(self):
        events = [{"event": "trained", "checkpoint": "/output/run/checkpoint-0001.pt"},
                  {"event": "checkpoint-saved", "checkpoint": "/output/run/checkpoint-0002.pt", "iteration": 2},
                  game("inflight", True, 500)]
        metrics = {"iteration": 2, "teacher_positions": 100, "teacher_games": 8, "mixed_games": 3,
                   "bootstrap_target_reached": False, "training": {"samples": 75}}
        payloads = {"run/events.jsonl": "\n".join(json.dumps(event) for event in events),
                    "run/metrics-0002.json": metrics}
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            report = status.collect("play-ghq", URI, folder, runner=mock_runner(payloads, calls))
            saved = Path(report["snapshot_directory"]) / "run/metrics-0002.json"
            self.assertEqual(json.loads(saved.read_text()), metrics)
        self.assertEqual(calls[-1][0][3], URI + "/run/metrics-0002.json")
        self.assertEqual(len(calls), len(status.OBJECTS) + 1)
        self.assertTrue(all(command[3].startswith(URI + "/") for command, _ in calls))
        self.assertEqual(report["reported_cumulative"]["teacher_positions"], 100)
        self.assertEqual(report["reported_cumulative"]["teacher_games"], 8)
        self.assertEqual(report["reported_cumulative"]["mixed_games"], 3)
        self.assertFalse(report["reported_cumulative"]["bootstrap_target_reached"])
        self.assertEqual(report["event_counts"]["teacher"]["sampled_positions"], 500)
        self.assertEqual(report["reported_cumulative_source"]["kind"], "latest_saved_checkpoint_metrics")
        self.assertEqual(report["reported_cumulative_source"]["iteration"], 2)
        self.assertIn("excludes later in-flight work", status.format_summary(report))

    def test_invalid_latest_checkpoint_event_does_not_trigger_metrics_read_or_fallback(self):
        invalid = [None, "gs://other-bucket/run/checkpoint-0002.pt", "/output/../run/checkpoint-0002.pt",
                   "/output/run/checkpoint-0002.pt/", "/output/run/checkpoint-*.pt", "checkpoint-2.pt",
                   "checkpoint--001.pt", "checkpoint-00002.pt", "checkpoint-0002.pt\n"]
        invalid_events = [{"event": "trained", "checkpoint": value} for value in invalid]
        invalid_events.extend([{"event": "trained", "checkpoint": "checkpoint-0002.pt", "iteration": value}
                               for value in (3, "2", True)])
        for event in invalid_events:
            with self.subTest(event=event), tempfile.TemporaryDirectory() as folder:
                calls = []
                events = [{"event": "trained", "checkpoint": "checkpoint-0001.pt"}, event]
                payloads = {"run/events.jsonl": "\n".join(json.dumps(item) for item in events)}
                report = status.collect("play-ghq", URI, folder, runner=mock_runner(payloads, calls))
                self.assertEqual(len(calls), len(status.OBJECTS))
                self.assertIsNone(report["reported_cumulative_source"])
                self.assertIsNone(report["reported_cumulative"]["teacher_positions"])

    def test_first_zero_based_curriculum_checkpoint_fetches_matching_metrics(self):
        # save_update emits training metrics directly in the event (no iteration
        # field), while cumulative counters live in metrics-0000.json.
        training = {"before": None, "after": None, "samples": 15867, "value_labels": 15036,
                    "validation_before": None,
                    "validation_after": {"policy": 3.1327786178363817, "value": .5955440222689535,
                                         "total": 3.7283226401053353},
                    "validation_samples": 2712, "interrupted": False, "steps": 124,
                    "seconds": 33.35432215800006}
        event = {"event": "trained", "checkpoint": "/output/run/checkpoint-0000.pt",
                 "phase": "teacher", **training}
        metrics = {"iteration": 0, "phase": "teacher", "training": training, "games": [],
                   "collection_incomplete": False, "teacher_positions": 15867, "teacher_games": 100,
                   "mixed_games": 0, "bootstrap_target_reached": False,
                   "collection_error": None, "promoted": False}
        payloads = {"run/events.jsonl": json.dumps(event) + "\n" + json.dumps(game("inflight", True, 51)),
                    "run/metrics-0000.json": metrics}
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            report = status.collect("play-ghq", URI, folder, runner=mock_runner(payloads, calls))
        self.assertEqual(calls[-1][0][3], URI + "/run/metrics-0000.json")
        self.assertEqual(report["reported_cumulative_source"]["iteration"], 0)
        self.assertEqual(report["reported_cumulative"]["teacher_positions"], 15867)
        self.assertEqual(report["reported_cumulative"]["mixed_games"], 0)
        self.assertFalse(report["reported_cumulative"]["bootstrap_target_reached"])
        self.assertEqual(report["event_counts"]["teacher"]["sampled_positions"], 51)
        self.assertIn("iteration 0", status.format_summary(report))

    def test_missing_failed_or_mismatched_metrics_never_supply_cumulative_totals(self):
        values = [None, subprocess.CompletedProcess([], 1, "", "403 permission denied"), "bad-json",
                  {"iteration": 1, "teacher_positions": 999}, {"iteration": "2", "teacher_positions": 999}]
        for value in values:
            with self.subTest(value=value), tempfile.TemporaryDirectory() as folder:
                payloads = {"run/events.jsonl": json.dumps({"event": "trained", "checkpoint": "checkpoint-0002.pt"}),
                            "run/metrics-0002.json": value}
                report = status.collect("play-ghq", URI, folder, runner=mock_runner(payloads, []))
                self.assertIsNone(report["reported_cumulative"]["teacher_positions"])
                self.assertIsNone(report["reported_cumulative_source"])
                self.assertEqual(report["fetches"]["run/metrics-0002.json"]["status"], "missing" if value is None else "error")

    def test_final_summary_precedes_saved_checkpoint_metrics_without_extra_read(self):
        payloads = {"run/events.jsonl": json.dumps({"event": "trained", "checkpoint": "checkpoint-0002.pt"}),
                    "run/metrics-0002.json": {"iteration": 2, "teacher_positions": 100},
                    "run/summary.json": {"status": "complete", "teacher_positions": 130,
                                         "bootstrap_target_reached": True}}
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            report = status.collect("play-ghq", URI, folder, runner=mock_runner(payloads, calls))
        self.assertEqual(len(calls), len(status.OBJECTS))
        self.assertEqual(report["reported_cumulative"]["teacher_positions"], 130)
        self.assertTrue(report["reported_cumulative"]["bootstrap_target_reached"])
        self.assertEqual(report["reported_cumulative_source"], {"kind": "pipeline_summary", "object": "run/summary.json"})
        self.assertIn("Cumulative source: final pipeline summary", status.format_summary(report))

    def test_gcloud_missing_object_wording_is_normal_and_cli_exits_successfully(self):
        payloads = {name: subprocess.CompletedProcess(
            [], 1, "", "ERROR: (gcloud.storage.cat) The following URLs matched no objects or files:\n" + URI + "/" + name)
            for name in status.OBJECTS}
        with tempfile.TemporaryDirectory() as folder:
            report = status.collect("play-ghq", URI, folder, runner=mock_runner(payloads, []))
            self.assertTrue(all(info["status"] == "missing" for info in report["fetches"].values()))
            self.assertIsNone(report["event_counts"])
            argv = ["status.py", "--project", "play-ghq", "--run-uri", URI, "--output", folder]
            with patch("sys.argv", argv), patch.object(status, "collect", return_value=report), patch("builtins.print"):
                self.assertEqual(status.main(), 0)

            payloads["hardware.json"] = subprocess.CompletedProcess(
                [], 1, "", "ERROR: (gcloud.storage.cat) Permission denied while listing objects or files")
            denied = status.collect("play-ghq", URI, folder, runner=mock_runner(payloads, []))
            self.assertEqual(denied["fetches"]["hardware.json"]["status"], "error")

    def test_missing_permission_failure_and_timeout_never_reuse_previous_counts(self):
        calls = []
        with tempfile.TemporaryDirectory() as folder:
            first = status.collect("play-ghq", URI, folder, runner=mock_runner(
                {"run/events.jsonl": json.dumps(game("one", True, 8))}, calls))
            payloads = {"run/events.jsonl": subprocess.CompletedProcess([], 1, "", "403 permission denied"),
                        "hardware.json": subprocess.TimeoutExpired("gcloud", 20)}
            second = status.collect("play-ghq", URI, folder, runner=mock_runner(payloads, calls))
            self.assertNotEqual(first["snapshot_directory"], second["snapshot_directory"])
            self.assertIsNone(second["event_counts"])
            self.assertIsNone(second["latest_checkpoint_reported"])
            self.assertEqual(second["fetches"]["run/events.jsonl"]["status"], "error")
            self.assertEqual(second["fetches"]["job-config.json"]["status"], "missing")
            self.assertEqual(second["final_upload"]["status"], "unknown")
            self.assertIn("Teacher: unknown", status.format_summary(second))

    def test_completion_confirms_upload_and_summary_keeps_cumulative_counts(self):
        payloads = {"completion.json": {"run_id": "ghq-train-example", "exit_code": 0,
                                       "final_sync_succeeded": True, "finished_at": 100},
                    "run/summary.json": {"status": "curriculum-deadline", "teacher_positions": 1234,
                                         "teacher_games": 10, "mixed_games": 5, "checkpoint": "/output/run/checkpoint-0003.pt"}}
        with tempfile.TemporaryDirectory() as folder:
            report = status.collect("play-ghq", URI, folder, runner=mock_runner(payloads, []))
        self.assertEqual(report["final_upload"]["status"], "confirmed")
        self.assertEqual(report["phase"], "worker_completed")
        self.assertEqual(report["reported_cumulative"]["teacher_positions"], 1234)
        self.assertIsNone(report["event_counts"])
        self.assertIn("remote snapshot age and VM liveness are unknown", report["freshness"])

    def test_timeout_diagnostics_remain_visible_after_successful_worker_shutdown(self):
        reason = {"kind": "global_deadline", "message": "Collection wall-time budget exceeded"}
        timeout = {"kind": "game_timeout", "game_id": "slow", "elapsed_s": 3601}
        events = [{"event": "collection-start", "teacher": False, "mixed_games": 27},
                  {"event": "game-timeout", **timeout},
                  {"event": "collection", "teacher": False, "completed_games": 31,
                   "timed_out_games": 1, "game_timeouts": [timeout],
                   "stop_reason": {"kind": "completed_with_game_timeouts"}}]
        payloads = {"run/events.jsonl": "\n".join(json.dumps(e) for e in events),
                    "run/summary.json": {"status": "curriculum-deadline", "stop_reason": reason},
                    "completion.json": {"run_id": "ghq-train-example", "exit_code": 0,
                                        "final_sync_succeeded": True}}
        with tempfile.TemporaryDirectory() as folder:
            report = status.collect("play-ghq", URI, folder, runner=mock_runner(payloads, []))
        self.assertEqual(report["phase"], "worker_completed")
        self.assertEqual(report["pipeline_status"], "curriculum-deadline")
        self.assertEqual(report["pipeline_stop_reason"], reason)
        self.assertEqual(report["last_collection_start"]["mixed_games"], 27)
        self.assertEqual(len(report["game_timeout_events"]), 1)
        self.assertEqual(report["batching"]["latest"]["game_timeouts"], [timeout])
        self.assertIn("global_deadline", status.format_summary(report))
        self.assertIn("individual game timeouts: 1", status.format_summary(report))
        # A watchdog event is neither a finished game nor a draw, and a start
        # event does not count as a saved checkpoint's cumulative progress.
        self.assertEqual(report["all_event_outcomes"]["finished_game_tasks"], 0)
        self.assertEqual(report["all_event_outcomes"]["draws"], 0)
        self.assertIsNone(report["reported_cumulative"]["mixed_games"])

    def test_partial_jsonl_and_duplicates_are_explicit_and_invalid_file_is_unknown(self):
        event = json.dumps(game("same", True, 9))
        raw = event + "\n" + event + '\n{"event":'
        with tempfile.TemporaryDirectory() as folder:
            report = status.collect("play-ghq", URI, folder, runner=mock_runner({"run/events.jsonl": raw}, []))
        self.assertEqual(report["event_counts"]["teacher"]["finished_game_tasks"], 1)
        self.assertEqual(report["event_quality"]["duplicate_game_events"], 1)
        self.assertTrue(report["event_quality"]["partial_tail"])
        with tempfile.TemporaryDirectory() as folder:
            bad = status.collect("play-ghq", URI, folder, runner=mock_runner({"run/events.jsonl": "not-json"}, []))
        self.assertIsNone(bad["event_counts"])
        self.assertEqual(bad["event_quality"]["malformed_lines"], [1])

    def test_location_validation_blocks_broad_or_injected_reads_before_any_call(self):
        for project, uri in (("--quiet", URI), ("play-ghq", "gs://bucket/runs/*"),
                             ("play-ghq", URI + "/checkpoint.pt"), ("play-ghq", "https://example.com"),
                             ("play-ghq", "gs://bucket/runs/../secret")):
            with self.subTest(project=project, uri=uri), tempfile.TemporaryDirectory() as folder:
                calls = []
                with self.assertRaises(ValueError):
                    status.collect(project, uri, folder, runner=mock_runner({}, calls))
                self.assertEqual(calls, [])
        self.assertEqual(status.validate_location("play-ghq", URI + "/"), URI)

    def test_upload_failure_and_wrong_run_completion_cannot_claim_success(self):
        payloads = {"completion.json": {"run_id": "other-run", "exit_code": 0, "final_sync_succeeded": True},
                    "upload-failed.json": {"run_id": "ghq-train-example", "exit_code": 1, "final_sync_succeeded": False}}
        with tempfile.TemporaryDirectory() as folder:
            report = status.collect("play-ghq", URI, folder, runner=mock_runner(payloads, []))
        self.assertEqual(report["final_upload"]["status"], "failed")
        self.assertEqual(report["identity_conflicts"], ["completion.json"])
        with tempfile.TemporaryDirectory() as folder:
            unknown = status.collect("play-ghq", URI, folder, runner=mock_runner({"upload-failed.json": {}}, []))
        self.assertEqual(unknown["final_upload"]["status"], "unknown")


if __name__ == "__main__":
    unittest.main()
