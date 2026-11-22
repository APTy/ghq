from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Neural benchmark requires optional dependencies")

from ghqnet import benchmark_collector as bench
from ghqnet.benchmark_collector import summarize_reports
from ghqnet.collector import CollectionTimeout
from ghqnet.rules import e


class CollectorBenchmarkTests(unittest.TestCase):
    def test_cutoffs_not_draws_and_rates_use_concurrent_wall_time(self):
        reports = [dict(terminal=None, reason="max-turns", turns=8, actions=24,
                        seconds=10, moves=["skip"], root_legal_actions=100,
                        root_visited_actions=25, search_calls=4, root_policy_entropy_sum=8),
                   dict(terminal={"winner": None, "reason": "double-skip"}, reason="double-skip",
                        turns=2, actions=2, seconds=10, moves=["skip", "skip"],
                        root_legal_actions=50, root_visited_actions=5, search_calls=2,
                        root_policy_entropy_sum=1)]
        result = summarize_reports(reports, 12)
        self.assertEqual(result["outcomes"], {"unfinished": 1, "draw": 1})
        self.assertEqual(result["summed_game_seconds"], 20)
        self.assertEqual(result["actions_per_second"], 26 / 12)
        self.assertEqual(result["root_visited_fraction"], .2)
        self.assertEqual(result["mean_root_policy_entropy_nats"], 1.5)
        self.assertIsNone(result["opposite_player_evaluator_fraction"])
        self.assertEqual(result["terminal_games"], 1)
        self.assertEqual(result["unfinished_rollouts"], 1)
        self.assertEqual(result["returned_rollouts"], 2)
        self.assertEqual(result["terminal_games_per_second"], 1 / 12)
        self.assertIsNone(result["neural_rows"])

    def args(self, directory, *extra):
        checkpoint = Path(directory) / "checkpoint.pt"
        checkpoint.write_bytes(b"frozen benchmark model")
        parser = bench.build_parser()
        args = parser.parse_args([str(checkpoint), "--output", str(Path(directory) / "report.json"),
                                 "--device", "cpu", "--threads", "1", "--games", "2", "--max-turns", "1",
                                 "--simulations", "1", "--workers", "1", "--batch-size", "2",
                                 "--batch-wait-ms", "0", *extra])
        bench.validate_args(args, parser)
        return args

    @staticmethod
    def rollout(task):
        return dict(game_id=task["game_id"], seed=task["seed"], terminal=None, reason="max-turns",
                    turns=1, actions=3, seconds=5, moves=["skip"], search_simulations=8,
                    search_nodes=5, root_legal_actions=12, root_visited_actions=4)

    def run_mock(self, args, collect):
        with patch.object(bench, "load_checkpoint", return_value=(object(), {})), \
                patch.object(bench, "TorchEvaluator"), patch.object(bench, "collect_games", side_effect=collect), \
                redirect_stdout(io.StringIO()):
            return bench.run_benchmark(args)

    def test_single_flags_compatible_and_grid_counts_are_exact(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory, "--workers", "8", "16", "32", "--batch-size", "16", "32",
                             "--batch-wait-ms", "0", "1", "5", "--inference-protocol", "legacy", "encoded",
                             "--repeats", "2", "--profile-sample")
            grid = bench.case_grid(args)
            self.assertEqual(len(grid), 74)
            self.assertEqual([row[5] for row in grid[:2]], ["legacy", "encoded"])
            self.assertTrue(all(row[-2:] == (True, True) for row in grid[:2]))
            self.assertTrue(all(row[-2:] == (False, False) for row in grid[2:]))
            args = self.args(directory)
            self.assertEqual(args.batch_size, [2])
            self.assertEqual(args.batch_wait_ms, [0])

    def test_invalid_grids_and_mixed_opponents_rejected(self):
        choices = [("--batch-size", "0"), ("--workers", "65"), ("--repeats", "0"),
                   ("--batch-wait-ms", "nan"), ("--case-max-seconds", "inf"),
                   ("--batch-size", "4", "4"), ("--batch-wait-ms", "120000"),
                   ("--cpu-fraction", ".8", "--history-fraction", ".3"),
                   ("--history-fraction", ".2"), ("--profile-sample", "--profile-inference")]
        with tempfile.TemporaryDirectory() as directory, redirect_stderr(io.StringIO()):
            for choice in choices:
                with self.subTest(choice=choice), self.assertRaises(SystemExit):
                    self.args(directory, *choice)

    def test_grid_preserves_workload_profiles_and_reports_actual_workers(self):
        seen = []
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory, "--workers", "1", "4", "--batch-size", "1", "2",
                             "--inference-protocol", "legacy", "encoded", "--profile-sample")

            def collect(tasks, evaluators, **kwargs):
                seen.append((tasks, kwargs["inference_protocol"], kwargs["profile"]))
                kwargs["metrics"].update(workers=min(kwargs["workers"], len(tasks)), rows=10,
                    seconds=10, startup_seconds=1, active_seconds=8, cleanup_seconds=1,
                    phase_seconds={"request_decode": .2}, inference_profile={"model": {"forward_seconds": .3}})
                for index, task in enumerate(tasks):
                    kwargs["on_result"](index, ([], self.rollout(task)))

            result = self.run_mock(args, collect)
            self.assertTrue(result["complete"])
            self.assertEqual(len(result["cases"]), 10)
            self.assertEqual(seen[0][0], seen[-1][0])
            self.assertEqual([item[2] for item in seen[:2]], [True, True])
            self.assertTrue(all(not item[2] for item in seen[2:]))
            self.assertEqual({case["task_manifest_sha256"] for case in result["cases"]},
                             {result["cases"][0]["task_manifest_sha256"]})
            self.assertTrue(all(case["moves_match_same_simulation_reference"] for case in result["cases"]))
            last = result["cases"][-1]
            self.assertEqual(last["actual_workers"], 2)
            self.assertEqual(last["requested_workers"], 4)
            self.assertEqual(last["summary"]["terminal_games"], 0)
            self.assertEqual(last["summary"]["new_search_nodes"], 10)
            self.assertEqual(last["summary"]["neural_rows"], 10)
            self.assertEqual(last["timing"]["startup_seconds"], 1)
            self.assertEqual(last["timing"]["phase_seconds"], {"request_decode": .2})
            self.assertEqual(json.loads(args.output.read_text()), result)

    def test_partial_case_deadline_retains_completed_only_then_continues(self):
        calls = []
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory, "--workers", "1", "2", "--case-max-seconds", "2")

            def collect(tasks, evaluators, **kwargs):
                calls.append(kwargs["collection_timeout_s"])
                kwargs["metrics"].update(workers=1, rows=17)
                report = self.rollout(tasks[0])
                # Carry a completed result on the exception even if no callback
                # was observed, as older collector callers may do.
                raise CollectionTimeout("case deadline", [([], report)], [0], stop_reason="global_deadline")

            result = self.run_mock(args, collect)
            self.assertFalse(result["complete"])
            self.assertEqual(calls, [2, 2])
            self.assertEqual(result["cases_attempted"], 2)
            first = result["cases"][0]
            self.assertEqual(first["summary"]["returned_rollouts"], 1)
            self.assertEqual(first["summary"]["search_simulations"], 8)
            self.assertEqual(first["summary"]["neural_rows"], 17)
            self.assertEqual(first["summary"]["terminal_games"], 0)

    def test_overall_timeout_stops_grid_and_persists_empty_parity_as_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory, "--workers", "1", "2")

            def collect(tasks, evaluators, **kwargs):
                kwargs["metrics"].update(workers=1, rows=3)
                raise CollectionTimeout("overall deadline", stop_reason="global_deadline")

            result = self.run_mock(args, collect)
            self.assertEqual(result["cases_attempted"], 1)
            self.assertEqual(result["cases_not_started"], 1)
            self.assertEqual(result["stop_reason"]["kind"], "overall_deadline")
            self.assertIsNone(result["cases"][0]["moves_match_same_simulation_reference"])
            self.assertEqual(result["cases"][0]["summary"]["returned_rollouts"], 0)
            self.assertFalse(json.loads(args.output.read_text())["complete"])

    def test_infrastructure_failure_stops_and_is_saved(self):
        with tempfile.TemporaryDirectory() as directory:
            args = self.args(directory, "--workers", "1", "2", "--case-max-seconds", "2")

            def collect(tasks, evaluators, **kwargs):
                raise CollectionTimeout("request expired", stop_reason="request_timeout")

            with self.assertRaisesRegex(RuntimeError, "request expired"):
                self.run_mock(args, collect)
            report = json.loads(args.output.read_text())
            self.assertEqual(report["stop_reason"]["kind"], "collection_failure")
            self.assertEqual(len(report["cases"]), 1)

    def test_positions_are_seeded_distinct_and_mixed_tasks_keep_model_identity(self):
        board = e.BaseBoard()
        states = [board.serialize()]
        for _ in range(3):
            move = next(move for move in board.generate_legal_moves() if move.uci() != "skip")
            board.push(move)
            states.append(board.serialize())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "positions.jsonl"
            header = {"version": 2, "rules_sha256": bench.rules_source_hash(), "metadata": {"kind": "training"}}
            path.write_text("\n".join(json.dumps(row) for row in
                [header, *[{"state": state} for state in states + states]]) + "\n")
            positions, metadata = bench.load_positions(path, 2, 123)
            self.assertEqual((positions, metadata), bench.load_positions(path, 2, 123))
            self.assertEqual(metadata["source_rows"], 8)
            self.assertEqual(metadata["unique_exact_states"], 4)
            self.assertEqual(len(positions), 2)
            args = self.args(directory, "--games", "8", "--history-fraction", ".5",
                             "--historical-checkpoint", str(Path(directory) / "history.pt"))
            tasks = bench.build_tasks(args, 7, 0, "current", "history", positions)
            self.assertEqual(sum(task["opponent"] == "historical" for task in tasks), 4)
            self.assertEqual({task["model_id"] for task in tasks}, {"current"})
            self.assertEqual({task["opponent_model_id"] for task in tasks if task["opponent"] == "historical"}, {"history"})
            self.assertTrue(all(task["opening_turns"] == 0 for task in tasks))
            self.assertEqual(tasks[0]["start_state"], positions[0])
            again = bench.build_tasks(args, 7, 1, "current", "history", positions)
            self.assertNotEqual(tasks[0]["seed"], again[0]["seed"])
            self.assertEqual(tasks[0]["start_state"], again[0]["start_state"])

    def test_position_rules_mismatch_and_missing_states_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "positions.jsonl"
            path.write_text(json.dumps({"metadata": {}, "rules_sha256": "wrong"}) + "\n")
            with self.assertRaisesRegex(ValueError, "rules hash"):
                bench.load_positions(path, 2, 1)
            path.write_text('{"other": 1}\n')
            with self.assertRaisesRegex(ValueError, "serialized state"):
                bench.load_positions(path, 2, 1)


if __name__ == "__main__":
    unittest.main()
