"""Actual spawned-worker parity, batching, model isolation, and cleanup."""
import multiprocessing as mp
import time
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Collector tests require optional neural dependencies")

from ghqnet.collector import collect_games, CollectionError, CollectionTimeout
from ghqnet.rules import e
from ghqnet.selfplay import play_game
from ghqnet.model import NetworkConfig, PolicyValueNet, TorchEvaluator
from ghqnet.encoding import ACTION_KINDS


class EncodedSkipEvaluator:
    """Board and numeric paths choose the same exact legal Skip action."""
    def predict_batch(self, boards, moves):
        return [([float(move.name == "Skip") for move in actions], 0.0) for actions in moves]

    def predict_encoded_batch(self, boards, actions):
        return [((row[:, 4] == ACTION_KINDS["Skip"]).astype(float).tolist(), 0.0) for row in actions]


class SkipEvaluator:
    def __init__(self, fail=False):
        self.batch_sizes, self.fail = [], fail

    def predict(self, board, moves):
        return [float(move.name == "Skip") for move in moves], 0.0

    def predict_batch(self, boards, moves):
        self.batch_sizes.append(len(boards))
        if self.fail:
            raise RuntimeError("deliberate evaluator failure")
        for board, actions in zip(boards, moves):
            self.assert_exact_actions(board, actions)
        return [self.predict(board, actions) for board, actions in zip(boards, moves)]

    @staticmethod
    def assert_exact_actions(board, moves):
        assert [m.uci() for m in moves] == sorted(m.uci() for m in board.generate_legal_moves())


def position():
    board = e.BaseBoard()
    board.clear_board()
    for square, symbol in {"a1": "Q", "h8": "q"}.items():
        board.set_piece_at(e.parse_square(square), e.Piece.from_symbol(symbol))
    return board.serialize()


def scripted_worker(connection, request_timeout):
    """Small pipe peer for deterministic watchdog/IPC tests, without game search."""
    try:
        connection.send({"kind": "ready"})
        while True:
            message = connection.recv()
            if message["kind"] == "stop":
                return
            task, index = message["task"], message["index"]
            if task.get("script") == "predict":
                board = e.BaseBoard.deserialize(task["start_state"])
                connection.send({"kind": "predict", "model_id": "current", "sequence": index + 1,
                                 "state": task["start_state"],
                                 "actions": sorted(m.uci() for m in board.generate_legal_moves())})
                reply = connection.recv()
                if reply["kind"] == "stop":
                    return
                assert (reply["kind"], reply["model_id"], reply["sequence"]) == (
                    "prediction", "current", index + 1)
            time.sleep(task.get("delay", 0))
            connection.send({"kind": "result", "index": index,
                             "result": ([], {"game_id": task["game_id"]})})
    except (EOFError, BrokenPipeError, OSError):
        pass
    finally:
        connection.close()


def unready_worker(connection, request_timeout):
    time.sleep(10)


class SlowFirstEvaluator(SkipEvaluator):
    def __init__(self, delay):
        super().__init__()
        self.delay = delay

    def predict_batch(self, boards, moves):
        if not self.batch_sizes:
            time.sleep(self.delay)
        return super().predict_batch(boards, moves)


class NeuralCollectorTests(unittest.TestCase):
    def jobs(self, count=2, **extra):
        return [dict(seed=i + 17, game_id=f"test-{i}", simulations=3, max_turns=2,
                     start_state=position(), **extra) for i in range(count)]

    def test_batched_spawn_matches_serial_samples_and_seeded_order(self):
        evaluator, metrics, arrived = SkipEvaluator(), {}, []
        tasks = self.jobs(4)
        results = collect_games(tasks, {"current": evaluator}, workers=2, batch_size=2,
                                batch_wait_ms=100, metrics=metrics,
                                on_result=lambda index, result: arrived.append(index))
        self.assertEqual(sorted(arrived), [0, 1, 2, 3])
        self.assertIn(2, evaluator.batch_sizes)
        self.assertEqual(metrics["completed_games"], 4)
        self.assertEqual(metrics["cancelled_games"], 0)
        self.assertEqual(metrics["rows"], metrics["requests"])
        for task, (samples, report) in zip(tasks, results):
            serial_samples, serial_report = play_game(SkipEvaluator(), model_id="current", **task)
            self.assertEqual(samples, serial_samples)
            for key in ("game_id", "seed", "moves", "final_state", "terminal", "search_simulations"):
                self.assertEqual(report[key], serial_report[key])

    def test_model_version_routing_and_mixed_learner_only_results(self):
        current, old = SkipEvaluator(), SkipEvaluator()
        tasks = self.jobs(opponent="historical", learner_color="RED", opponent_model_id="old")
        metrics = {}
        results = collect_games(tasks, {"current": current, "old": old}, workers=2,
                                batch_size=2, batch_wait_ms=100, metrics=metrics)
        self.assertEqual(set(metrics["models"]), {"current", "old"})
        self.assertTrue(current.batch_sizes)
        self.assertTrue(old.batch_sizes)
        for samples, report in results:
            self.assertEqual(len(samples), 1)
            self.assertEqual(report["sample_action_indices"], [0])
            self.assertEqual(report["action_sources"], ["mcts:current", "mcts:old"])
            self.assertEqual(samples[0].outcome, 1)

    def test_cpu_teacher_runs_in_same_pool_without_neural_evaluator(self):
        metrics = {}
        tasks = self.jobs(1, teacher=True, teacher_ms=2, teacher_depth=3, teacher_beam_width=8)
        samples, report = collect_games(tasks, {}, workers=1, metrics=metrics)[0]
        self.assertGreater(len(samples), 0)
        self.assertGreater(report["teacher_search_calls"], 0)
        self.assertEqual(report["evaluator_calls"], 0)
        self.assertEqual(metrics["requests"], 0)
        self.assertEqual(report["teacher_depth"], 3)
        self.assertEqual(report["teacher_beam_width"], 8)

    def test_evaluator_error_propagates_and_workers_are_reaped(self):
        before = {p.pid for p in mp.active_children()}
        metrics = {}
        with self.assertRaisesRegex(CollectionError, "deliberate evaluator failure"):
            collect_games(self.jobs(), {"current": SkipEvaluator(fail=True)}, workers=2, metrics=metrics)
        self.assertEqual({p.pid for p in mp.active_children()}, before)
        self.assertEqual(metrics["completed_games"], 0)
        self.assertEqual(metrics["cancelled_games"], 2)
        self.assertEqual(metrics["stop_reason"]["kind"], "evaluator_failure")

    def test_worker_error_and_collection_deadline_are_explicit(self):
        invalid = self.jobs(1)
        invalid[0]["start_state"] = "invalid"
        with self.assertRaisesRegex(CollectionError, "Game 0 failed"):
            collect_games(invalid, {"current": SkipEvaluator()}, workers=1)
        metrics = {}
        with self.assertRaises(CollectionTimeout) as caught:
            collect_games(self.jobs(), {"current": SkipEvaluator()}, workers=2,
                          collection_timeout_s=.001, metrics=metrics)
        self.assertEqual(caught.exception.results, [])
        self.assertEqual(caught.exception.completed_indices, [])
        self.assertEqual(metrics["cancelled_games"], 2)
        self.assertEqual(metrics["stop_reason"]["kind"], "global_deadline")
        self.assertEqual(caught.exception.stop_reason["kind"], "global_deadline")
        self.assertEqual(metrics["unstarted_indices"], [0, 1])
        self.assertEqual(metrics["inflight_indices"], [])

    def test_completed_results_survive_later_worker_failure(self):
        tasks = self.jobs(2)
        tasks[1]["start_state"] = "invalid"
        metrics, received = {}, []
        with self.assertRaises(CollectionError) as caught:
            collect_games(tasks, {"current": SkipEvaluator()}, workers=1, metrics=metrics,
                          on_result=lambda index, result: received.append(index))
        self.assertEqual(received, [0])
        self.assertEqual(caught.exception.completed_indices, [0])
        self.assertEqual(caught.exception.results[0][1]["game_id"], "test-0")
        self.assertEqual(metrics["completed_games"], 1)

    def test_real_torch_parent_evaluator_produces_legal_complete_turns(self):
        previous = torch.get_num_threads()
        try:
            torch.set_num_threads(1)
            torch.manual_seed(123)
            evaluator = TorchEvaluator(PolicyValueNet(NetworkConfig(blocks=1, channels=8)), "cpu")
            tasks = self.jobs(2)
            for task in tasks:
                task.update(simulations=2, max_turns=1)
            results = collect_games(tasks, {"current": evaluator}, workers=2,
                                    batch_size=2, batch_wait_ms=20)
        finally:
            torch.set_num_threads(previous)
        for samples, report in results:
            board = e.BaseBoard.deserialize(position())
            for action in report["moves"]:
                legal = {move.uci(): move for move in board.generate_legal_moves()}
                self.assertIn(action, legal)
                board.push(legal[action])
            self.assertEqual(board.turn, e.BLUE)
            self.assertEqual(board.serialize(), report["final_state"])
            self.assertTrue(report["truncated"])
            for sample in samples:
                sample.validate()

    def test_missing_model_and_duplicate_game_ids_fail_before_spawn(self):
        with self.assertRaisesRegex(ValueError, "Missing predict_batch"):
            collect_games(self.jobs(), {})
        tasks = self.jobs()
        tasks[1]["game_id"] = tasks[0]["game_id"]
        with self.assertRaisesRegex(ValueError, "unique"):
            collect_games(tasks, {"current": SkipEvaluator()})

    def test_encoded_and_legacy_transports_preserve_game_targets(self):
        tasks = self.jobs(4)
        runs = {}
        for protocol in ("legacy", "encoded"):
            metrics = {}
            runs[protocol] = collect_games(tasks, {"current": EncodedSkipEvaluator()}, workers=2,
                                           batch_size=2, batch_wait_ms=10,
                                           inference_protocol=protocol, profile=True, metrics=metrics)
            self.assertEqual(metrics["model_protocols"], {"current": protocol})
            self.assertGreater(metrics["phase_seconds"]["worker_prepare"], 0)
            self.assertGreater(metrics["phase_seconds"]["reply_send"], 0)
            self.assertGreater(metrics["startup_seconds"], 0)
            self.assertGreater(metrics["active_seconds"], 0)
        for (left, left_report), (right, right_report) in zip(runs["legacy"], runs["encoded"]):
            self.assertEqual(left, right)
            for key in ("moves", "final_state", "terminal", "search_simulations"):
                self.assertEqual(left_report[key], right_report[key])

    def test_encoded_transport_never_parses_board_or_move_in_parent(self):
        tasks = self.jobs(1)
        with patch.object(e.BaseBoard, "deserialize", side_effect=AssertionError("parent board parse")), \
                patch.object(e.Move, "from_uci", side_effect=AssertionError("parent action parse")):
            result = collect_games(tasks, {"current": EncodedSkipEvaluator()}, workers=1,
                                   inference_protocol="encoded")
        self.assertEqual(len(result), 1)

    def test_protocol_selection_is_per_model_and_explicit_encoded_requires_support(self):
        with self.assertRaisesRegex(ValueError, "does not support encoded"):
            collect_games(self.jobs(), {"current": SkipEvaluator()}, inference_protocol="encoded")
        metrics = {}
        collect_games(self.jobs(opponent="historical", learner_color="RED", opponent_model_id="old"),
                      {"current": EncodedSkipEvaluator(), "old": SkipEvaluator()}, workers=2,
                      metrics=metrics)
        self.assertEqual(metrics["model_protocols"], {"current": "encoded", "old": "legacy"})
        self.assertFalse(metrics["profiling"])
        self.assertEqual(set(metrics["phase_seconds"].values()), {0.0})

    def test_single_waiting_worker_does_not_sleep_for_batch_timeout(self):
        metrics = {}
        collect_games(self.jobs(1), {"current": SkipEvaluator()}, workers=1,
                      batch_size=32, batch_wait_ms=1000, metrics=metrics)
        # Queue latency excludes process startup. There cannot be a second
        # request while the sole worker is blocked, so waiting 1s is wasteful.
        self.assertLess(metrics["request_latency_ms"]["max"], 500)

    def test_game_timeout_isolated_with_healthy_results_and_queued_request_cleanup(self):
        tasks = self.jobs(16)
        tasks[0]["script"] = "predict"
        for task in tasks[1:]:
            task.update(teacher=True, delay=.16)
        evaluator, metrics, arrived, events = SkipEvaluator(), {}, [], []
        before = {p.pid for p in mp.active_children()}
        with patch("ghqnet.collector._worker", scripted_worker):
            results = collect_games(tasks, {"current": evaluator}, workers=2,
                                    game_timeout_s=.3, request_timeout_s=5, batch_wait_ms=1000,
                                    metrics=metrics, on_result=lambda i, r: arrived.append(i),
                                    on_event=events.append)
        self.assertEqual({p.pid for p in mp.active_children()}, before)
        self.assertEqual([report["game_id"] for _, report in results],
                         [task["game_id"] for task in tasks[1:]])
        self.assertEqual(sorted(arrived), list(range(1, 16)))
        self.assertEqual(metrics["completed_indices"], list(range(1, 16)))
        self.assertEqual(metrics["timed_out_indices"], [0])
        self.assertEqual(metrics["cancelled_games"], 1)
        self.assertEqual(metrics["timed_out_games"], 1)
        self.assertEqual(metrics["stop_reason"]["kind"], "completed_with_game_timeouts")
        self.assertEqual(metrics["unstarted_indices"], [])
        self.assertEqual(metrics["inflight_indices"], [])
        self.assertGreaterEqual(metrics["replacement_workers"], 1)
        self.assertEqual(metrics["discarded_queued_requests"], 1)
        self.assertEqual(evaluator.batch_sizes, [])
        self.assertEqual(events, metrics["game_timeouts"])
        self.assertEqual(events[0]["event"], "game_timeout")
        self.assertEqual(events[0]["game_id"], tasks[0]["game_id"])
        self.assertEqual(events[0]["inference_requests"], 1)
        self.assertEqual(events[0]["inference_responses"], 0)
        self.assertIsNotNone(events[0]["last_request_age_s"])

    def test_expired_inflight_prediction_cannot_reach_replacement(self):
        tasks = self.jobs(2)
        for task in tasks:
            task["script"] = "predict"
        metrics, arrived = {}, []
        before = {p.pid for p in mp.active_children()}
        with patch("ghqnet.collector._worker", scripted_worker):
            results = collect_games(tasks, {"current": SlowFirstEvaluator(.35)}, workers=1,
                                    game_timeout_s=.2, request_timeout_s=5, metrics=metrics,
                                    on_result=lambda i, r: arrived.append(i))
        self.assertEqual({p.pid for p in mp.active_children()}, before)
        self.assertEqual(arrived, [1])
        self.assertEqual(results[0][1]["game_id"], tasks[1]["game_id"])
        self.assertEqual(metrics["timed_out_indices"], [0])
        self.assertEqual(metrics["discarded_predictions"], 1)
        self.assertEqual(metrics["worker_starts"], 2)
        self.assertEqual(metrics["completed_indices"], [1])

    def test_all_games_timing_out_returns_no_fabricated_results(self):
        tasks = self.jobs(2)
        for task in tasks:
            task["delay"] = 10
        metrics, arrived = {}, []
        with patch("ghqnet.collector._worker", scripted_worker):
            results = collect_games(tasks, {"current": SkipEvaluator()}, workers=2,
                                    game_timeout_s=.1, metrics=metrics,
                                    on_result=lambda i, r: arrived.append(i))
        self.assertEqual(results, [])
        self.assertEqual(arrived, [])
        self.assertEqual(metrics["completed_indices"], [])
        self.assertEqual(metrics["timed_out_indices"], [0, 1])
        self.assertEqual(metrics["inflight_indices"], [])
        self.assertEqual(metrics["unstarted_indices"], [])
        self.assertEqual(metrics["stop_reason"]["kind"], "completed_with_game_timeouts")

    def test_inflight_request_timeout_is_fatal_even_when_game_also_expires(self):
        tasks = self.jobs(1)
        tasks[0]["script"] = "predict"
        metrics = {}
        with patch("ghqnet.collector._worker", scripted_worker):
            with self.assertRaises(CollectionTimeout) as caught:
                collect_games(tasks, {"current": SlowFirstEvaluator(.2)}, workers=1,
                              game_timeout_s=.1, request_timeout_s=.08, metrics=metrics)
        self.assertEqual(caught.exception.stop_reason["kind"], "request_timeout")
        self.assertEqual(metrics["stop_reason"]["kind"], "request_timeout")
        self.assertEqual(metrics["timed_out_indices"], [])
        self.assertEqual(metrics["inflight_indices"], [0])

    def test_startup_timeout_has_distinct_fatal_reason(self):
        metrics = {}
        with patch("ghqnet.collector._worker", unready_worker), \
                patch("ghqnet.collector._WORKER_STARTUP_TIMEOUT", .05):
            with self.assertRaises(CollectionTimeout) as caught:
                collect_games(self.jobs(1), {"current": SkipEvaluator()}, metrics=metrics)
        self.assertEqual(caught.exception.stop_reason["kind"], "worker_startup_timeout")
        self.assertEqual(metrics["completed_games"], 0)
        self.assertEqual(metrics["unstarted_indices"], [0])

    def test_global_deadline_preserves_completed_callbacks_and_separates_pending_tasks(self):
        tasks = self.jobs(4)
        tasks[1]["delay"] = 10
        metrics, arrived = {}, []

        # A patched monotonic clock advances only in the callback. Child game
        # delays still use real time, while the parent's global budget is exact.
        original_clock = time.perf_counter
        offset = [0]

        def completed_callback(index, result):
            arrived.append(index)
            offset[0] = 100

        with patch("ghqnet.collector._worker", scripted_worker), \
                patch("ghqnet.collector.time.perf_counter", lambda: original_clock() + offset[0]):
            with self.assertRaises(CollectionTimeout) as caught:
                collect_games(tasks, {"current": SkipEvaluator()}, workers=1,
                              collection_timeout_s=50, metrics=metrics, on_result=completed_callback)
        self.assertEqual(arrived, [0])
        self.assertEqual(caught.exception.completed_indices, [0])
        self.assertEqual(metrics["completed_indices"], [0])
        self.assertEqual(metrics["inflight_indices"], [1])
        self.assertEqual(metrics["unstarted_indices"], [2, 3])
        self.assertEqual(metrics["stop_reason"]["kind"], "global_deadline")
        self.assertEqual(metrics["timed_out_games"], 0)


if __name__ == "__main__":
    unittest.main()
