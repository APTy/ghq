"""Spawn CPU game workers and batch their independent requests in the parent.

Only the parent owns evaluators/models/CUDA. Each worker has at most one
outstanding request, so a batch never changes the sequence of simulations in
one game. Tasks and seeds are fixed before dispatch; results retain task order.
Call from an import-safe main module (the standard multiprocessing main guard).
"""
from collections import Counter, defaultdict, deque
import math
import multiprocessing as mp
from multiprocessing.connection import wait
import os
import pickle
import time
import traceback

from .rules import e

_WORKER_STARTUP_TIMEOUT = 60


class CollectionError(RuntimeError):
    def __init__(self, message, results=(), completed_indices=(), *, stop_reason="collection_failure", details=None):
        super().__init__(message)
        self.results = list(results)
        self.completed_indices = list(completed_indices)
        self.stop_reason = dict(stop_reason) if isinstance(stop_reason, dict) else {"kind": stop_reason}
        self.details = details or {}


class CollectionTimeout(TimeoutError):
    def __init__(self, message, results=(), completed_indices=(), *, stop_reason="global_deadline", details=None):
        super().__init__(message)
        self.results = list(results)
        self.completed_indices = list(completed_indices)
        self.stop_reason = dict(stop_reason) if isinstance(stop_reason, dict) else {"kind": stop_reason}
        self.details = details or {}


class _RemoteEvaluator:
    def __init__(self, connection, model_id, timeout, protocol="legacy", profile=False):
        self.connection, self.model_id, self.timeout = connection, model_id, timeout
        self.protocol, self.profile = protocol, profile
        self.sequence = 0
        self.metrics = {key: 0.0 for key in ("prepare_seconds", "send_seconds", "wait_seconds", "receive_seconds")}
        self.metrics["calls"] = 0

    def predict(self, board, legal_moves):
        started = time.perf_counter() if self.profile else None
        self.sequence += 1
        request = {"kind": "predict", "model_id": self.model_id, "sequence": self.sequence,
                   "protocol": self.protocol, "action_count": len(legal_moves)}
        if self.protocol == "encoded":
            from .encoding import encode_actions, encode_board
            request.update(board_input=encode_board(board), action_input=encode_actions(board, legal_moves))
        else:
            request.update(state=board.serialize(), actions=[move.uci() for move in legal_moves])
        if self.profile:
            request["worker_prepare_seconds"] = time.perf_counter() - started
            self.metrics["prepare_seconds"] += request["worker_prepare_seconds"]
            tick = time.perf_counter()
        self.connection.send(request)
        if self.profile:
            self.metrics["send_seconds"] += time.perf_counter() - tick
            tick = time.perf_counter()
        available = self.connection.poll(self.timeout)
        if self.profile:
            self.metrics["wait_seconds"] += time.perf_counter() - tick
            tick = time.perf_counter()
        if not available:
            raise TimeoutError(f"Evaluator {self.model_id!r} request exceeded {self.timeout}s")
        reply = self.connection.recv()
        if self.profile:
            self.metrics["receive_seconds"] += time.perf_counter() - tick
            self.metrics["calls"] += 1
        if (reply.get("kind"), reply.get("model_id"), reply.get("sequence")) != (
                "prediction", self.model_id, self.sequence):
            raise RuntimeError("Unexpected evaluator reply")
        return reply["result"]


def _worker(connection, request_timeout):
    # spawn starts a fresh interpreter, never inheriting initialized CUDA.
    # Environment limits are installed before spawn as well as here.
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "1"
    try:
        from .selfplay import play_game
        import torch
        torch.set_num_threads(1)
        connection.send({"kind": "ready"})
        while True:
            message = connection.recv()
            if message["kind"] == "stop":
                break
            index, task = message["index"], dict(message["task"])
            model_id = task.get("model_id", None if task.get("teacher") else "current")
            task["model_id"] = model_id
            protocols, profile = message.get("protocols", {}), message.get("profile", False)
            evaluator = None if task.get("teacher") else _RemoteEvaluator(
                connection, model_id, request_timeout, protocols.get(model_id, "legacy"), profile)
            other = None
            if task.get("opponent") in ("historical", "history"):
                other_id = task["opponent_model_id"]
                other = _RemoteEvaluator(
                    connection, other_id, request_timeout, protocols.get(other_id, "legacy"), profile)
                task["opponent_evaluator"] = other
            result = play_game(evaluator, **task)
            if profile:
                worker_profile = {}
                for item in (evaluator, other):
                    if item is not None:
                        aggregate = worker_profile.setdefault(item.model_id, {})
                        for key, value in item.metrics.items():
                            aggregate[key] = aggregate.get(key, 0) + value
                result[1]["inference_worker_profile"] = worker_profile
            connection.send({"kind": "result", "index": index, "result": result})
    except (EOFError, BrokenPipeError):
        pass
    except BaseException as error:
        try:
            connection.send({"kind": "error", "error": repr(error), "traceback": traceback.format_exc(),
                             "timeout": isinstance(error, TimeoutError)})
        except (EOFError, BrokenPipeError, OSError):
            pass
    finally:
        connection.close()


def collect_games(tasks, evaluators, *, workers=8, batch_size=32, batch_wait_ms=5,
                  request_timeout_s=120, game_timeout_s=900, collection_timeout_s=None,
                  metrics=None, on_result=None, on_event=None, inference_protocol="auto", profile=False):
    """Return ``[(Samples, report), ...]`` in task order, with optional streaming.

    ``tasks`` are serializable play_game keyword dictionaries; model_id and
    opponent_model_id route to immutable parent-side ``predict_batch`` objects
    in ``evaluators``. ``on_result(index, result)`` runs immediately on game
    completion. Opponent positions are omitted by play_game, not assigned
    fabricated targets. Pure CPU teacher tasks require no evaluators.

    A game's absolute watchdog retires only that game; a replacement worker
    handles remaining tasks. Its unfinished data is omitted, never labeled a
    draw. Successful returns can therefore contain fewer results than tasks,
    in input-index order; metrics.completed_indices maps those results.
    ``metrics`` records game_timeouts and a structured stop_reason.
    ``on_event(event)`` receives each game_timeout immediately, including its
    original task index and inference activity (not fabricated partial labels).
    Global, startup and inference-request timeouts raise CollectionTimeout carrying
    completed results/indices; evaluator/worker errors raise CollectionError.
    All worker processes are reaped on exit.
    A synchronous predict_batch already executing cannot be preempted; deadline
    detection may overshoot by one batch's latency, as with ordinary inference.
    Batch sizes are capped by workers and split by immutable model version.
    ``auto`` sends exact numeric inputs for evaluators with predict_encoded_batch;
    custom board evaluators retain the legacy protocol. ``profile`` enables
    phase timing (and explicit device synchronization in supporting evaluators).
    Profiled throughput must not be compared with unprofiled throughput.
    """
    if inference_protocol not in ("auto", "legacy", "encoded"):
        raise ValueError("inference_protocol must be auto, legacy, or encoded")
    protocols = {}
    for identifier, evaluator in evaluators.items():
        encoded = callable(getattr(evaluator, "predict_encoded_batch", None))
        if inference_protocol == "encoded" and not encoded:
            raise ValueError(f"Evaluator {identifier!r} does not support encoded inference")
        protocols[identifier] = "encoded" if encoded and inference_protocol != "legacy" else "legacy"
    if (not isinstance(workers, int) or not 1 <= workers <= 64 or
            not isinstance(batch_size, int) or not 1 <= batch_size <= 64):
        raise ValueError("workers and batch_size must be integers from 1 to 64")
    if not math.isfinite(batch_wait_ms) or batch_wait_ms < 0:
        raise ValueError("batch_wait_ms must be finite and nonnegative")
    for name, value in (("request_timeout_s", request_timeout_s), ("game_timeout_s", game_timeout_s),
                        ("collection_timeout_s", collection_timeout_s)):
        if value is not None and (not math.isfinite(value) or value <= 0):
            raise ValueError(f"{name} must be finite and positive")
    if request_timeout_s is None or game_timeout_s is None:
        raise ValueError("Request and game timeouts cannot be disabled")
    if batch_wait_ms / 1000 >= request_timeout_s:
        raise ValueError("Batch wait must be less than request timeout")
    tasks = [dict(task) for task in tasks]
    identifiers = set()
    for index, task in enumerate(tasks):
        if "start_board" in task or "evaluator" in task or "opponent_evaluator" in task:
            raise ValueError("Tasks must use start_state and model IDs, not boards/evaluators")
        task.setdefault("seed", index)
        task.setdefault("game_id", f"collection-{index:06d}")
        if not isinstance(task["seed"], int):
            raise ValueError("Task seed must be an integer")
        if task["game_id"] in identifiers:
            raise ValueError("Task game_id values must be unique")
        identifiers.add(task["game_id"])
        if not task.get("teacher"):
            ids = [task.setdefault("model_id", "current")]
            if task.get("opponent") in ("historical", "history"):
                if not task.get("opponent_model_id"):
                    raise ValueError("Historical opponent requires opponent_model_id")
                ids.append(task["opponent_model_id"])
            for identifier in ids:
                if (not isinstance(identifier, str) or not identifier or identifier not in evaluators or
                        not callable(getattr(evaluators[identifier], "predict_batch", None))):
                    raise ValueError(f"Missing predict_batch evaluator for {identifier!r}")
        pickle.dumps(task)

    stats = {"tasks": len(tasks), "workers": min(workers, len(tasks)), "requested_batch_size": batch_size,
             "batch_wait_ms": batch_wait_ms, "requests": 0, "batches": 0, "rows": 0,
             "evaluator_seconds": 0.0, "completed_games": 0, "completed_indices": [],
             "batch_size_histogram": {}, "models": {}, "cancelled_games": 0,
             "game_timeouts": [], "stop_reason": {"kind": "completed"}, "failure": None,
             "worker_starts": 0, "replacement_workers": 0,
             "discarded_queued_requests": 0, "discarded_predictions": 0}
    stats.update(inference_protocol=inference_protocol, model_protocols=protocols, profiling=bool(profile),
                 phase_seconds={key: 0.0 for key in ("worker_prepare", "request_receive", "request_decode",
                                                   "queue_wait", "reply_send", "collector_wait")},
                 inference_profile={}, worker_profile_completed_games={}, timing_notes=[
                     "Worker preparation and queue wait sum concurrent requests; they are not fractions of collection wall time.",
                     "Parent request_receive includes all pipe receives (ready, inference, result); decode/reply/wait are serial wall durations.",
                     "Worker send includes pickling and pipe writes; wait includes serving/queueing; receive includes unpickling. Only returned games contribute worker profiles.",
                     "Fine phase timings are populated only when profiling is enabled; zero otherwise means unmeasured.",
                     "Profiling can synchronize CUDA phases; use unprofiled cases for throughput comparisons."])
    started = time.perf_counter()
    first_dispatch = None
    results = {}
    processes, connections, states = [], [], {}
    pending = defaultdict(deque)
    sizes, latencies, request_latencies = Counter(), [], []
    model_stats = defaultdict(lambda: {"requests": 0, "batches": 0, "evaluator_seconds": 0.0})
    context = mp.get_context("spawn")
    next_task = 0
    initial_startup = True
    timed_out_indices = set()

    def completed():
        indices = sorted(results)
        return [results[i] for i in indices], indices

    def progress(connection, now=None):
        state = states[connection]
        now = time.perf_counter() if now is None else now
        index = state["index"]
        return {"index": index, "game_id": tasks[index]["game_id"] if index is not None else None,
                "elapsed_s": now - state["game_started"] if index is not None else None,
                "inference_requests": state["requests"], "inference_responses": state["responses"],
                "last_request_elapsed_s": (state["last_request"] - state["game_started"]
                                           if state["last_request"] is not None else None),
                "last_request_age_s": now - state["last_request"] if state["last_request"] is not None else None,
                "last_activity_age_s": now - state["last_activity"],
                "progress_scope": "inference activity, not completed game turns"}

    def fail(message, timeout=False, *, stop_reason="collection_failure", connection=None, **details):
        values, indices = completed()
        event = {"kind": stop_reason, "message": message, "collection_elapsed_s": time.perf_counter() - started}
        if connection is not None and connection in states:
            event.update(progress(connection))
        event.update(details)
        stats.update(stop_reason=event, failure=event)
        raise (CollectionTimeout if timeout else CollectionError)(
            message, values, indices, stop_reason=event, details=event)

    def dispatch(connection):
        nonlocal next_task, first_dispatch
        if next_task < len(tasks):
            index = next_task
            next_task += 1
            now = time.perf_counter()
            if first_dispatch is None:
                first_dispatch = now
            states[connection].update(index=index, game_started=now, pending=False,
                                      requests=0, responses=0, last_activity=now, last_request=None)
            connection.send({"kind": "game", "index": index, "task": tasks[index],
                             "protocols": protocols, "profile": bool(profile)})
        else:
            states[connection]["index"] = None

    def spawn_worker(replacement=False):
        parent, child = context.Pipe()
        process = context.Process(target=_worker, args=(child, request_timeout_s), daemon=True)
        # Install limits before spawn imports Torch, and restore the parent.
        keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
        previous = {name: os.environ.get(name) for name in keys}
        try:
            for name in keys:
                os.environ[name] = "1"
            process.start()
        except BaseException:
            parent.close()
            raise
        finally:
            for name, value in previous.items():
                if value is None:
                    os.environ.pop(name, None)
                else:
                    os.environ[name] = value
            child.close()
        processes.append(process)
        connections.append(parent)
        now = time.perf_counter()
        states[parent] = {"ready": False, "index": None, "spawned": now,
                          "game_started": 0, "pending": False, "process": process,
                          "requests": 0, "responses": 0, "last_activity": now, "last_request": None}
        stats["worker_starts"] += 1
        stats["replacement_workers"] += int(replacement)

    def retire_game(connection, now):
        event = {"event": "game_timeout", "kind": "game_timeout", "limit_s": game_timeout_s,
                 "collection_elapsed_s": now - started, **progress(connection, now)}
        state = states.pop(connection)
        index = state["index"]
        timed_out_indices.add(index)
        stats["game_timeouts"].append(event)
        connections.remove(connection)
        # Requests already in a synchronous GPU batch are discarded by the
        # response loop below. Remove queued requests before closing this pipe.
        for queue in pending.values():
            kept = [item for item in queue if item[0] is not connection]
            stats["discarded_queued_requests"] += len(queue) - len(kept)
            queue.clear()
            queue.extend(kept)
        process = state["process"]
        if process.is_alive():
            process.terminate()
        process.join(timeout=0.2)
        if process.is_alive():
            process.kill()
            process.join(timeout=1)
        connection.close()
        if process.is_alive():
            fail("Timed-out game worker could not be stopped", stop_reason="worker_failure")
        processes.remove(process)
        process.close()
        if on_event is not None:
            on_event(event)
        if next_task < len(tasks):
            spawn_worker(replacement=True)

    def check_deadlines(inflight=()):
        now = time.perf_counter()
        if collection_timeout_s is not None and now - started >= collection_timeout_s:
            fail("Collection wall-time budget exceeded", timeout=True, stop_reason="global_deadline")
        for connection, state in states.items():
            if not state["ready"] and now - state["spawned"] > _WORKER_STARTUP_TIMEOUT:
                fail(f"Worker startup exceeded {_WORKER_STARTUP_TIMEOUT} seconds", timeout=True,
                     stop_reason="worker_startup_timeout", connection=connection)
        for requests in pending.values():
            if requests and now - requests[0][2] >= request_timeout_s:
                fail("Evaluator request timeout", timeout=True, stop_reason="request_timeout",
                     connection=requests[0][0])
        # A request currently inside synchronous inference has already left
        # pending. Infrastructure timeouts take priority over game retirement.
        for connection, _, queued in inflight:
            if now - queued >= request_timeout_s:
                fail("Evaluator request timeout", timeout=True, stop_reason="request_timeout",
                     connection=connection)
        for connection, state in list(states.items()):
            if state["index"] is not None and now - state["game_started"] >= game_timeout_s:
                retire_game(connection, now)

    try:
        for _ in range(min(workers, len(tasks))):
            spawn_worker()

        while len(results) + len(timed_out_indices) < len(tasks):
            check_deadlines()
            if len(results) + len(timed_out_indices) == len(tasks):
                break
            # If every active game is waiting, no additional request can
            # arrive until we answer one. Do not impose the batching delay
            # on a single worker or the final underfilled batch.
            all_blocked = all(state["index"] is None or state["pending"] for state in states.values())
            for identifier, requests in list(pending.items()):
                if not requests or (len(requests) < batch_size and not all_blocked and
                                    time.perf_counter() - requests[0][2] < batch_wait_ms / 1000):
                    continue
                batch = [requests.popleft() for _ in range(min(batch_size, len(requests)))]
                decode_started = time.perf_counter()
                if profile:
                    stats["phase_seconds"]["queue_wait"] += sum(decode_started - queued for _, _, queued in batch)
                if protocols[identifier] == "encoded":
                    boards = [message["board_input"] for _, message, _ in batch]
                    actions = [message["action_input"] for _, message, _ in batch]
                    predict = evaluators[identifier].predict_encoded_batch
                else:
                    boards = [e.BaseBoard.deserialize(message["state"]) for _, message, _ in batch]
                    actions = [[e.Move.from_uci(action) for action in message["actions"]] for _, message, _ in batch]
                    predict = evaluators[identifier].predict_batch
                if profile:
                    stats["phase_seconds"]["request_decode"] += time.perf_counter() - decode_started
                tick = time.perf_counter()
                try:
                    if profile and getattr(evaluators[identifier], "supports_inference_profile", False):
                        predictions = predict(boards, actions, profile=stats["inference_profile"].setdefault(identifier, {}))
                    else:
                        predictions = predict(boards, actions)
                except Exception as error:
                    fail(f"Evaluator {identifier!r} failed: {error}", stop_reason="evaluator_failure",
                         model_id=identifier)
                elapsed = time.perf_counter() - tick
                if len(predictions) != len(batch):
                    fail(f"Evaluator {identifier!r} returned the wrong batch length", stop_reason="evaluator_failure",
                         model_id=identifier)
                stats["batches"] += 1
                stats["rows"] += len(batch)
                stats["evaluator_seconds"] += elapsed
                sizes[len(batch)] += 1
                model_stats[identifier]["batches"] += 1
                model_stats[identifier]["evaluator_seconds"] += elapsed
                latencies.append(elapsed * 1000)
                if len(latencies) > 4096:
                    del latencies[:2048]
                check_deadlines(batch)
                for (connection, message, queued), prediction in zip(batch, predictions):
                    if connection not in states:
                        stats["discarded_predictions"] += 1
                        continue
                    if time.perf_counter() - queued >= request_timeout_s:
                        fail("Evaluator request timeout", timeout=True, stop_reason="request_timeout",
                             connection=connection)
                    policy, value = prediction
                    policy, value = [float(p) for p in policy], float(value)
                    action_count = message.get("action_count", len(message.get("actions", ())))
                    if (len(policy) != action_count or not math.isfinite(value) or
                            any(not math.isfinite(p) or p < 0 for p in policy)):
                        fail(f"Evaluator {identifier!r} returned invalid probabilities/value",
                             stop_reason="evaluator_failure", model_id=identifier)
                    request_latencies.append((time.perf_counter() - queued) * 1000)
                    if len(request_latencies) > 4096:
                        del request_latencies[:2048]
                    reply_started = time.perf_counter() if profile else None
                    connection.send({"kind": "prediction", "model_id": identifier,
                                     "sequence": message["sequence"], "result": (policy, value)})
                    if profile:
                        stats["phase_seconds"]["reply_send"] += time.perf_counter() - reply_started
                    states[connection]["pending"] = False
                    states[connection]["responses"] += 1
                    states[connection]["last_activity"] = time.perf_counter()
            if len(results) + len(timed_out_indices) == len(tasks):
                break
            until_batch = [max(0, batch_wait_ms / 1000 - (time.perf_counter() - q[0][2]))
                           for q in pending.values() if q]
            timeout = min([0.05, *until_batch])
            wait_started = time.perf_counter() if profile else None
            ready = wait(connections, timeout=timeout)
            if profile:
                stats["phase_seconds"]["collector_wait"] += time.perf_counter() - wait_started
            for connection in ready:
                try:
                    receive_started = time.perf_counter() if profile else None
                    message = connection.recv()
                    if profile:
                        stats["phase_seconds"]["request_receive"] += time.perf_counter() - receive_started
                except (EOFError, OSError):
                    fail("Game worker exited unexpectedly", stop_reason="worker_failure", connection=connection)
                kind, state = message.get("kind"), states[connection]
                state["last_activity"] = time.perf_counter()
                if kind == "ready":
                    if state["ready"]:
                        fail("Worker repeated ready message", stop_reason="worker_failure", connection=connection)
                    state["ready"] = True
                    if initial_startup:
                        if all(item["ready"] for item in states.values()):
                            initial_startup = False
                            for worker_connection in connections:
                                dispatch(worker_connection)
                    else:
                        # A replacement must not redispatch any healthy game.
                        dispatch(connection)
                elif kind == "predict":
                    identifier = message["model_id"]
                    if state["index"] is None or state["pending"] or identifier not in evaluators:
                        fail("Invalid worker inference request", stop_reason="worker_failure", connection=connection)
                    if message.get("protocol", "legacy") != protocols[identifier]:
                        fail("Worker inference protocol mismatch", stop_reason="worker_failure", connection=connection)
                    if profile:
                        prepared = message.get("worker_prepare_seconds", 0.0)
                        if not isinstance(prepared, (int, float)) or not math.isfinite(prepared) or prepared < 0:
                            fail("Invalid worker preparation timing", stop_reason="worker_failure", connection=connection)
                        stats["phase_seconds"]["worker_prepare"] += prepared
                    state["pending"] = True
                    state["requests"] += 1
                    state["last_request"] = time.perf_counter()
                    pending[identifier].append((connection, message, time.perf_counter()))
                    stats["requests"] += 1
                    model_stats[identifier]["requests"] += 1
                elif kind == "result":
                    index = message["index"]
                    if index != state["index"] or state["pending"]:
                        fail("Unexpected worker result index", stop_reason="worker_failure", connection=connection)
                    results[index] = message["result"]
                    if profile:
                        for model_id, measurements in message["result"][1].get("inference_worker_profile", {}).items():
                            aggregate = stats["worker_profile_completed_games"].setdefault(model_id, {})
                            for key, value in measurements.items():
                                aggregate[key] = aggregate.get(key, 0) + value
                    if on_result is not None:
                        on_result(index, results[index])
                    dispatch(connection)
                elif kind == "error":
                    fail(f"Game {state['index']} failed: {message['error']}\n{message['traceback']}",
                         timeout=message.get("timeout", False),
                         stop_reason="request_timeout" if message.get("timeout", False) else "worker_failure",
                         connection=connection)
                else:
                    fail(f"Unknown worker message {kind!r}", stop_reason="worker_failure", connection=connection)
        stats["stop_reason"] = {"kind": "completed_with_game_timeouts" if timed_out_indices else "completed"}
    except (CollectionError, CollectionTimeout):
        raise
    except Exception as error:
        fail(f"Collection failed: {error}")
    finally:
        cleanup_started = time.perf_counter()
        for connection in connections:
            try:
                connection.send({"kind": "stop"})
            except (BrokenPipeError, EOFError, OSError):
                pass
        cleanup_deadline = time.perf_counter() + 0.5
        for process in processes:
            process.join(timeout=max(0, cleanup_deadline - time.perf_counter()))
        for process in processes:
            if process.is_alive():
                process.terminate()
        cleanup_deadline = time.perf_counter() + 2
        for process in processes:
            process.join(timeout=max(0, cleanup_deadline - time.perf_counter()))
            if process.is_alive():
                process.kill()
                process.join(timeout=1)
        for connection in connections:
            connection.close()
        for process in processes:
            if not process.is_alive():
                process.close()
        elapsed = time.perf_counter() - started
        stats.update(startup_seconds=(first_dispatch or cleanup_started) - started,
                     active_seconds=cleanup_started - first_dispatch if first_dispatch is not None else 0.0,
                     cleanup_seconds=time.perf_counter() - cleanup_started)
        stats.update(seconds=elapsed, completed_games=len(results), completed_indices=sorted(results),
                     cancelled_games=len(tasks) - len(results), timed_out_games=len(timed_out_indices),
                     timed_out_indices=sorted(timed_out_indices),
                     unstarted_indices=list(range(next_task, len(tasks))),
                     inflight_indices=sorted(state["index"] for state in states.values()
                                             if state["index"] is not None and state["index"] not in results
                                             and state["index"] not in timed_out_indices),
                     batch_size_histogram=dict(sorted(sizes.items())),
                     mean_batch_size=stats["rows"] / stats["batches"] if stats["batches"] else None,
                     evaluator_rows_per_second=stats["rows"] / stats["evaluator_seconds"] if stats["evaluator_seconds"] else None,
                     collection_games_per_second=len(results) / elapsed if elapsed else None,
                     models=dict(model_stats))
        for name, values in (("batch_latency_ms", latencies), ("request_latency_ms", request_latencies)):
            ordered = sorted(values)
            stats[name] = {"sample_count": len(ordered), "scope": "most recent up to 4096 observations",
                           "mean": sum(ordered) / len(ordered) if ordered else None,
                           "p95": ordered[min(len(ordered) - 1, int(.95 * len(ordered)))] if ordered else None,
                           "max": max(ordered) if ordered else None}
        if metrics is not None:
            metrics.clear()
            metrics.update(stats)
    return [results[index] for index in sorted(results)]
