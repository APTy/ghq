"""Curriculum coverage, independent holdouts, and recoverable partial runs."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
import hashlib
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Neural curriculum tests require optional Torch dependencies")

from ghqnet import pipeline
from ghqnet.collector import CollectionError, CollectionTimeout
from ghqnet.curriculum import ReplayPartition, fingerprint, opponent_schedule, retain_replay
from ghqnet.data import Sample, read_replay, write_replay
from ghqnet.model import NetworkConfig, PolicyValueNet, load_checkpoint, save_checkpoint
from ghqnet.rules import e
from ghqnet.train import fit


def sample(board, game_id, source="mcts", outcome=0):
    actions = sorted(move.uci() for move in board.generate_legal_moves())
    return Sample(board.serialize(), actions, [float(action == "skip") for action in actions],
                  outcome=outcome, game_id=game_id, source=source)


def fixture_collect(tasks, evaluators, *, on_result, metrics, **options):
    results = []
    for index, task in enumerate(tasks):
        board = e.BaseBoard()
        rng = random.Random(task["seed"])
        for _ in range(3):
            moves = [move for move in board.generate_legal_moves() if move.name != "Skip"]
            board.push(rng.choice(moves))
        rows = [sample(board, task["game_id"], "ghq-cpu-4" if task["teacher"] else "mcts")]
        board.push(next(move for move in board.generate_legal_moves() if move.name != "Skip"))
        rows.append(sample(board, task["game_id"], rows[0].source))
        report = {"game_id": task["game_id"], "teacher": task["teacher"], "truncated": False,
                  "terminal": {"winner": "RED", "reason": "fixture"}, "opponent": task["opponent"]}
        result = (rows, report)
        on_result(index, result)
        results.append(result)
    metrics.update(completed_games=len(results), requested_games=len(tasks))
    return results


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def args(self, output, *extra):
        return pipeline.parser().parse_args([
            "--output", str(output), "--curriculum", "--device", "cpu", "--threads", "1",
            "--iterations", "1", "--games", "4", "--teacher-positions", "4", "--teacher-max-games", "4",
            "--teacher-update-games", "2", "--workers", "2", "--inference-batch-size", "2",
            "--simulations", "1", "--max-turns", "2", "--epochs", "1", "--batch-size", "2",
            "--replay-size", "20", "--blocks", "0", "--channels", "4", *extra])

    def run_fixture(self, args, collector=fixture_collect):
        with patch("ghqnet.collector.collect_games", side_effect=collector), redirect_stdout(io.StringIO()):
            return pipeline.run(args)

    def test_whole_family_split_and_mirrored_state_exclusion_survive_eviction(self):
        partition = ReplayPartition(19, 0.5)
        train_family = next(f"family-{i}" for i in range(100) if partition.split_for(f"family-{i}") == "train")
        validation_family = next(f"family-{i}" for i in range(100) if partition.split_for(f"family-{i}") == "validation")
        board = e.BaseBoard()
        row = sample(board, "first-game")
        training, validation, excluded = partition.add_game([row], family_id=train_family)
        self.assertEqual((len(training), len(validation), excluded), (1, 0, 0))
        mirror = sample(board.mirror(), "different-game")
        self.assertEqual(fingerprint(row), fingerprint(mirror))
        restored = ReplayPartition.from_dict(partition.to_dict())
        training, validation, excluded = restored.add_game([mirror], family_id=validation_family)
        self.assertEqual((len(training), len(validation), excluded), (0, 0, 1))
        board.push(e.Move.reinforce(e.INFANTRY, e.A1))
        novel = sample(board, "third-game")
        training, validation, _ = restored.add_game([novel], family_id=validation_family)
        self.assertEqual((len(training), len(validation)), (0, 1))
        training, validation, excluded = restored.add_game([replace(novel, game_id="fourth")], family_id=train_family)
        self.assertEqual((len(training), len(validation), excluded), (0, 0, 1))
        self.assertFalse(restored.train_states & restored.validation_states)

    def test_duplicate_collection_updates_do_not_retrain_existing_replay(self):
        original_rows = []

        def duplicate_collect(tasks, evaluators, **options):
            if not original_rows:
                result = fixture_collect(tasks, evaluators, **options)
                original_rows.extend(result[0][0])
                return result
            results = []
            for index, task in enumerate(tasks):
                rows = [replace(row, game_id=task['game_id']) for row in original_rows]
                report = {'game_id': task['game_id'], 'teacher': False, 'truncated': False,
                          'terminal': {'winner': 'RED', 'reason': 'fixture'}, 'opponent': 'self'}
                options['on_result'](index, (rows, report))
                results.append((rows, report))
            return results

        args = self.args(self.root/'duplicates', '--teacher-positions', '0', '--iterations', '3',
                         '--games', '1', '--validation-fraction', '0', '--epochs', '10', '--replay-size', '1')
        with patch('ghqnet.curriculum_pipeline.fit', wraps=fit) as training:
            result = self.run_fixture(args, duplicate_collect)
        self.assertEqual(training.call_count, 1)
        self.assertEqual(training.call_args.kwargs['max_examples'], 2)
        ledger = result['replay_exposure']
        self.assertEqual(ledger['new_unique_training_states'], 2)
        self.assertEqual(ledger['trained_examples'], 2)
        self.assertEqual(float(ledger['credit']), 0)
        metrics = json.loads((args.output/'metrics-0002.json').read_text())['training']
        self.assertEqual(metrics['new_unique_training_states'], 0)
        self.assertEqual(metrics['trained_examples'], 0)
        states = [json.loads((args.output/f'curriculum-{index:04d}.json').read_text()) for index in range(3)]
        self.assertEqual(states[0]['history'], states[1]['history'])
        self.assertEqual(states[0]['history'], states[2]['history'])
        self.assertEqual(len(list(args.output.glob('history-*.pt'))), 1)
        self.assertEqual(len(read_replay(args.output/'replay-buffer-0002.jsonl')), 1)
        events = [json.loads(line) for line in (args.output/'events.jsonl').read_text().splitlines()]
        self.assertEqual([event['event'] for event in events if event['event'] in ('trained', 'checkpoint-saved')],
                         ['trained', 'checkpoint-saved', 'checkpoint-saved'])

    def test_fractional_credit_survives_resume_and_ratio_change_is_rejected(self):
        args = self.args(self.root/'fractional', '--teacher-positions', '0', '--games', '1',
                         '--validation-fraction', '0', '--replay-ratio', '.25')
        with patch('ghqnet.curriculum_pipeline.fit', side_effect=AssertionError('Less than one credit')):
            first = self.run_fixture(args)
        self.assertEqual(float(first['replay_exposure']['credit']), .5)
        resume = self.args(self.root/'resumed', '--teacher-positions', '0', '--games', '1',
                           '--validation-fraction', '0', '--replay-ratio', '.25', '--resume', first['checkpoint'])
        second = self.run_fixture(resume)
        self.assertEqual(second['replay_exposure']['new_unique_training_states'], 4)
        self.assertEqual(second['replay_exposure']['trained_examples'], 1)
        self.assertEqual(float(second['replay_exposure']['credit']), 0)
        rejected = self.args(self.root/'ratio-mismatch', '--teacher-positions', '0', '--games', '1',
                             '--validation-fraction', '0', '--replay-ratio', '.5', '--resume', second['checkpoint'])
        with self.assertRaisesRegex(ValueError, 'ratio must match'):
            self.run_fixture(rejected)
        self.assertFalse(rejected.output.exists())

    def test_epoch_upper_bound_carries_unspent_credit(self):
        args = self.args(self.root/'epoch-cap', '--teacher-positions', '0', '--games', '1',
                         '--validation-fraction', '0', '--replay-ratio', '3', '--epochs', '1')
        result = self.run_fixture(args)
        self.assertEqual(result['replay_exposure']['trained_examples'], 2)
        self.assertEqual(float(result['replay_exposure']['credit']), 4)
        checkpoint = Path(result['checkpoint'])
        _, saved = load_checkpoint(checkpoint)
        state = json.loads((checkpoint.parent/saved['metadata']['curriculum_state']).read_text())
        self.assertEqual(state['replay_exposure'], result['replay_exposure'])

    def test_interrupted_update_debits_only_completed_rows_and_resume_carries_credit(self):
        args = self.args(self.root/'interrupted-fit', '--teacher-positions', '0', '--games', '1',
                         '--validation-fraction', '0')

        def partial_fit(model, rows, **options):
            self.assertEqual(options['max_examples'], 2)
            optimizer, metrics = fit(model, rows, **{**options, 'max_examples': 1})
            metrics.update(interrupted=True, stopped_reason='deadline')
            return optimizer, metrics

        with patch('ghqnet.curriculum_pipeline.fit', side_effect=partial_fit):
            first = self.run_fixture(args)
        self.assertEqual(first['replay_exposure']['trained_examples'], 1)
        self.assertEqual(float(first['replay_exposure']['credit']), 1)
        resume = self.args(self.root/'after-interruption', '--teacher-positions', '0', '--games', '1',
                           '--validation-fraction', '0', '--resume', first['checkpoint'])
        with patch('ghqnet.curriculum_pipeline.fit', wraps=fit) as training:
            second = self.run_fixture(resume)
        self.assertEqual(training.call_args.kwargs['max_examples'], 3)
        self.assertEqual(second['replay_exposure']['trained_examples'], 4)
        self.assertEqual(float(second['replay_exposure']['credit']), 0)

    def test_legacy_resume_retains_replay_but_grants_no_retroactive_credit(self):
        args = self.args(self.root/'legacy-source', '--teacher-positions', '0', '--games', '1',
                         '--validation-fraction', '0')
        first = self.run_fixture(args)
        checkpoint_path = Path(first['checkpoint'])
        model, saved = load_checkpoint(checkpoint_path)
        metadata = saved['metadata']
        state_path = checkpoint_path.parent/metadata['curriculum_state']
        state = json.loads(state_path.read_text())
        del state['replay_exposure']
        state_path.write_text(json.dumps(state))
        metadata['curriculum_state_sha256'] = hashlib.sha256(state_path.read_bytes()).hexdigest()
        legacy = checkpoint_path.parent/'legacy.pt'
        optimizer = torch.optim.AdamW(model.parameters())
        optimizer.load_state_dict(saved['optimizer_state'])
        save_checkpoint(legacy, model, optimizer, **metadata)
        resume = self.args(self.root/'legacy-resume', '--teacher-positions', '0', '--games', '1',
                           '--validation-fraction', '0', '--resume', str(legacy))
        with patch('ghqnet.curriculum_pipeline.fit', wraps=fit) as training:
            second = self.run_fixture(resume)
        self.assertEqual(training.call_args.kwargs['max_examples'], 2)
        ledger = second['replay_exposure']
        self.assertEqual(ledger['legacy_uncredited_training_states'], 2)
        self.assertEqual(ledger['new_unique_training_states'], 2)
        self.assertEqual(ledger['trained_examples'], 2)
        self.assertIn('zero retroactive credit', ledger['migration'])
        _, saved = load_checkpoint(second['checkpoint'])
        self.assertEqual(len(read_replay(Path(second['checkpoint']).parent/saved['metadata']['replay_buffer'])), 4)

    def test_zero_ratio_saves_one_baseline_and_never_uses_current_model_as_history(self):
        tasks_seen = []

        def collect(tasks, evaluators, **options):
            tasks_seen.extend(tasks)
            return fixture_collect(tasks, evaluators, **options)

        args = self.args(self.root/'zero-ratio', '--teacher-positions', '0', '--games', '1',
                         '--iterations', '3', '--validation-fraction', '0', '--replay-ratio', '0',
                         '--cpu-fraction', '0', '--history-fraction', '1')
        with patch('ghqnet.curriculum_pipeline.fit', side_effect=AssertionError('No exposure budget')):
            result = self.run_fixture(args, collect)
        self.assertTrue(all(task['opponent'] == 'cpu' for task in tasks_seen))
        self.assertEqual(result['replay_exposure']['trained_examples'], 0)
        self.assertEqual(len(list(args.output.glob('history-*.pt'))), 1)

    def test_retention_preserves_teacher_and_hard_minimums(self):
        row = sample(e.BaseBoard(), "base")
        examples = ([replace(row, game_id=f"t{i}", source="ghq-cpu-4") for i in range(30)] +
                    [replace(row, game_id=f"h{i}", priority="hard") for i in range(20)] +
                    [replace(row, game_id=f"s{i}") for i in range(150)])
        retained = retain_replay(examples, 100, seed=27)
        self.assertEqual(len(retained), 100)
        self.assertGreaterEqual(sum(row.source.startswith("ghq-cpu") for row in retained), 20)
        self.assertGreaterEqual(sum(row.priority == "hard" for row in retained), 10)
        self.assertEqual(retained, retain_replay(examples, 100, seed=27))
        self.assertEqual(len({row.game_id for row in retained}), 100)

    def test_opponent_mix_is_seeded_and_color_balanced(self):
        jobs = opponent_schedule(200, seed=19)
        self.assertEqual(jobs, opponent_schedule(200, seed=19))
        for kind, expected in (("self", 80), ("cpu", 80), ("historical", 40)):
            selected = [job for job in jobs if job["opponent"] == kind]
            self.assertEqual(len(selected), expected)
            self.assertEqual(sum(job["learner_color"] == "RED" for job in selected), expected // 2)

    def test_opponent_schedule_is_identical_across_small_chunks_and_resume_offsets(self):
        expected = opponent_schedule(200, seed=31)
        sizes = [32, 1, 7, 32, 32, 32, 32, 32]
        actual, cursor = [], 0
        for size in sizes:
            actual.extend(opponent_schedule(size, seed=31, start_index=cursor))
            cursor += size
        self.assertEqual(cursor, 200)
        self.assertEqual(actual, expected)
        for kind in ("self", "cpu", "historical"):
            colors = [row["learner_color"] for row in actual if row["opponent"] == kind]
            self.assertTrue(all(a != b for a, b in zip(colors, colors[1:])))

    def test_isolated_game_timeouts_continue_until_cumulative_completed_target(self):
        calls = []

        def partial(tasks, evaluators, **options):
            calls.append([task["game_id"] for task in tasks])
            timed = [len(tasks) - 1] if len(tasks) > 1 else []
            results = fixture_collect(tasks[:-1] if timed else tasks, evaluators, **options)
            options["metrics"].update(timed_out_indices=timed, timed_out_games=len(timed),
                                      game_timeouts=[{"kind": "game_timeout", "index": i,
                                                      "game_id": tasks[i]["game_id"], "elapsed_s": 10}
                                                     for i in timed],
                                      stop_reason="completed_with_game_timeouts" if timed else "completed")
            for timeout in options["metrics"]["game_timeouts"]:
                options["on_event"]({"event": "game_timeout", **timeout})
            return results

        result = self.run_fixture(self.args(self.root / "run", "--teacher-positions", "0",
                                           "--validation-fraction", "0", "--iterations", "3",
                                           "--total-games", "5"), partial)
        self.assertEqual(result["status"], "curriculum-not-promoted")
        self.assertEqual(result["mixed_games"], 5)
        self.assertTrue(result["mixed_target_reached"])
        self.assertEqual(result["mixed_attempts"], 7)
        self.assertEqual(result["game_timeout_counts"]["mixed"], 2)
        self.assertEqual(len({game for call in calls for game in call}), 7)
        events = [json.loads(line) for line in (self.root / "run" / "events.jsonl").read_text().splitlines()]
        collections = [event for event in events if event["event"] == "collection"]
        self.assertEqual([event["stop_reason"]["kind"] for event in collections],
                         ["completed_with_game_timeouts", "completed_with_game_timeouts", "completed"])
        self.assertEqual([event["incomplete"] for event in collections], [True, True, False])
        timeouts = [event for event in events if event["event"] == "game-timeout"]
        self.assertEqual(len(timeouts), 2)
        self.assertTrue(all(event["kind"] == "game_timeout" and not event["teacher"] for event in timeouts))

    def test_all_game_timeouts_stop_without_training_and_resume_with_fresh_ids(self):
        def timed_out(tasks, evaluators, **options):
            options["metrics"].update(timed_out_indices=list(range(len(tasks))),
                                      stop_reason="completed_with_game_timeouts")
            return []

        args = self.args(self.root / "run", "--teacher-positions", "0", "--iterations", "20")
        with patch("ghqnet.curriculum_pipeline.fit", side_effect=AssertionError("No new data to fit")):
            first = self.run_fixture(args, timed_out)
        self.assertEqual(first["status"], "curriculum-no-progress")
        self.assertEqual(first["updates"], 1)
        self.assertEqual(first["mixed_games"], 0)
        self.assertEqual(first["mixed_attempts"], 4)
        self.assertEqual(first["stop_reason"]["kind"], "no_completed_games")
        resumed = self.run_fixture(self.args(self.root / "resumed", "--resume", first["checkpoint"],
                                             "--teacher-positions", "0", "--games", "2"))
        self.assertEqual(resumed["mixed_games"], 2)
        self.assertEqual(resumed["mixed_attempts"], 6)
        self.assertEqual({path.stem.rsplit("-", 1)[-1] for path in (self.root / "resumed" / "replay").glob("*.jsonl")},
                         {"0000004", "0000005"})

    def test_infrastructure_timeout_preserves_reason_and_does_not_train(self):
        def failed(tasks, evaluators, **options):
            results = fixture_collect(tasks[:1], evaluators, **options)
            options["metrics"].update(stop_reason="request_timeout",
                                      failure={"kind": "request_timeout", "game_id": tasks[1]["game_id"],
                                               "elapsed_s": 120, "message": "fixture request stalled"})
            raise CollectionTimeout("fixture request stalled", results, [0])

        with patch("ghqnet.curriculum_pipeline.fit", side_effect=AssertionError("Unsafe after infrastructure failure")):
            with self.assertRaisesRegex(CollectionTimeout, "fixture request stalled"):
                self.run_fixture(self.args(self.root / "run", "--teacher-positions", "0"), failed)
        summary = json.loads((self.root / "run" / "summary.json").read_text())
        self.assertEqual(summary["status"], "curriculum-collection-failed")
        self.assertEqual(summary["stop_reason"]["kind"], "request_timeout")
        self.assertEqual(summary["stop_reason"]["elapsed_s"], 120)
        self.assertEqual(summary["mixed_games"], 1)

    def test_historical_cohorts_use_one_snapshot_and_resume_the_same_rotation(self):
        first = self.run_fixture(self.args(self.root / "first", "--teacher-positions", "0",
                                           "--iterations", "4", "--games", "10"))
        _, payload = load_checkpoint(first["checkpoint"])
        initial_state = json.loads((Path(first["checkpoint"]).parent / payload["metadata"]["curriculum_state"]).read_text())
        captured = []

        def inspect(tasks, evaluators, **options):
            ids = {task["opponent_model_id"] for task in tasks if task["opponent"] == "historical"}
            self.assertEqual(len(ids), 1)
            self.assertEqual(len(evaluators), 2)
            captured.append((tasks, next(iter(ids))))
            return fixture_collect(tasks, evaluators, **options)

        result = self.run_fixture(self.args(self.root / "resumed", "--resume", first["checkpoint"],
                                            "--teacher-positions", "0", "--iterations", "2", "--games", "10"), inspect)
        eligible = [entry for entry in initial_state["history"] if entry["iteration"] < initial_state["last_iteration"]]
        self.assertEqual(captured[0][1], eligible[initial_state["history_cohort_cursor"] % len(eligible)]["sha256"])
        self.assertNotEqual(captured[0][1], captured[1][1])
        assignments = [{"opponent": task["opponent"], "learner_color": task["learner_color"]}
                       for tasks, _ in captured for task in tasks]
        self.assertEqual(assignments, opponent_schedule(20, seed=self.args(self.root / "x").seed,
                                                       start_index=initial_state["mixed_schedule_cursor"]))
        self.assertEqual(result["teacher_games"], 0)
        self.assertEqual(result["mixed_games"], 60)

    def test_legacy_resume_migrates_abandoned_cursor_and_clears_old_stop_reason(self):
        first = self.run_fixture(self.args(self.root / "first", "--teacher-positions", "0"))
        path = Path(first["checkpoint"])
        _, payload = load_checkpoint(path)
        state_path = path.parent / payload["metadata"]["curriculum_state"]
        state = json.loads(state_path.read_text())
        for key in ("teacher_attempts", "mixed_schedule_cursor", "history_cohort_cursor",
                    "game_timeout_counts", "opponent_schedule_config"):
            state.pop(key)
        state["last_collection"]["abandoned_game_ids"] = [f"old-abandoned-{i}" for i in range(3)]
        state["next_serial"] += 3
        state["stop_reason"] = {"kind": "global_deadline"}
        state_path.write_text(json.dumps(state))
        payload["metadata"]["curriculum_state_sha256"] = hashlib.sha256(state_path.read_bytes()).hexdigest()
        torch.save(payload, path)
        result = self.run_fixture(self.args(self.root / "resumed", "--resume", str(path),
                                            "--teacher-positions", "0", "--games", "2"))
        self.assertEqual(result["teacher_games"], 0)
        self.assertEqual(result["mixed_games"], 6)
        self.assertEqual(result["mixed_attempts"], 9)
        self.assertIsNone(result["stop_reason"])
        _, saved = load_checkpoint(result["checkpoint"])
        saved_state = json.loads((Path(result["checkpoint"]).parent / saved["metadata"]["curriculum_state"]).read_text())
        self.assertEqual(saved_state["next_serial"], 9)
        self.assertEqual(saved_state["last_collection"]["opponent_schedule_start"], 7)
        self.assertIsNone(saved_state["stop_reason"])

    def test_teacher_attempt_cap_counts_watchdogs_and_does_not_loop(self):
        teacher_calls = []

        def partial(tasks, evaluators, **options):
            if not tasks[0]["teacher"]:
                return fixture_collect(tasks, evaluators, **options)
            teacher_calls.append(len(tasks))
            results = fixture_collect(tasks[:1], evaluators, **options)
            options["metrics"].update(timed_out_indices=list(range(1, len(tasks))),
                                      stop_reason="completed_with_game_timeouts")
            return results

        result = self.run_fixture(self.args(self.root / "run", "--teacher-positions", "100",
                                           "--teacher-max-games", "3", "--teacher-update-games", "3"), partial)
        self.assertEqual(teacher_calls, [3])
        self.assertEqual(result["teacher_attempts"], 3)
        self.assertEqual(result["teacher_games"], 1)
        self.assertFalse(result["teacher_target_reached"])
        self.assertEqual(result["mixed_games"], 4)

    def test_validation_changes_metrics_but_never_optimizer_updates(self):
        board = e.BaseBoard()
        training = [sample(board, "train")]
        board.push(e.Move.reinforce(e.INFANTRY, e.A1))
        validation = [sample(board, "heldout", outcome=2)]
        model = PolicyValueNet(NetworkConfig(0, 4))
        reference = deepcopy(model)
        _, measured = fit(model, training, validation_samples=validation, epochs=1, batch_size=1, seed=9)
        fit(reference, training, epochs=1, batch_size=1, seed=9)
        self.assertEqual(measured["validation_samples"], 1)
        self.assertIsNotNone(measured["validation_after"])
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, reference.state_dict()[name])

    def test_expired_training_budget_does_not_prepare_or_mutate_weights(self):
        rows = [sample(e.BaseBoard(), "train")]
        model = PolicyValueNet(NetworkConfig(0, 4))
        reference = deepcopy(model.state_dict())
        with patch.object(Sample, "validate", side_effect=AssertionError("Expired fit must not validate records")):
            optimizer, metrics = fit(model, rows, max_seconds=0, validation_samples=rows)
        self.assertEqual(metrics["steps"], 0)
        self.assertTrue(metrics["preparation_interrupted"])
        self.assertEqual(len(optimizer.state), 0)
        for name, value in model.state_dict().items():
            torch.testing.assert_close(value, reference[name])

    def test_curriculum_resume_preserves_holdout_and_history_identity(self):
        first = self.run_fixture(self.args(self.root / "first"))
        checkpoint_path = Path(first["checkpoint"])
        _, checkpoint = load_checkpoint(checkpoint_path)
        metadata = checkpoint["metadata"]
        validation = read_replay(checkpoint_path.parent / metadata["validation_buffer"])
        state = json.loads((checkpoint_path.parent / metadata["curriculum_state"]).read_text())
        resumed = self.run_fixture(self.args(self.root / "resumed", "--resume", str(checkpoint_path), "--games", "2"))
        self.assertEqual(resumed["teacher_games"], first["teacher_games"])
        self.assertEqual(resumed["mixed_games"], first["mixed_games"] + 2)
        new_path = Path(resumed["checkpoint"])
        _, new_checkpoint = load_checkpoint(new_path)
        new_state = json.loads((new_path.parent / new_checkpoint["metadata"]["curriculum_state"]).read_text())
        self.assertTrue(set(state["partition"]["validation_states"]) <= set(new_state["partition"]["validation_states"]))
        training = read_replay(new_path.parent / new_checkpoint["metadata"]["replay_buffer"])
        self.assertFalse({fingerprint(row) for row in training} & {fingerprint(row) for row in validation})
        self.assertTrue(all((new_path.parent / entry["file"]).is_file() for entry in new_state["history"]))

    def test_partial_teacher_timeout_saves_and_resumes_remaining_bootstrap(self):
        def partial(tasks, evaluators, **options):
            results = fixture_collect(tasks[:1], evaluators, **options)
            raise CollectionTimeout("fixture deadline", results, [0])
        first = self.run_fixture(self.args(self.root / "first"), partial)
        self.assertEqual(first["status"], "curriculum-deadline")
        self.assertEqual(first["teacher_games"], 1)
        self.assertEqual(first["mixed_games"], 0)
        resumed = self.run_fixture(self.args(self.root / "resumed", "--resume", first["checkpoint"]))
        self.assertGreater(resumed["teacher_games"], 1)
        self.assertEqual(resumed["mixed_games"], 4)
        self.assertTrue(Path(resumed["checkpoint"]).exists())
        self.assertEqual(len(list((self.root / "first" / "replay").glob("*.jsonl"))), 1)

    def test_collection_failure_checkpoints_completed_game_without_training(self):
        first = self.run_fixture(self.args(self.root / "first", "--teacher-positions", "0",
                                          "--validation-fraction", "0", "--games", "2"))
        _, original = load_checkpoint(first["checkpoint"])
        failed_output = self.root / "failed"

        def partial(tasks, evaluators, **options):
            results = fixture_collect(tasks[:1], evaluators, **options)
            raise CollectionError("fixture inference failure", results, [0])

        with patch("ghqnet.curriculum_pipeline.fit", side_effect=AssertionError("Must not train after failure")):
            with self.assertRaisesRegex(CollectionError, "fixture inference failure"):
                self.run_fixture(self.args(failed_output, "--resume", first["checkpoint"],
                                           "--teacher-positions", "0", "--validation-fraction", "0"), partial)
        summary = json.loads((failed_output / "summary.json").read_text())
        self.assertEqual(summary["status"], "curriculum-collection-failed")
        self.assertEqual(summary["mixed_games"], first["mixed_games"] + 1)
        checkpoint_path = Path(summary["checkpoint"])
        _, saved = load_checkpoint(checkpoint_path)
        for name, value in original["model_state"].items():
            torch.testing.assert_close(value, saved["model_state"][name])
        self.assertEqual(original["optimizer_state"]["param_groups"], saved["optimizer_state"]["param_groups"])
        for parameter, values in original["optimizer_state"]["state"].items():
            for name, value in values.items():
                torch.testing.assert_close(value, saved["optimizer_state"]["state"][parameter][name])
        metadata = saved["metadata"]
        rows = read_replay(checkpoint_path.parent / metadata["replay_buffer"])
        completed_rows = read_replay(next((failed_output / "replay").glob("*.jsonl")))
        self.assertTrue(completed_rows)
        self.assertTrue({row.game_id for row in completed_rows} <= {row.game_id for row in rows})
        self.assertEqual(metadata["metrics"]["steps"], 0)
        state = json.loads((checkpoint_path.parent / metadata["curriculum_state"]).read_text())
        self.assertEqual(state["last_collection"]["completed"], 1)
        self.assertEqual(len(state["last_collection"]["abandoned_game_ids"]), 3)
        self.assertIn("fixture inference failure", state["last_collection_error"])
        # A failure snapshot carries the same exact-buffer references as a
        # successful update and can resume without reimporting the raw games.
        resumed = self.run_fixture(self.args(self.root / "recovered", "--resume", str(checkpoint_path),
                                            "--teacher-positions", "0", "--validation-fraction", "0", "--games", "1"))
        self.assertEqual(resumed["mixed_games"], summary["mixed_games"] + 1)

    def test_corrupt_validation_or_changed_split_rejected(self):
        first = self.run_fixture(self.args(self.root / "first"))
        checkpoint_path = Path(first["checkpoint"])
        _, checkpoint = load_checkpoint(checkpoint_path)
        with self.assertRaisesRegex(ValueError, "validation fraction"):
            self.run_fixture(self.args(self.root / "bad-split", "--resume", str(checkpoint_path), "--validation-fraction", "0.2"))
        path = checkpoint_path.parent / checkpoint["metadata"]["validation_buffer"]
        path.write_bytes(path.read_bytes() + b"\n")
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_fixture(self.args(self.root / "bad-hash", "--resume", str(checkpoint_path)))

    def test_reserved_game_never_enters_training_or_validation(self):
        reserved = "curr-20261120-teacher-0000000"
        manifest = self.root / "reserved.manifest.json"
        manifest.write_text(json.dumps({"exclusions": {"game_ids": [reserved]}}))
        result = self.run_fixture(self.args(self.root / "run", "--exclude-manifest", str(manifest)))
        path = Path(result["checkpoint"])
        _, checkpoint = load_checkpoint(path)
        for key in ("replay_buffer", "validation_buffer"):
            self.assertNotIn(reserved, {row.game_id for row in read_replay(path.parent / checkpoint["metadata"][key])})
        state = json.loads((path.parent / checkpoint["metadata"]["curriculum_state"]).read_text())
        self.assertGreaterEqual(state["excluded_games"], 1)
        self.assertEqual(state["exclusion_manifest_sha256"], [hashlib.sha256(manifest.read_bytes()).hexdigest()])
        self.assertEqual(state['replay_exposure']['new_unique_training_states'], len(state['partition']['train_states']))
        self.assertTrue(state['partition']['validation_states'])
        self.assertEqual(state['replay_exposure']['trained_examples'], len(state['partition']['train_states']))

    def test_reserved_replay_import_is_rejected(self):
        source = self.root / "reserved.jsonl"
        write_replay(source, [sample(e.BaseBoard(), "reserved")])
        manifest = self.root / "reserved.manifest.json"
        manifest.write_text(json.dumps({"exclusions": {"replay_sha256": [hashlib.sha256(source.read_bytes()).hexdigest()]}}))
        with self.assertRaisesRegex(ValueError, "reserved for evaluation"):
            self.run_fixture(self.args(self.root / "run", "--input-replay", str(source),
                                       "--exclude-manifest", str(manifest)))

    def test_validation_only_has_no_outputs_and_rejects_invalid_configs(self):
        output = self.root / "unused"
        with redirect_stdout(io.StringIO()):
            report = pipeline.run(self.args(output, "--validate-only"))
        self.assertTrue(report["valid"])
        self.assertFalse(output.exists())
        for flags in (("--workers", "65"), ("--inference-batch-size", "65"),
                      ("--cpu-fraction", "nan"), ("--max-seconds", "inf"),
                      ("--cpu-fraction", "0.9", "--history-fraction", "0.2")):
            with self.subTest(flags=flags), self.assertRaises(ValueError):
                pipeline.run(self.args(output, "--validate-only", *flags))


if __name__ == "__main__":
    unittest.main()
