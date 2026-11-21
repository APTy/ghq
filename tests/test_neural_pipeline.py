"""Checkpoint resume preserves exactly its own bounded replay buffer."""
from contextlib import redirect_stdout
from dataclasses import asdict
import hashlib
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Neural pipeline tests require optional Torch training dependencies")

from ghqnet import pipeline
from ghqnet.data import Sample, read_replay, write_replay
from ghqnet.model import load_checkpoint, save_checkpoint
from ghqnet.rules import e


def fixture_game(evaluator=None, **options):
    board = e.BaseBoard()
    actions = sorted(move.uci() for move in board.generate_legal_moves())
    sample = Sample(board.serialize(), actions, [1.0] + [0.0] * (len(actions) - 1),
                    outcome=0, game_id=options["game_id"], source="fixture")
    return [sample], {"game_id": options["game_id"], "truncated": False,
                      "terminal": {"winner": "RED", "reason": "fixture"}}


class NeuralPipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def arguments(self, output, resume=None):
        argv = ["--output", str(output), "--device", "cpu", "--threads", "1",
                "--iterations", "1", "--games", "1", "--teacher-games", "3",
                "--simulations", "1", "--max-turns", "1", "--epochs", "1",
                "--batch-size", "2", "--replay-size", "2", "--blocks", "0", "--channels", "4"]
        if resume is not None:
            argv += ["--resume", str(resume)]
        return pipeline.parser().parse_args(argv)

    def run_pipeline(self, arguments):
        with patch.object(pipeline, "play_game", side_effect=fixture_game), redirect_stdout(io.StringIO()):
            return pipeline.run(arguments)

    def first_checkpoint(self):
        directory = self.root / "original"
        self.run_pipeline(self.arguments(directory))
        return directory / "checkpoint-0001.pt"

    def test_resume_loads_exact_checkpoint_buffer_not_destination_replays(self):
        checkpoint = self.first_checkpoint()
        _, saved = load_checkpoint(checkpoint)
        metadata = saved["metadata"]
        buffer = checkpoint.parent / metadata["replay_buffer"]
        self.assertEqual(metadata["replay_buffer_sha256"], hashlib.sha256(buffer.read_bytes()).hexdigest())
        prior = read_replay(buffer)
        self.assertEqual(len(prior), 2)
        self.assertEqual(Path(metadata["replay_buffer"]).name, metadata["replay_buffer"])

        destination = self.root / "fork"
        poison, _ = fixture_game(game_id="unrelated-stale-game")
        write_replay(destination / "replay" / "0000-unrelated.jsonl", poison)
        arguments = self.arguments(destination, checkpoint)
        arguments.learning_rate = 0.9  # Existing optimizer settings win on resume.
        trained = []
        original_fit = pipeline.fit

        def capture_fit(model, samples, **options):
            trained.append([asdict(sample) for sample in samples])
            return original_fit(model, samples, **options)

        with patch.object(pipeline, "fit", side_effect=capture_fit):
            self.run_pipeline(arguments)
        incoming, _ = fixture_game(game_id="0002-selfplay-0000")
        expected = random.Random(arguments.seed + 2).sample(prior + incoming, 2)
        self.assertEqual(trained, [[asdict(sample) for sample in expected]])
        _, resumed = load_checkpoint(destination / "checkpoint-0002.pt")
        self.assertEqual(resumed["metadata"]["seed"], saved["metadata"]["seed"])
        self.assertEqual(resumed["optimizer_state"]["param_groups"][0]["lr"], 1e-3)
        configuration = json.loads((destination / "run-0002.json").read_text())
        self.assertEqual(configuration["effective_learning_rates"], [1e-3])
        self.assertEqual(read_replay(destination / resumed["metadata"]["replay_buffer"]), expected)

    def test_buffer_hash_mismatch_is_rejected_before_creating_output(self):
        checkpoint = self.first_checkpoint()
        _, saved = load_checkpoint(checkpoint)
        buffer = checkpoint.parent / saved["metadata"]["replay_buffer"]
        buffer.write_bytes(buffer.read_bytes() + b"\n")
        output = self.root / "rejected"
        with self.assertRaisesRegex(ValueError, "hash mismatch"):
            self.run_pipeline(self.arguments(output, checkpoint))
        self.assertFalse(output.exists())

    def test_resume_seed_mismatch_is_rejected(self):
        checkpoint = self.first_checkpoint()
        output = self.root / "rejected"
        arguments = self.arguments(output, checkpoint)
        arguments.seed += 1
        with self.assertRaisesRegex(ValueError, "seed must match"):
            self.run_pipeline(arguments)
        self.assertFalse(output.exists())

    def test_older_checkpoint_without_exact_buffer_metadata_is_rejected(self):
        checkpoint = self.first_checkpoint()
        model, saved = load_checkpoint(checkpoint)
        legacy = checkpoint.parent / "legacy.pt"
        save_checkpoint(legacy, model, iteration=1, seed=saved["metadata"]["seed"])
        with self.assertRaisesRegex(ValueError, "exact replay-buffer reference"):
            self.run_pipeline(self.arguments(self.root / "rejected", legacy))

    def test_resume_refuses_future_checkpoint_metrics_or_run_configuration(self):
        checkpoint = self.first_checkpoint()
        events = (checkpoint.parent / "events.jsonl").read_bytes()
        for filename in ("checkpoint-0003.pt", "metrics-0002.json", "run-0002.json"):
            with self.subTest(filename=filename):
                existing = checkpoint.parent / filename
                existing.write_text("preserve this existing branch")
                with self.assertRaisesRegex(ValueError, "overwrite existing run output"):
                    self.run_pipeline(self.arguments(checkpoint.parent, checkpoint))
                self.assertEqual(existing.read_text(), "preserve this existing branch")
                self.assertEqual((checkpoint.parent / "events.jsonl").read_bytes(), events)
                existing.unlink()


if __name__ == "__main__":
    unittest.main()
