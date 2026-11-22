"""Frozen experiment preparation/assembly without inference or real collection."""
from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

try:
    import numpy
    import torch
except ImportError:
    raise unittest.SkipTest("Neural experiment tests require optional dependencies")

from ghqnet import value_experiment_data as experiment
from ghqnet.ablate_replay import file_sha
from ghqnet.collector import CollectionError
from ghqnet.curriculum import fingerprint
from ghqnet.data import Sample, read_replay, write_replay
from ghqnet.rules import e


def row(index, *, family=None, game=None, mirrored=False):
    board = e.BaseBoard()
    # Extra infantry distinguishes otherwise ordinary, nonterminal turn starts.
    square = e.parse_square("abcdefgh"[index % 8] + str(3 + index // 8))
    board.set_piece_at(square, e.Piece.from_symbol("I"))
    if mirrored:
        board = board.mirror()
    actions = sorted(move.uci() for move in board.generate_legal_moves())
    return Sample(board.serialize(), actions, [float(action == "skip") for action in actions],
                  outcome=1, game_id=game or f"game-{index}", family_id=family or f"family-{index}",
                  source="mcts:legacy")


class ValueExperimentDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def config(self, training=None, validation=None, *, fresh_train=1, fresh_validation=1, excluded=()):
        training = [row(0, family="training-family")] if training is None else training
        validation = [row(1, family="validation-family")] if validation is None else validation
        checkpoint = self.root / "checkpoint.pt"
        checkpoint.write_bytes(b"opaque frozen checkpoint: never load for preparation")
        train_path, validation_path = self.root / "training.jsonl", self.root / "validation.jsonl"
        write_replay(train_path, training, kind="training")
        write_replay(validation_path, validation, kind="validation")
        partition = self.root / "partition.json"
        partition.write_text(json.dumps({
            "partition": {"train_states": sorted({fingerprint(item) for item in training}),
                          "validation_states": sorted({fingerprint(item) for item in validation})},
            "exclusions": {"canonical_state_ids": list(excluded), "game_ids": [], "family_ids": []}}))
        config = {"inputs": {"checkpoint": str(checkpoint), "training_roots": str(train_path),
                              "validation_roots": str(validation_path), "partition": str(partition)},
                  "checkpoint_sha256": file_sha(checkpoint), "seed": 20261126,
                  "collection": {"training_games": 1, "validation_games": 1, "simulations": 1,
                                 "max_turns": 2, "workers": 1, "threads": 1,
                                 "inference_batch_size": 1, "game_timeout_seconds": 10, "max_seconds": 20},
                  "panels": {"train_fresh": fresh_train, "validation_fresh": fresh_validation,
                             "train_legacy": 1, "validation_legacy": 1}}
        path = self.root / "config.json"
        path.write_text(json.dumps(config))
        return path, config

    def prepared(self, **kwargs):
        config, raw = self.config(**kwargs)
        folder = self.root / "prepared"
        manifest = experiment.prepare(config, folder)
        return folder, manifest, raw

    def game(self, folder, entry, rows):
        identity = entry["task"]["game_id"]
        task = entry["task"]
        final = e.BaseBoard()
        final.did_offer_draw = final.did_accept_draw = True
        terminal = {"winner": None, "reason": "double-skip"}
        player = {"model_id": task["model_id"], "search": {
            "kind": "neural", "simulations": task["simulations"], "cpuct": 1.5,
            "max_nodes": 2048, "max_depth": 256, "root_noise": task["add_noise"],
            "dirichlet_alpha": 0.3, "noise_fraction": 0.25,
            "temperature_turns": task["temperature_turns"], "temperature": 1.0, "late_temperature": 0.0}}
        provenance = {"version": 1, "collection_kind": "self", "players": {"RED": player, "BLUE": player},
                      "terminal": terminal, "termination_reason": "double-skip", "final_state": final.serialize()}
        rows = [replace(item, family_id=entry["family_id"], game_id=identity,
                        value_provenance=provenance) for item in rows]
        report = {"game_id": identity, "terminal": terminal, "reason": "double-skip",
                  "seed": task["seed"], "model_id": task["model_id"], "value_provenance": provenance}
        path = folder / "games" / (identity + ".jsonl")
        write_replay(path, rows, game=report, family_id=entry["family_id"])
        return path

    def test_prepare_freezes_inputs_and_exact_root_family_without_loading_model(self):
        config, raw = self.config()
        before = {name: file_sha(path) for name, path in raw["inputs"].items()}
        output = self.root / "prepared"
        with patch("ghqnet.model.load_checkpoint", side_effect=AssertionError("No model loading")), \
             patch("ghqnet.model.TorchEvaluator", side_effect=AssertionError("No inference")), \
             patch("ghqnet.collector.collect_games", side_effect=AssertionError("No games")), \
             patch("ghqnet.train.fit", side_effect=AssertionError("No training")):
            manifest = experiment.prepare(config, output)
        self.assertTrue(manifest["prepared_only"])
        self.assertFalse(manifest["training_started"])
        self.assertEqual(len(manifest["tasks"]), 2)
        self.assertEqual({name: value["sha256"] for name, value in manifest["inputs"].items()}, before)
        self.assertEqual(before, {name: file_sha(path) for name, path in raw["inputs"].items()})
        self.assertEqual(manifest["config_sha256"], file_sha(config))
        originals = {"train": row(0, family="training-family"), "validation": row(1, family="validation-family")}
        for entry in manifest["tasks"]:
            parent = originals[entry["split"]]
            self.assertEqual(entry["family_id"], parent.family_id)
            self.assertEqual(entry["parent_game_id"], parent.game_id)
            self.assertEqual(entry["task"]["start_state"], parent.state)
            self.assertEqual(entry["canonical_root"], fingerprint(parent))
            self.assertEqual(entry["task"]["model_id"], manifest["model_id"])
        self.assertEqual(experiment.load_plan(output), manifest)
        self.assertEqual({item.name for item in output.iterdir()}, {"collection-plan.json"})
        with self.assertRaisesRegex(ValueError, "new directory"):
            experiment.prepare(config, output)

    def test_frozen_inputs_are_checked_before_collection_or_assembly(self):
        folder, manifest, _ = self.prepared()
        input_path = Path(manifest["inputs"]["checkpoint"]["path"])
        input_path.write_bytes(input_path.read_bytes() + b"changed")
        with self.assertRaisesRegex(ValueError, "Frozen input changed"):
            experiment.load_plan(folder)
        with self.assertRaisesRegex(ValueError, "Frozen input changed"):
            experiment.assemble(folder)
        self.assertFalse((folder / "dataset").exists())

    def test_prepare_rejects_cross_split_canonical_mirrors_before_output(self):
        training = [row(0, family="training")]
        validation = [row(0, family="validation", game="other", mirrored=True)]
        config, _ = self.config(training, validation)
        output = self.root / "prepared"
        with self.assertRaisesRegex(ValueError, "overlapping"):
            experiment.prepare(config, output)
        self.assertFalse(output.exists())

    def test_root_selection_is_family_distinct_and_honors_canonical_exclusion(self):
        candidates = [row(0, family="a"), row(2, family="a"), row(3, family="b"), row(4, family="c")]
        mirrored = row(0, family="mirror", mirrored=True)
        selected = experiment.roots(candidates + [mirrored], 2, 3, {fingerprint(candidates[-1])})
        self.assertEqual(len({item.family_id for item in selected}), 2)
        self.assertEqual(len({fingerprint(item) for item in selected}), 2)
        self.assertNotIn(fingerprint(candidates[-1]), {fingerprint(item) for item in selected})
        with self.assertRaisesRegex(ValueError, "independent legal root families"):
            experiment.roots(candidates, 4, 3, set())

    def test_panel_rounds_balance_families_deduplicate_mirrors_and_apply_cap(self):
        rows = [row(index, family="a" if index < 4 else "b" if index < 7 else "c") for index in range(10)]
        rows.append(row(0, family="b", mirrored=True, game="mirror"))
        selected = experiment.balanced_panel(rows, 6, 5, max_per_family=2)
        self.assertEqual(Counter(item.family_id for item in selected), {"a": 2, "b": 2, "c": 2})
        self.assertEqual(len({fingerprint(item) for item in selected}), 6)
        self.assertEqual(selected, experiment.balanced_panel(rows, 6, 5, max_per_family=2))
        with self.assertRaisesRegex(ValueError, "within family caps"):
            experiment.balanced_panel(rows, 7, 5, max_per_family=2)

    def test_incomplete_fixed_collection_never_builds_a_successful_subset(self):
        folder, manifest, _ = self.prepared()
        entry = next(item for item in manifest["tasks"] if item["split"] == "validation")
        self.game(folder, entry, [row(3)])
        with patch.object(experiment, "value_eligible", return_value=True), \
             self.assertRaisesRegex(ValueError, "Incomplete fixed collection"):
            experiment.assemble(folder)
        self.assertFalse((folder / "dataset").exists())

    def test_assembly_preserves_split_excludes_old_and_fresh_mirrors_and_keeps_fixed_sizes(self):
        reserved = row(10)
        folder, manifest, raw = self.prepared(fresh_validation=2, excluded=[fingerprint(reserved)])
        tasks = {entry["split"]: entry for entry in manifest["tasks"]}
        # Existing same-split inputs must not be sampled again as fresh labels.
        # Holdout-first processing also reserves a shared new continuation.
        validation = [row(1), row(0, mirrored=True), row(3), row(4), reserved]
        training = [row(0), row(1, mirrored=True), row(3, mirrored=True), row(5)]
        self.game(folder, tasks["validation"], validation)
        self.game(folder, tasks["train"], training)
        frozen = {name: file_sha(path) for name, path in raw["inputs"].items()}
        with patch.object(experiment, "value_eligible", return_value=True) as eligibility:
            report = experiment.assemble(folder)
        self.assertTrue(eligibility.called)
        self.assertTrue(all(call.args[1:] == ("selfplay", [manifest["model_id"]]) for call in eligibility.call_args_list))
        self.assertEqual(report["panel_rows"], {"train": 2, "validation": 3})
        self.assertEqual(report["omitted"], {"validation_collision": 2, "train_collision": 2})
        self.assertFalse(report["promoted"])
        training_panel = read_replay(folder / "dataset" / "training.jsonl")
        validation_panel = read_replay(folder / "dataset" / "validation.jsonl")
        train_ids = {fingerprint(item) for item in training_panel}
        val_ids = {fingerprint(item) for item in validation_panel}
        self.assertEqual(train_ids, {fingerprint(row(0)), fingerprint(row(5))})
        self.assertEqual(val_ids, {fingerprint(row(1)), fingerprint(row(3)), fingerprint(row(4))})
        self.assertFalse(train_ids & val_ids)
        self.assertEqual({item.family_id for item in training_panel}, {tasks["train"]["family_id"]})
        self.assertEqual({item.family_id for item in validation_panel}, {tasks["validation"]["family_id"]})
        state = json.loads((folder / "dataset" / "partition.json").read_text())["partition"]
        self.assertFalse(set(state["train_states"]) & set(state["validation_states"]))
        self.assertTrue(train_ids <= set(state["train_states"]))
        self.assertTrue(val_ids <= set(state["validation_states"]))
        self.assertNotIn(fingerprint(reserved), set(state["train_states"]) | set(state["validation_states"]))
        self.assertEqual(frozen, {name: file_sha(path) for name, path in raw["inputs"].items()})
        with self.assertRaisesRegex(ValueError, "must be new"):
            experiment.assemble(folder)

    def test_assembly_rejects_wrong_family_and_does_not_write_partial_dataset(self):
        folder, manifest, _ = self.prepared()
        for entry in manifest["tasks"]:
            path = self.game(folder, entry, [row(3 if entry["split"] == "train" else 4)])
            if entry["split"] == "validation":
                lines = path.read_text().splitlines()
                record = json.loads(lines[1])
                record["family_id"] = "incorrect-family"
                lines[1] = json.dumps(record)
                path.write_text("\n".join(lines) + "\n")
        with patch.object(experiment, "value_eligible", return_value=True), \
             self.assertRaisesRegex(ValueError, "family or game identity"):
            experiment.assemble(folder)
        self.assertFalse((folder / "dataset").exists())

    def test_interrupted_collection_records_attempts_before_dispatch_and_never_rerolls(self):
        folder, manifest, _ = self.prepared()

        def interrupted(tasks, evaluators, *, on_result, metrics, **kwargs):
            persisted = json.loads((folder / "collection-status.json").read_text())
            self.assertEqual(persisted["attempted"], [0, 1])
            first = tasks[0]
            sample = replace(row(3), game_id=first["game_id"], family_id="ignored-parent")
            on_result(0, ([sample], {"game_id": first["game_id"], "terminal": None, "reason": "max-turns"}))
            metrics["completed_games"] = 1
            raise CollectionError("synthetic worker interruption")

        with patch("torch.set_num_threads"), \
             patch("ghqnet.model.load_checkpoint", return_value=(object(), {})) as checkpoint, \
             patch("ghqnet.model.TorchEvaluator", return_value=object()), \
             patch("ghqnet.train.device_name", return_value="cpu"), \
             patch("ghqnet.train.fit", side_effect=AssertionError("No training during collection")), \
             patch("ghqnet.collector.collect_games", side_effect=interrupted) as collector:
            with self.assertRaisesRegex(CollectionError, "synthetic worker"):
                experiment.collect(folder)
            with self.assertRaisesRegex(ValueError, "explicit recovery"):
                experiment.collect(folder)
            status = json.loads((folder / "collection-status.json").read_text())
            self.assertEqual(collector.call_count, 1)
            self.assertEqual(checkpoint.call_count, 1)
        self.assertEqual(status["attempted"], [0, 1])
        self.assertEqual(len(status["runs"]), 1)
        self.assertEqual(status["runs"][0]["metrics"]["completed_games"], 1)
        self.assertIn("synthetic worker", status["runs"][0]["metrics"]["error"])
        first, second = manifest["tasks"]
        saved = read_replay(folder / "games" / (first["task"]["game_id"] + ".jsonl"))
        self.assertEqual(saved[0].family_id, first["family_id"])
        self.assertFalse((folder / "games" / (second["task"]["game_id"] + ".jsonl")).exists())

    def test_collection_resume_dispatches_only_unattempted_root_and_keeps_parent_mapping(self):
        folder, manifest, _ = self.prepared()
        first, second = manifest["tasks"]
        original = self.game(folder, first, [row(3)])
        original_hash = file_sha(original)
        (folder / "collection-status.json").write_text(json.dumps({"attempted": [0], "runs": []}))

        def one_remaining(tasks, evaluators, *, on_result, metrics, **kwargs):
            self.assertEqual(tasks, [second["task"]])
            sample = replace(row(4), game_id=tasks[0]["game_id"], family_id="overwrite-me")
            on_result(0, ([sample], {"game_id": tasks[0]["game_id"], "terminal": None, "reason": "max-turns"}))
            metrics["completed_games"] = 1

        with patch("torch.set_num_threads"), \
             patch("ghqnet.model.load_checkpoint", return_value=(object(), {})), \
             patch("ghqnet.model.TorchEvaluator", return_value=object()), \
             patch("ghqnet.train.device_name", return_value="cpu"), \
             patch("ghqnet.collector.collect_games", side_effect=one_remaining):
            status = experiment.collect(folder)
        self.assertEqual(status["runs"][0]["indices"], [1])
        self.assertEqual(file_sha(original), original_hash)
        path = folder / "games" / (second["task"]["game_id"] + ".jsonl")
        saved = read_replay(path)
        self.assertEqual(saved[0].family_id, second["family_id"])
        header = json.loads(path.read_text().splitlines()[0])
        self.assertEqual(header["metadata"]["game"]["parent_game_id"], second["parent_game_id"])
        self.assertEqual(header["metadata"]["game"]["experiment_split"], second["split"])


if __name__ == "__main__":
    unittest.main()
