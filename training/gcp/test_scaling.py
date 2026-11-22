"""Offline contracts for bounded larger runs; no cloud resources are accessed."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile


def load(name):
    spec = importlib.util.spec_from_file_location("scaling_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner, worker, builder = load("run"), load("worker"), load("build_source")


class ScalingTests(unittest.TestCase):
    def test_pipeline_config_is_argv_and_preserves_scalars(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps({"curriculum": True, "games": 200, "cpu-fraction": .4,
                                        "start-pool": "/resume/safe name.jsonl", "unused": False}))
            args, config, digest = runner.load_pipeline_config(path)
        self.assertEqual(args, ["--curriculum", "--games", "200", "--cpu-fraction", "0.4",
                                "--start-pool", "/resume/safe name.jsonl"])
        self.assertEqual(len(digest), 64)
        self.assertTrue(config["curriculum"])

    def test_pipeline_config_cannot_override_worker_boundaries(self):
        for config in ({"output": "/tmp/elsewhere"}, {"device": "cpu"}, {"resume": "evil"},
                       {"games": [1, 2]}, {"games": float("nan")}, {"--help": True},
                       {"teacher-ms": "--help"}, {"validate-only": True}):
            with self.subTest(config=config), tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / "config.json"
                path.write_text(json.dumps(config))
                with self.assertRaises(ValueError):
                    runner.load_pipeline_config(path)

    def test_larger_t4_checks_actual_cpu_requirement(self):
        regional = [{"metric": "NVIDIA_T4_GPUS", "limit": 1, "usage": 0},
                    {"metric": "CPUS", "limit": 15, "usage": 0}]
        self.assertEqual(runner.machine_type("t4", 16), "n1-standard-16")
        with self.assertRaisesRegex(RuntimeError, "CPUS"):
            runner.check_gpu_quotas(regional, [], "t4", 16)
        regional[1]["limit"] = 16
        runner.check_gpu_quotas(regional, [], "t4", 16)
        with self.assertRaises(ValueError):
            runner.machine_type("l4", 16)

    def sources(self, folder):
        root = Path(folder) / "repo"
        (root / "ghqnet").mkdir(parents=True)
        (root / "public").mkdir()
        for name in ("ghqnet/__init__.py", "ghqnet/pipeline.py", "public/engine.py", "public/search.py"):
            (root / name).write_text("# reviewed source\n")
        (root / "secret.env").write_text("never bundle")
        (root / "ghqnet/private.json").write_text("never bundle")
        return root

    def test_source_bundle_is_reproducible_and_allowlisted(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.sources(folder)
            first = Path(folder) / "first.zip"
            second = Path(folder) / "second.zip"
            self.assertEqual(builder.build(root, first)["sha256"], builder.build(root, second)["sha256"])
            destination = Path(folder) / "unpacked"
            manifest = worker.extract_source_bundle(first, destination)
            self.assertEqual(len(manifest["files"]), 4)
            self.assertEqual((destination / "ghqnet/pipeline.py").read_text(), "# reviewed source\n")
            self.assertFalse((destination / "secret.env").exists())
            with self.assertRaises(ValueError):
                builder.build(root, first)

    def test_source_bundle_detects_payload_tampering_and_unexpected_paths(self):
        for change in ("tamper", "../escape.py", "ghqnet/credentials.json"):
            with self.subTest(change=change), tempfile.TemporaryDirectory() as folder:
                root = self.sources(folder)
                clean, bad = Path(folder) / "clean.zip", Path(folder) / "bad.zip"
                builder.build(root, clean)
                with zipfile.ZipFile(clean) as source, zipfile.ZipFile(bad, "w") as output:
                    for entry in source.infolist():
                        data = source.read(entry)
                        if change == "tamper" and entry.filename == "ghqnet/pipeline.py":
                            data = b"# tampered source\n"
                        output.writestr(entry, data)
                    if change != "tamper":
                        output.writestr(change, "bad")
                with self.assertRaises((ValueError, RuntimeError)):
                    worker.extract_source_bundle(bad, Path(folder) / "unpacked")

    def test_source_bundle_preserves_only_evaluation_manifests(self):
        with tempfile.TemporaryDirectory() as folder:
            root = self.sources(folder)
            benchmarks = root / "training/benchmarks"
            benchmarks.mkdir(parents=True)
            (benchmarks / "reserved.manifest.json").write_text('{"exclusions": {}}')
            (benchmarks / "cases.jsonl").write_text("not a training input")
            archive = Path(folder) / "source.zip"
            builder.build(root, archive)
            destination = Path(folder) / "unpacked"
            worker.extract_source_bundle(archive, destination)
            self.assertEqual((destination / "training/benchmarks/reserved.manifest.json").read_text(),
                             '{"exclusions": {}}')
            self.assertFalse((destination / "training/benchmarks/cases.jsonl").exists())

    def test_code_overlay_is_readonly_and_does_not_change_output_mount(self):
        command = worker.training_command({"source_directory": "/opt/source", "image_id": "sha256:test",
                                           "pipeline_args": ["--curriculum"]}, Path("/host/output"))
        self.assertIn("type=bind,source=/opt/source,target=/app,readonly", command)
        self.assertIn("type=bind,source=/host/output,target=/output", command)
        self.assertEqual(command[command.index("--output") + 1], "/output/run")
        self.assertIn("--network=none", command)

    def test_resume_accepts_only_flat_curriculum_companions(self):
        names = ["checkpoint.pt", "replay-buffer-0003.jsonl", "validation-buffer-0003.jsonl",
                 "curriculum-0003.json", "history-0123456789abcdef.pt"]
        with tempfile.TemporaryDirectory() as folder:
            archive = Path(folder) / "resume.zip"
            with zipfile.ZipFile(archive, "w") as output:
                for name in names:
                    output.writestr(name, "trusted artifact")
            destination = Path(folder) / "unpacked"
            worker.extract_resume_bundle(archive, destination)
            self.assertEqual(sorted(p.name for p in destination.iterdir()), sorted(names))


if __name__ == "__main__":
    unittest.main()
