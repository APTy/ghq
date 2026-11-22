"""Offline launch/worker contracts for the standalone collector benchmark."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


def load(name):
    spec = importlib.util.spec_from_file_location("collector_cloud_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner, worker = load("run"), load("worker")


class CollectorConfigurationTests(unittest.TestCase):
    def read(self, config):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "config.json"
            path.write_text(json.dumps(config))
            return runner.load_collector_benchmark_config(path)

    def test_allowlisted_lists_booleans_and_resume_inputs_become_argv(self):
        config = {"max-seconds": 900, "workers": [16, 32], "simulations": [128, 256],
                  "batch-size": [16, 32], "batch-wait-ms": [0, 1, 5],
                  "inference-protocol": ["legacy", "encoded"], "profile-inference": True,
                  "positions": "/resume/replay-buffer-0004.jsonl", "position-count": 16,
                  "historical-checkpoint": "/resume/history-0123456789abcdef.pt", "history-fraction": .2,
                  "cpu-fraction": .4, "repeats": 2, "case-max-seconds": 45, "teacher-ms": 3200,
                  "teacher-depth": 4, "teacher-beam-width": 12}
        arguments, returned, digest = self.read(config)
        self.assertEqual(returned, config)
        self.assertEqual(len(digest), 64)
        self.assertEqual(arguments[:6], ["--max-seconds", "900", "--workers", "16", "32", "--simulations"])
        self.assertIn("--profile-inference", arguments)
        self.assertIn("/resume/replay-buffer-0004.jsonl", arguments)
        self.assertNotIn("True", arguments)
        self.assertNotIn("--profile-inference", self.read({"max-seconds": 5, "profile-inference": False})[0])
        self.assertIn("--profile-sample", self.read({"max-seconds": 5, "profile-sample": True})[0])

    def test_rejects_paths_training_flags_shell_text_and_invalid_counts(self):
        configs = [{"output": "/output/evil"}, {"device": "cpu"}, {"epochs": 3}, {"resume": "evil"},
                   {"workers": []}, {"workers": [16, 16]}, {"workers": [65]}, {"workers": True},
                   {"batch-wait-ms": [float("nan")]}, {"batch-wait-ms": [0, 0.0]}, {"max-seconds": float("inf")},
                   {"max-seconds": 0}, {"max-seconds": "900; touch /tmp/evil"},
                   {"profile-inference": "true"}, {"inference-protocol": ["encoded", "--help"]},
                   {"profile-inference": True, "profile-sample": True},
                   {"positions": "/resume/../secret.jsonl"}, {"positions": "$(touch /tmp/evil)"},
                   {"historical-checkpoint": "/output/history-0123456789abcdef.pt"},
                   {"history-fraction": .2}, {"cpu-fraction": 1.1}, {"teacher-ms": -1},
                   {"cpu-fraction": .8, "history-fraction": .4,
                    "historical-checkpoint": "/resume/history-0123456789abcdef.pt"}]
        for options in configs:
            with self.subTest(options=options), self.assertRaises(ValueError):
                self.read({"max-seconds": 900, **options})
        with self.assertRaisesRegex(ValueError, "explicit max-seconds"):
            self.read({"games": 1})


class CollectorLaunchTests(unittest.TestCase):
    def argv(self):
        return ["run.py", "--bucket=test", "--subnet=test", "--service-account=trainer@example.test",
                "--image-uri=gs://test/image.tar.gz", "--image-sha256=" + "a" * 64,
                "--image-id=sha256:" + "b" * 64, "--max-minutes=30"]

    def resume(self):
        return ["--resume-bundle-uri=gs://test/resume.zip", "--resume-bundle-sha256=" + "c" * 64]

    def test_requires_resume_rejects_conflicting_modes_and_budget_before_cloud(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "benchmark.json"
            path.write_text('{"max-seconds":900}')
            option = ["--collector-benchmark-config", str(path)]
            for extra in (option, option + self.resume() + ["--pipeline-config", str(path)],
                          option + self.resume() + ["--max-minutes=15"]):
                with self.subTest(extra=extra), patch.object(runner, "preflight") as preflight, \
                        patch("sys.argv", self.argv() + extra), contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit):
                    runner.main()
                preflight.assert_not_called()

    def test_launch_forwards_only_benchmark_controls_with_existing_delete_guarantee(self):
        captured = {}

        def create(command, **kwargs):
            metadata = next(value for value in command if value.startswith("--metadata-from-file="))
            config_path = next(value.split("=", 1)[1] for value in metadata.split(",") if value.startswith("ghq-config="))
            captured.update(json.loads(Path(config_path).read_text()))
            self.assertIn("--max-run-duration=30m", command)
            self.assertIn("--instance-termination-action=DELETE", command)
            self.assertTrue(kwargs["name"].startswith("ghq-collect-bench-"))

        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "benchmark.json"
            path.write_text('{"max-seconds":900,"workers":[16,32],"profile-inference":true}')
            with patch.object(runner, "preflight", return_value={"vm_image": "approved"}), \
                    patch.object(runner, "create_vm", side_effect=create), \
                    patch.object(runner, "wait_for_completion", return_value={"exit_code": 0, "final_sync_succeeded": True}), \
                    patch.object(runner, "delete_vm", return_value=True) as cleanup, \
                    patch("sys.argv", self.argv() + self.resume() + ["--collector-benchmark-config", str(path), "--launch"]), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 0)
        self.assertEqual(captured["job_mode"], "collector-benchmark")
        self.assertEqual(captured["pipeline_args"], [])
        self.assertEqual(captured["collector_benchmark_max_seconds"], 900)
        self.assertEqual(captured["collector_benchmark_args"], ["--max-seconds", "900", "--workers", "16", "32", "--profile-inference"])
        cleanup.assert_called_once()


class CollectorWorkerTests(unittest.TestCase):
    def config(self):
        return {"run_id": "benchmark-test", "job_mode": "collector-benchmark", "image_id": "sha256:test",
                "pipeline_args": [], "collector_benchmark_args": ["--max-seconds", "900", "--workers", "16", "32"],
                "collector_benchmark_max_seconds": 900}

    def test_workload_is_collector_only_checkpoint_and_sources_readonly(self):
        config = {**self.config(), "source_directory": "/host/source"}
        with patch.object(worker, "training_command", side_effect=AssertionError("must not train")):
            command, log, timeout = worker.workload(config, Path("/host/output"), resume=Path("/host/resume"))
        self.assertEqual(log, "collector-benchmark.log")
        self.assertEqual(timeout, 1020)
        self.assertIn("type=bind,source=/host/resume,target=/resume,readonly", command)
        self.assertIn("type=bind,source=/host/source,target=/app,readonly", command)
        self.assertIn("--network=none", command)
        self.assertEqual(command[command.index("-m") + 1:],
                         ["ghqnet.benchmark_collector", "/resume/checkpoint.pt", "--device", "cuda", "--output",
                          "/output/collector-benchmark.json", *config["collector_benchmark_args"]])
        self.assertNotIn("--resume", command)
        self.assertNotIn("/output/run", command)

    def test_worker_rejects_missing_checkpoint_conflicting_args_or_budget(self):
        for updates, resume in (({}, None), ({"pipeline_args": ["--epochs", "2"]}, Path("/resume")),
                                ({"collector_benchmark_args": ["--max-seconds", "900", "--output=elsewhere"]}, Path("/resume")),
                                ({"collector_benchmark_max_seconds": 5}, Path("/resume")),
                                ({"collector_benchmark_max_seconds": float("nan")}, Path("/resume"))):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                worker.workload({**self.config(), **updates}, Path("/output"), resume=resume)

    def test_external_watchdog_stops_only_benchmark_container(self):
        process = Mock()
        process.wait.side_effect = [subprocess.TimeoutExpired("benchmark", 1020), 137]
        with patch.object(worker.subprocess, "run") as stop, self.assertRaises(subprocess.TimeoutExpired):
            worker.wait_for_workload(process, self.config(), timeout=1020)
        stop.assert_called_once_with(["docker", "stop", "--time", "20", "ghq-collector-benchmark"],
                                     timeout=30, check=False)
        process.kill.assert_not_called()

    def test_worker_main_skips_training_and_auxiliary_microbenchmark(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            config = {**self.config(), "output_uri": "gs://test/out", "image_uri": "gs://test/image",
                      "image_sha256": "a" * 64, "resume_bundle_uri": "gs://test/resume",
                      "resume_bundle_sha256": "b" * 64}
            commands = []
            gpu_process, job_process = Mock(), Mock()
            job_process.wait.return_value = 0

            def run(command, **kwargs):
                commands.append(command)
                if command[:3] == ["gcloud", "storage", "cp"] and command[2 + 1].startswith("gs://"):
                    Path(command[4]).write_bytes(b"verified separately")
                if command[:2] == ["docker", "load"]:
                    return SimpleNamespace(stdout="Loaded image: approved:latest", stderr="")
                if command[0] == "docker" and "-c" in command:
                    return SimpleNamespace(stdout='{"gpu":"test","cuda":"12","torch":"2"}')
                return SimpleNamespace(stdout="gpu", returncode=0)

            def extract(archive, destination):
                destination.mkdir()
                (destination / "checkpoint.pt").write_bytes(b"untouched")

            def path(value):
                return root / Path(value).name if str(value).startswith(("/var/lib/", "/opt/ghq-job/")) else Path(value)

            with patch.object(worker, "metadata", return_value=json.dumps(config)), \
                    patch.object(worker, "Path", side_effect=path), patch.object(worker.signal, "signal"), \
                    patch.object(worker.platform, "platform", return_value="test-platform"), \
                    patch.object(worker.threading, "Thread"), patch.object(worker.subprocess, "run", side_effect=run), \
                    patch.object(worker.subprocess, "Popen", side_effect=[gpu_process, job_process]) as popen, \
                    patch.object(worker, "verify_sha256") as verify, \
                    patch.object(worker, "resolve_loaded_image", return_value={"resolved_id": "sha256:verified"}), \
                    patch.object(worker, "extract_resume_bundle", side_effect=extract), \
                    patch.object(worker, "training_command", side_effect=AssertionError("must not train")), \
                    patch.object(worker, "benchmark_command", side_effect=AssertionError("must not run auxiliary benchmark")):
                exit_code = worker.main()
                self.assertEqual(exit_code, 0, (root / "ghq-output/worker-error.txt").read_text()
                                 if (root / "ghq-output/worker-error.txt").exists() else "No worker error")
            self.assertEqual(verify.call_count, 2)
            self.assertIn("ghqnet.benchmark_collector", popen.call_args_list[-1].args[0])
            job_process.wait.assert_called_once_with(timeout=1020)
            self.assertEqual((root / "resume/checkpoint.pt").read_bytes(), b"untouched")
            output = root / "ghq-output"
            self.assertTrue((output / "collector-benchmark.log").exists())
            self.assertFalse((output / "training.log").exists())
            self.assertFalse((output / "benchmark.log").exists())
            complete = json.loads((output / "completion.json").read_text())
            self.assertEqual(complete["exit_code"], 0)
            self.assertTrue(complete["final_sync_succeeded"])
            self.assertTrue(any(command[:3] == ["gcloud", "storage", "rsync"] for command in commands))


if __name__ == "__main__":
    unittest.main()
