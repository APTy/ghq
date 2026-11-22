"""Offline, mocked launch/worker checks; no Torch, games, or cloud access."""
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
    spec = importlib.util.spec_from_file_location("ablation_cloud_" + name, Path(__file__).with_name(name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


runner, worker = load("run"), load("worker")


class ReplayAblationConfigurationTests(unittest.TestCase):
    def read(self, config):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ablation.json"
            path.write_text(json.dumps(config))
            return runner.load_replay_ablation_config(path)

    def test_exact_numeric_options_become_literal_argv(self):
        config = {"learning-rates": [.001, .0003], "epochs": 1, "batch-size": 128,
                  "threads": 2, "seed": 20261121, "max-seconds": 1200}
        arguments, returned, digest = self.read(config)
        self.assertEqual(returned, config)
        self.assertEqual(len(digest), 64)
        self.assertEqual(arguments, ["--learning-rates", "0.001", "0.0003", "--epochs", "1",
                                    "--batch-size", "128", "--threads", "2", "--seed", "20261121",
                                    "--max-seconds", "1200"])
        self.assertEqual(self.read({"max-seconds": 1200})[0], ["--max-seconds", "1200"])

    def test_rejects_unsafe_flags_nonnumeric_values_duplicates_and_ranges(self):
        invalid = [{"output": "/resume/checkpoint.pt"}, {"device": "cpu"}, {"resume": "wrong"},
                   {"games": 2}, {"simulations": 8}, {"validate-only": True}, {"help": True},
                   {"learning-rates": []}, {"learning-rates": [.001]}, {"learning-rates": [.001] * 2},
                   {"learning-rates": [.1, .2, .3, .4, .5]}, {"learning-rates": [.1, float("nan")]},
                   {"learning-rates": [.1, float("inf")]}, {"learning-rates": [.1, 0]},
                   {"learning-rates": [.1, -1]}, {"learning-rates": [.1, 2]},
                   {"learning-rates": [.1, 10 ** 400]}, {"learning-rates": [.1, True]},
                   {"learning-rates": [.1, "0.001; touch /tmp/unwanted"]},
                   {"epochs": 0}, {"epochs": 5}, {"epochs": 1.0}, {"epochs": True},
                   {"batch-size": 513}, {"threads": 65}, {"seed": -1}, {"seed": 2 ** 32},
                   {"max-seconds": 0}, {"max-seconds": float("nan")}, {"max-seconds": float("inf")},
                   {"max-seconds": 13801}, {"max-seconds": 10 ** 400}, {"max-seconds": "1200"}]
        for changes in invalid:
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.read({"max-seconds": 1200, **changes})
        with self.assertRaisesRegex(ValueError, "explicit max-seconds"):
            self.read({"epochs": 1})


class ReplayAblationLaunchTests(unittest.TestCase):
    def argv(self):
        return ["run.py", "--bucket=test", "--subnet=test", "--service-account=trainer@example.test",
                "--image-uri=gs://test/image.tar.gz", "--image-sha256=" + "a" * 64,
                "--image-id=sha256:" + "b" * 64, "--max-minutes=30"]

    def resume(self):
        return ["--resume-bundle-uri=gs://test/resume.zip", "--resume-bundle-sha256=" + "c" * 64]

    def test_missing_resume_conflicting_modes_and_budget_fail_before_cloud(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ablation.json"
            path.write_text('{"max-seconds":1200}')
            option = ["--replay-ablation-config", str(path)]
            extras = [option, option + self.resume() + ["--max-minutes=29"],
                      option + self.resume() + ["--pipeline-config", str(path)],
                      option + self.resume() + ["--collector-benchmark-config", str(path)]]
            for extra in extras:
                with self.subTest(extra=extra), patch.object(runner, "preflight") as preflight, \
                        patch("sys.argv", self.argv() + extra), contextlib.redirect_stderr(io.StringIO()), \
                        self.assertRaises(SystemExit):
                    runner.main()
                preflight.assert_not_called()

    def test_launch_uses_separate_mode_and_preserves_delete_bound(self):
        captured = {}

        def create(command, **kwargs):
            metadata = next(value for value in command if value.startswith("--metadata-from-file="))
            path = next(value.split("=", 1)[1] for value in metadata.split(",") if value.startswith("ghq-config="))
            captured.update(json.loads(Path(path).read_text()))
            self.assertIn("--max-run-duration=30m", command)
            self.assertIn("--instance-termination-action=DELETE", command)
            self.assertIn("--boot-disk-auto-delete", command)
            self.assertTrue(kwargs["name"].startswith("ghq-replay-ablate-"))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "ablation.json"
            path.write_text('{"max-seconds":1200,"learning-rates":[0.001,0.0003],"threads":2}')
            with patch.object(runner, "preflight", return_value={"vm_image": "approved"}), \
                    patch.object(runner, "create_vm", side_effect=create), \
                    patch.object(runner, "wait_for_completion", return_value={"exit_code": 0, "final_sync_succeeded": True}), \
                    patch.object(runner, "delete_vm", return_value=True) as cleanup, \
                    patch("sys.argv", self.argv() + self.resume() + ["--replay-ablation-config", str(path), "--launch"]), \
                    contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(runner.main(), 0)
        self.assertEqual(captured["job_mode"], "replay-ablation")
        self.assertEqual(captured["pipeline_args"], [])
        self.assertNotIn("collector_benchmark_args", captured)
        self.assertEqual(captured["replay_ablation_max_seconds"], 1200)
        self.assertEqual(captured["replay_ablation_args"], ["--max-seconds", "1200", "--learning-rates", "0.001", "0.0003", "--threads", "2"])
        cleanup.assert_called_once()


class ReplayAblationWorkerTests(unittest.TestCase):
    def config(self):
        return {"run_id": "ablation-test", "job_mode": "replay-ablation", "image_id": "sha256:test",
                "pipeline_args": [], "replay_ablation_args": ["--learning-rates", "0.001", "0.0003",
                "--epochs", "1", "--batch-size", "128", "--threads", "2", "--seed", "20261121",
                "--max-seconds", "1200"], "replay_ablation_max_seconds": 1200}

    def test_command_is_ablation_only_with_readonly_inputs(self):
        config = {**self.config(), "source_directory": "/host/source"}
        with patch.object(worker, "training_command", side_effect=AssertionError("must not run pipeline")), \
                patch.object(worker, "collector_benchmark_command", side_effect=AssertionError("must not collect")):
            command, log, timeout = worker.workload(config, Path("/host/output"), resume=Path("/host/resume"))
        self.assertEqual(log, "replay-ablation.log")
        self.assertEqual(timeout, 1320)
        self.assertIn("type=bind,source=/host/resume,target=/resume,readonly", command)
        self.assertIn("type=bind,source=/host/source,target=/app,readonly", command)
        self.assertIn("--network=none", command)
        self.assertEqual(command[command.index("-m") + 1:], ["ghqnet.ablate_replay", "/resume/checkpoint.pt",
            "--output", "/output/replay-ablation", "--device", "cuda", *config["replay_ablation_args"]])
        self.assertNotIn("--resume", command)
        self.assertNotIn("/output/run", command)

    def test_worker_independently_rejects_override_abbreviation_duplicate_and_bad_budget(self):
        invalid_args = [["--output", "/tmp/other"], ["--output=/tmp/other"], ["--out", "/tmp/other"],
                        ["--device", "cpu"], ["--validate-only"], ["--epochs", "5"], ["--batch-size", "513"],
                        ["--threads", "0"], ["--seed", "4294967296"], ["--learning-rates", "0.1"],
                        ["--learning-rates", "0.1", "0.1"], ["--learning-rates", "0.1", "inf"],
                        ["--learning-rates", "0.1", "$(touch /tmp/unwanted)"],
                        ["--max-seconds", "1200", "--max-seconds", "1200"], ["--max-seconds", "nan"],
                        ["--epochs", "1.0"], ["--epochs", "1", "extra"]]
        for arguments in invalid_args:
            if "--max-seconds" not in arguments:
                arguments = ["--max-seconds", "1200", *arguments]
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                worker.workload({**self.config(), "replay_ablation_args": arguments}, Path("/output"), resume=Path("/resume"))
        for changes, resume in (({}, None), ({"pipeline_args": ["--epochs", "1"]}, Path("/resume")),
                                ({"collector_benchmark_args": ["--games", "1"]}, Path("/resume")),
                                ({"replay_ablation_args": ["--epochs", "1"]}, Path("/resume")),
                                ({"replay_ablation_max_seconds": 5}, Path("/resume")),
                                ({"replay_ablation_max_seconds": float("nan")}, Path("/resume"))):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                worker.workload({**self.config(), **changes}, Path("/output"), resume=resume)

    def test_watchdog_targets_ablation_and_escalates_if_docker_client_hangs(self):
        process = Mock()
        process.wait.side_effect = [subprocess.TimeoutExpired("ablation", 1320),
                                    subprocess.TimeoutExpired("ablation", 10), 137]
        with patch.object(worker.subprocess, "run") as stop, self.assertRaises(subprocess.TimeoutExpired):
            worker.wait_for_workload(process, self.config(), timeout=1320)
        stop.assert_called_once_with(["docker", "stop", "--time", "20", "ghq-replay-ablation"], timeout=30, check=False)
        process.kill.assert_called_once()
        self.assertEqual(worker.workload_container_name({}), "ghq-training")
        self.assertEqual(worker.workload_container_name({"job_mode": "collector-benchmark"}), "ghq-collector-benchmark")

    def test_main_skips_pipeline_collection_and_microbenchmark_then_syncs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = {**self.config(), "output_uri": "gs://test/out", "image_uri": "gs://test/image",
                      "image_sha256": "a" * 64, "resume_bundle_uri": "gs://test/resume",
                      "resume_bundle_sha256": "b" * 64}
            commands, signal_handlers = [], {}
            gpu_process, job_process = Mock(), Mock()
            job_process.wait.return_value = 0

            def run(command, **kwargs):
                commands.append(command)
                if command[:3] == ["gcloud", "storage", "cp"] and command[3].startswith("gs://"):
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
                    patch.object(worker, "Path", side_effect=path), \
                    patch.object(worker.signal, "signal", side_effect=lambda signum, handler: signal_handlers.update({signum: handler})), \
                    patch.object(worker.platform, "platform", return_value="test-platform"), \
                    patch.object(worker.threading, "Thread"), patch.object(worker.subprocess, "run", side_effect=run), \
                    patch.object(worker.subprocess, "Popen", side_effect=[gpu_process, job_process]) as popen, \
                    patch.object(worker, "verify_sha256") as verify, \
                    patch.object(worker, "resolve_loaded_image", return_value={"resolved_id": "sha256:verified"}), \
                    patch.object(worker, "extract_resume_bundle", side_effect=extract), \
                    patch.object(worker, "training_command", side_effect=AssertionError("must not run pipeline")), \
                    patch.object(worker, "collector_benchmark_command", side_effect=AssertionError("must not collect")), \
                    patch.object(worker, "benchmark_command", side_effect=AssertionError("must not run microbenchmark")):
                self.assertEqual(worker.main(), 0)
                with self.assertRaises(SystemExit):
                    signal_handlers[worker.signal.SIGTERM](worker.signal.SIGTERM, None)
            self.assertEqual(verify.call_count, 2)
            self.assertIn("ghqnet.ablate_replay", popen.call_args_list[-1].args[0])
            job_process.wait.assert_called_once_with(timeout=1320)
            self.assertEqual((root / "resume/checkpoint.pt").read_bytes(), b"untouched")
            output = root / "ghq-output"
            self.assertTrue((output / "replay-ablation.log").exists())
            self.assertFalse((output / "training.log").exists())
            self.assertFalse((output / "benchmark.log").exists())
            complete = json.loads((output / "completion.json").read_text())
            self.assertEqual(complete["exit_code"], 0)
            self.assertTrue(complete["final_sync_succeeded"])
            self.assertIn(["docker", "stop", "--time", "20", "ghq-replay-ablation"], commands)
            self.assertTrue(any(command[:3] == ["gcloud", "storage", "rsync"] for command in commands))

    def test_shutdown_includes_ablation_container(self):
        with patch.object(worker, "metadata", return_value='{"output_uri":"gs://test/out"}'), \
                patch.object(worker.subprocess, "run") as run:
            worker.shutdown()
        self.assertIn("ghq-replay-ablation", run.call_args_list[0].args[0])
        self.assertEqual(run.call_args_list[-1].args[0][:3], ["gcloud", "storage", "rsync"])


if __name__ == "__main__":
    unittest.main()
