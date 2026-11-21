"""Offline tests for the launch boundary and cleanup; no Google calls."""
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("ghq_cloud_run", Path(__file__).with_name("run.py"))
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)
worker_spec = importlib.util.spec_from_file_location("ghq_cloud_worker", Path(__file__).with_name("worker.py"))
worker = importlib.util.module_from_spec(worker_spec)
worker_spec.loader.exec_module(worker)


class LaunchBoundaryTests(unittest.TestCase):
    def argv(self):
        return ["run.py", "--bucket=test", "--subnet=test", "--service-account=trainer@example.test",
                "--image-uri=gs://test/image.tar.gz", "--image-sha256=" + "a" * 64,
                "--image-id=sha256:" + "b" * 64]

    def plan(self):
        return {"vm_image": "approved-image"}

    def test_default_is_read_only(self):
        with patch.object(runner, "preflight", return_value=self.plan()), \
                patch.object(runner.subprocess, "run") as command, \
                patch("sys.argv", self.argv()), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 0)
        command.assert_not_called()

    def test_launch_deletes_after_confirmed_upload(self):
        complete = {"exit_code": 0, "final_sync_succeeded": True}
        responses = [SimpleNamespace(returncode=0),
                     SimpleNamespace(returncode=0, stdout=json.dumps(complete)),
                     SimpleNamespace(returncode=0)]
        with patch.object(runner, "preflight", return_value=self.plan()), \
                patch.object(runner.subprocess, "run", side_effect=responses) as command, \
                patch("sys.argv", self.argv() + ["--launch"]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 0)
        create, read, delete = [call.args[0] for call in command.call_args_list]
        self.assertEqual(create[1:4], ["compute", "instances", "create"])
        self.assertIn("--no-address", create)
        self.assertIn("--scopes=storage-rw", create)
        self.assertIn("--max-run-duration=120m", create)
        self.assertIn("--instance-termination-action=DELETE", create)
        self.assertIn("--boot-disk-auto-delete", create)
        self.assertEqual(delete[1:4], ["compute", "instances", "delete"])
        self.assertEqual(create[4], delete[4])

    def test_interrupted_wait_still_deletes_only_created_vm(self):
        with patch.object(runner, "preflight", return_value=self.plan()), \
                patch.object(runner.subprocess, "run", side_effect=[None, KeyboardInterrupt(), None]) as command, \
                patch("sys.argv", self.argv() + ["--launch"]), \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(KeyboardInterrupt):
                runner.main()
        calls = [call.args[0] for call in command.call_args_list]
        self.assertEqual(calls[-1][1:4], ["compute", "instances", "delete"])
        self.assertEqual(calls[0][4], calls[-1][4])


    def test_ambiguous_create_cleans_only_confirmed_owned_vm(self):
        instance = {"metadata": {"items": [{"key": "ghq-config", "value": '{"run_id":"this-run"}'}]},
                    "labels": {"managed-by": "ghq-training"}}
        failure = subprocess.TimeoutExpired(["create"], 300)
        with patch.object(runner, "gcloud", return_value=instance), \
                patch.object(runner.subprocess, "run", side_effect=[failure, None]) as command:
            with self.assertRaises(subprocess.TimeoutExpired):
                runner.create_vm(["create"], name="this-run", project="play-ghq", zone="us-central1-a")
        self.assertEqual(command.call_args_list[-1].args[0][1:5],
                         ["compute", "instances", "delete", "this-run"])

    def test_ambiguous_create_does_not_delete_unconfirmed_vm(self):
        with patch.object(runner, "gcloud", return_value={"name": "this-run"}), \
                patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired(["create"], 300)) as command, \
                contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(subprocess.TimeoutExpired):
                runner.create_vm(["create"], name="this-run", project="play-ghq", zone="us-central1-a")
        self.assertEqual(command.call_count, 1)

    def test_worker_uses_empty_pipeline_subdirectory(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "job-config.json").write_text("{}")
            (output / "training.log").write_text("startup complete")
            command = worker.training_command({"image_id": "sha256:test", "pipeline_args": []}, output)
            self.assertEqual(command[command.index("--output") + 1], "/output/run")
            self.assertIn(f"type=bind,source={output},target=/output", command)
            self.assertFalse((output / "run").exists())


if __name__ == "__main__":
    unittest.main()
