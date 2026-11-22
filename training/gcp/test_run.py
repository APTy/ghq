"""Offline tests for the launch boundary and cleanup; no Google calls."""
import contextlib
import importlib.util
import io
import json
import hashlib
from pathlib import Path
import stat
import tarfile
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import zipfile

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
                patch.object(runner, "gcloud", return_value=None), \
                patch.object(runner.subprocess, "run", side_effect=responses) as command, \
                patch("sys.argv", self.argv() + ["--launch"]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 0)
        create, read, delete = [call.args[0] for call in command.call_args_list]
        self.assertEqual(create[1:4], ["compute", "instances", "create"])
        self.assertIn("--no-address", create)
        self.assertIn("--scopes=storage-rw", create)
        self.assertIn("--max-run-duration=60m", create)
        self.assertIn("--machine-type=g2-standard-4", create)
        self.assertFalse(any(flag.startswith("--accelerator=") for flag in create))
        self.assertIn("--instance-termination-action=DELETE", create)
        self.assertIn("--boot-disk-auto-delete", create)
        self.assertEqual(delete[1:4], ["compute", "instances", "delete"])
        self.assertEqual(create[4], delete[4])

    def test_explicit_t4_keeps_private_bounded_runtime(self):
        complete = {"exit_code": 0, "final_sync_succeeded": True}
        with patch.object(runner, "preflight", return_value=self.plan()) as preflight, \
                patch.object(runner, "gcloud", return_value=None), \
                patch.object(runner.subprocess, "run", side_effect=[SimpleNamespace(returncode=0),
                    SimpleNamespace(returncode=0, stdout=json.dumps(complete)), SimpleNamespace(returncode=0)]) as command, \
                patch("sys.argv", self.argv() + ["--launch", "--gpu=t4", "--max-minutes=55"]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 0)
        self.assertEqual(preflight.call_args.args[0].gpu, "t4")
        create = command.call_args_list[0].args[0]
        for flag in ("--machine-type=n1-standard-4", "--accelerator=type=nvidia-tesla-t4,count=1",
                     "--no-address", "--maintenance-policy=TERMINATE", "--max-run-duration=55m",
                     "--instance-termination-action=DELETE", "--boot-disk-auto-delete", "--scopes=storage-rw"):
            self.assertIn(flag, create)

    def test_interrupted_wait_still_deletes_only_created_vm(self):
        with patch.object(runner, "preflight", return_value=self.plan()), \
                patch.object(runner, "gcloud", return_value=None), \
                patch.object(runner.subprocess, "run", side_effect=[None, KeyboardInterrupt(), SimpleNamespace(returncode=0)]) as command, \
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
        with patch.object(runner, "gcloud", side_effect=[instance, None]), \
                patch.object(runner.subprocess, "run", side_effect=[failure, SimpleNamespace(returncode=0)]) as command, \
                contextlib.redirect_stdout(io.StringIO()):
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

    def test_resume_flags_forwarded_with_training_controls(self):
        captured = {}

        def create(command, **kwargs):
            metadata = next(value for value in command if value.startswith("--metadata-from-file="))
            config = next(value.split("=", 1)[1] for value in metadata.split(",") if value.startswith("ghq-config="))
            captured.update(json.loads(Path(config).read_text()))

        complete = {"exit_code": 0, "final_sync_succeeded": True}
        with patch.object(runner, "preflight", return_value=self.plan()), \
                patch.object(runner, "create_vm", side_effect=create), \
                patch.object(runner, "gcloud", return_value=None), \
                patch.object(runner.subprocess, "run", side_effect=[SimpleNamespace(returncode=0, stdout=json.dumps(complete)),
                    SimpleNamespace(returncode=0)]), \
                patch("sys.argv", self.argv() + ["--launch", "--resume-bundle-uri=gs://test/resume.zip",
                    "--resume-bundle-sha256=" + "c" * 64, "--teacher-games=0", "--teacher-ms=250",
                    "--seed=123", "--threads=3", "--batch-size=32", "--runtime-bundle-uri=gs://test/runtime.zip",
                    "--runtime-bundle-sha256=" + "d" * 64]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 0)
        self.assertEqual(captured["resume_bundle_uri"], "gs://test/resume.zip")
        self.assertEqual(captured["resume_bundle_sha256"], "c" * 64)
        self.assertEqual(captured["runtime_bundle_uri"], "gs://test/runtime.zip")
        self.assertEqual(captured["runtime_bundle_sha256"], "d" * 64)
        for name, value in (("teacher-games", "0"), ("teacher-ms", "250.0"),
                            ("seed", "123"), ("threads", "3"), ("batch-size", "32")):
            self.assertEqual(captured["pipeline_args"][captured["pipeline_args"].index("--" + name) + 1], value)

    def test_resume_flags_must_be_paired(self):
        with patch.object(runner, "preflight") as preflight, \
                patch("sys.argv", self.argv() + ["--resume-bundle-uri=gs://test/resume.zip"]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main()
        preflight.assert_not_called()

    def test_runtime_flags_must_be_paired(self):
        with patch.object(runner, "preflight") as preflight, \
                patch("sys.argv", self.argv() + ["--runtime-bundle-uri=gs://test/runtime.zip"]), \
                contextlib.redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
            runner.main()
        preflight.assert_not_called()

    def test_completed_training_with_unconfirmed_cleanup_is_failure(self):
        complete = {"exit_code": 0, "final_sync_succeeded": True}
        with patch.object(runner, "preflight", return_value=self.plan()), \
                patch.object(runner, "create_vm"), \
                patch.object(runner, "wait_for_completion", return_value=complete), \
                patch.object(runner, "delete_vm", return_value=False) as delete, \
                patch("sys.argv", self.argv() + ["--launch"]), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(runner.main(), 1)
        delete.assert_called_once()


class MonitoringTests(unittest.TestCase):
    def test_optional_resource_only_accepts_confirmed_absence(self):
        missing = ["HTTPError 404: requested instance does not exist", "NOT_FOUND: instance absent",
                   "The resource 'projects/test/zones/test/instances/run' was not found"]
        uncertain = ["HTTPError 503: temporarily unavailable", "PERMISSION_DENIED: not authorized",
                     "Connection reset by peer", "gcloud component not found"]
        for message in missing + uncertain:
            result = SimpleNamespace(returncode=1, stderr=message)
            with self.subTest(message=message), patch.object(runner.subprocess, "run", return_value=result):
                if message in missing:
                    self.assertIsNone(runner.gcloud("compute", "instances", "describe", "run",
                                                   project="test", optional=True))
                else:
                    with self.assertRaises(RuntimeError):
                        runner.gcloud("compute", "instances", "describe", "run", project="test", optional=True)

    def test_empty_success_response_is_not_confirmed_absence(self):
        for output in ("", "null", "[]"):
            with self.subTest(output=output), patch.object(runner.subprocess, "run",
                    return_value=SimpleNamespace(returncode=0, stdout=output)):
                with self.assertRaisesRegex(RuntimeError, "no valid object"):
                    runner.gcloud("compute", "instances", "describe", "run", project="test", optional=True)

    def test_timeout_and_status_error_recover_without_ending_monitor(self):
        complete = {"exit_code": 0, "final_sync_succeeded": True}
        for error in (RuntimeError("HTTPError 503"), RuntimeError("PERMISSION_DENIED"),
                      subprocess.TimeoutExpired(["describe"], 20)):
            with self.subTest(error=error), \
                    patch.object(runner.subprocess, "run", side_effect=[subprocess.TimeoutExpired(["cat"], 20),
                        SimpleNamespace(returncode=0, stdout=json.dumps(complete))]) as read, \
                    patch.object(runner, "gcloud", side_effect=error) as describe, \
                    patch.object(runner.time, "monotonic", return_value=0), \
                    patch.object(runner.time, "sleep") as sleep, \
                    contextlib.redirect_stdout(io.StringIO()):
                result = runner.wait_for_completion("run", "gs://test/run", project="test", zone="zone", deadline=20)
            self.assertEqual(result, complete)
            self.assertEqual(read.call_count, 2)
            describe.assert_called_once()
            sleep.assert_called_once_with(15)

    def test_persistent_read_errors_stop_at_existing_deadline(self):
        now = [0]

        def advance(seconds):
            now[0] += seconds

        with patch.object(runner.subprocess, "run",
                return_value=SimpleNamespace(returncode=1, stderr="HTTPError 503")) as read, \
                patch.object(runner, "gcloud", side_effect=RuntimeError("Network unavailable")) as describe, \
                patch.object(runner.time, "monotonic", side_effect=lambda: now[0]), \
                patch.object(runner.time, "sleep", side_effect=advance), \
                contextlib.redirect_stdout(io.StringIO()) as output:
            result = runner.wait_for_completion("run", "gs://test/run", project="test", zone="zone", deadline=20)
        self.assertIsNone(result)
        self.assertEqual(now[0], 20)
        self.assertEqual(read.call_count, 2)
        self.assertEqual([call.kwargs["timeout"] for call in read.call_args_list], [20, 5])
        self.assertEqual([call.kwargs["timeout"] for call in describe.call_args_list], [20, 5])
        self.assertIn("deadline reached", output.getvalue())

    def test_confirmed_vm_absence_ends_monitor(self):
        with patch.object(runner.subprocess, "run",
                return_value=SimpleNamespace(returncode=1, stderr="HTTPError 404")), \
                patch.object(runner, "gcloud", return_value=None), \
                patch.object(runner.time, "monotonic", return_value=0), \
                patch.object(runner.time, "sleep") as sleep, \
                contextlib.redirect_stdout(io.StringIO()) as output:
            result = runner.wait_for_completion("run", "gs://test/run", project="test", zone="zone", deadline=20)
        self.assertIsNone(result)
        sleep.assert_not_called()
        self.assertIn("absence confirmed", output.getvalue())


class CleanupTests(unittest.TestCase):
    def test_delete_retries_until_absence_is_confirmed(self):
        with patch.object(runner.subprocess, "run", side_effect=[
                SimpleNamespace(returncode=1, stderr="HTTPError 503"), SimpleNamespace(returncode=0)]) as delete, \
                patch.object(runner, "gcloud", side_effect=[{"status": "RUNNING"}, None]) as describe, \
                patch.object(runner.time, "sleep") as sleep, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertTrue(runner.delete_vm("this-run", project="test", zone="zone"))
        self.assertEqual(delete.call_count, 2)
        self.assertEqual([call.kwargs["timeout"] for call in delete.call_args_list], [60, 60])
        self.assertEqual([call.kwargs["timeout"] for call in describe.call_args_list], [15, 15])
        for call in delete.call_args_list:
            self.assertEqual(call.args[0], ["gcloud", "compute", "instances", "delete", "this-run",
                                            "--project=test", "--zone=zone", "--quiet"])
        sleep.assert_called_once_with(2)
        self.assertIn("Confirmed VM deleted", output.getvalue())

    def test_delete_timeout_can_still_have_confirmed_success(self):
        with patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired(["delete"], 60)), \
                patch.object(runner, "gcloud", return_value=None), \
                contextlib.redirect_stdout(io.StringIO()):
            self.assertTrue(runner.delete_vm("this-run", project="test", zone="zone"))

    def test_unconfirmed_cleanup_is_bounded_and_actionable(self):
        with patch.object(runner.subprocess, "run", side_effect=subprocess.TimeoutExpired(["delete"], 60)) as delete, \
                patch.object(runner, "gcloud", side_effect=RuntimeError("PERMISSION_DENIED")) as describe, \
                patch.object(runner.time, "sleep") as sleep, contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertFalse(runner.delete_vm("this-run", project="test", zone="zone"))
        self.assertEqual(delete.call_count, 3)
        self.assertEqual(describe.call_count, 3)
        self.assertEqual(sleep.call_count, 2)
        self.assertIn("CLEANUP NOT CONFIRMED for this-run", output.getvalue())
        self.assertIn("PERMISSION_DENIED", output.getvalue())
        self.assertIn("gcloud compute instances delete this-run --project=test --zone=zone --quiet", output.getvalue())
        self.assertNotIn("Confirmed VM deleted", output.getvalue())


class QuotaTests(unittest.TestCase):
    def quota(self, name, remaining=1):
        return {"metric": name, "limit": remaining, "usage": 0}

    def test_absent_global_metric_is_unknown_not_zero(self):
        status = runner.check_gpu_quotas([self.quota("NVIDIA_L4_GPUS")], [])
        self.assertIn("unreported", status)

    def test_reported_global_limit_is_enforced(self):
        with self.assertRaisesRegex(RuntimeError, "GPUS_ALL_REGIONS"):
            runner.check_gpu_quotas([self.quota("NVIDIA_L4_GPUS")], [self.quota("GPUS_ALL_REGIONS", 0)])

    def test_reported_preemptible_runtime_quota_is_enforced(self):
        with self.assertRaisesRegex(RuntimeError, "PREEMPTIBLE_NVIDIA_L4_GPUS"):
            runner.check_gpu_quotas([self.quota("NVIDIA_L4_GPUS"), self.quota("PREEMPTIBLE_NVIDIA_L4_GPUS", 0)], [])

    def test_family_metric_and_missing_regional_quota(self):
        runner.check_gpu_quotas([self.quota("GPU_FAMILY:NVIDIA_L4")], [])
        with self.assertRaisesRegex(RuntimeError, "regional L4"):
            runner.check_gpu_quotas([], [])

    def test_t4_checks_t4_instead_of_l4_and_four_cpus(self):
        regional = [self.quota("NVIDIA_T4_GPUS"), self.quota("CPUS", 4),
                    self.quota("PREEMPTIBLE_NVIDIA_T4_GPUS"), self.quota("PREEMPTIBLE_CPUS", 4)]
        runner.check_gpu_quotas(regional, [], "t4")
        with self.assertRaisesRegex(RuntimeError, "regional CPUS"):
            runner.check_gpu_quotas([self.quota("NVIDIA_T4_GPUS"), self.quota("CPUS", 3)], [], "t4")
        with self.assertRaisesRegex(RuntimeError, "regional T4"):
            runner.check_gpu_quotas([self.quota("NVIDIA_L4_GPUS"), self.quota("CPUS", 4)], [], "t4")
        with self.assertRaisesRegex(RuntimeError, "PREEMPTIBLE_NVIDIA_T4_GPUS"):
            runner.check_gpu_quotas([self.quota("NVIDIA_T4_GPUS"), self.quota("CPUS", 4),
                                   self.quota("PREEMPTIBLE_NVIDIA_T4_GPUS", 0)], [], "t4")

    def test_t4_checks_reported_global_and_preemptible_cpu_limits(self):
        regional = [self.quota("NVIDIA_T4_GPUS"), self.quota("CPUS", 4)]
        with self.assertRaisesRegex(RuntimeError, "CPUS_ALL_REGIONS"):
            runner.check_gpu_quotas(regional, [self.quota("CPUS_ALL_REGIONS", 3)], "t4")
        with self.assertRaisesRegex(RuntimeError, "PREEMPTIBLE_CPUS"):
            runner.check_gpu_quotas(regional + [self.quota("PREEMPTIBLE_CPUS", 3)], [], "t4")


class WorkerTests(unittest.TestCase):
    def runtime_bundle(self, folder, *, bad_hash=False, extra=False):
        data = {name: name.encode() for name in worker.RUNTIME_FILES - {"manifest.json"}}
        manifest = {"docker_version": "29.10.0", "nvidia_container_toolkit_version": "1.20.2-1",
                    "files": [{"name": name, "bytes": len(value), "sha256": hashlib.sha256(value).hexdigest()}
                              for name, value in data.items()]}
        if bad_hash:
            manifest["files"][0]["sha256"] = "0" * 64
        data["manifest.json"] = json.dumps(manifest).encode()
        if extra:
            data["../outside"] = b"bad"
        return self.bundle(folder, list(data.items()))

    def docker_tar(self, folder, *, unsafe=False):
        archive = Path(folder) / "docker.tgz"
        with tarfile.open(archive, "w:gz") as bundle:
            for name in ("docker", "dockerd", "containerd", "containerd-shim-runc-v2", "runc"):
                entry = tarfile.TarInfo("docker/" + name)
                data = name.encode()
                entry.size = len(data)
                bundle.addfile(entry, io.BytesIO(data))
            if unsafe:
                entry = tarfile.TarInfo("docker/escape")
                entry.type = tarfile.SYMTYPE
                entry.linkname = "/etc/passwd"
                bundle.addfile(entry)
        return archive

    def test_runtime_bundle_verifies_each_payload(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = self.runtime_bundle(folder)
            manifest = worker.extract_runtime_bundle(archive, Path(folder) / "runtime")
            self.assertEqual(len(manifest["files"]), 5)
        with tempfile.TemporaryDirectory() as folder:
            archive = self.runtime_bundle(folder, bad_hash=True)
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                worker.extract_runtime_bundle(archive, Path(folder) / "runtime")

    def test_runtime_bundle_rejects_extra_paths_and_oversize(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = self.runtime_bundle(folder, extra=True)
            with self.assertRaisesRegex(ValueError, "approved flat"):
                worker.extract_runtime_bundle(archive, Path(folder) / "runtime")
        with tempfile.TemporaryDirectory() as folder:
            archive = self.runtime_bundle(folder)
            with patch.object(worker, "MAX_RUNTIME_BYTES", 10), self.assertRaisesRegex(ValueError, "256 MiB"):
                worker.extract_runtime_bundle(archive, Path(folder) / "runtime")

    def test_docker_tar_only_writes_regular_new_binaries(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = self.docker_tar(folder)
            output = Path(folder) / "bin"
            names = worker.extract_docker_binaries(archive, output)
            self.assertIn("docker", names)
            self.assertEqual((output / "docker").read_bytes(), b"docker")
            self.assertEqual(stat.S_IMODE((output / "docker").stat().st_mode), 0o755)
            with self.assertRaisesRegex(ValueError, "overwrite"):
                worker.extract_docker_binaries(archive, output)
        with tempfile.TemporaryDirectory() as folder:
            archive = self.docker_tar(folder, unsafe=True)
            with self.assertRaisesRegex(ValueError, "non-regular"):
                worker.extract_docker_binaries(archive, Path(folder) / "bin")

    def test_existing_docker_is_not_replaced(self):
        with patch.object(worker.shutil, "which", return_value="/usr/bin/docker"), \
                patch.object(worker.subprocess, "run") as command:
            self.assertEqual(worker.install_offline_runtime({}, Path("/unused"))["status"], "existing-docker-preserved")
        command.assert_not_called()
        self.assertIn("--host=unix:///var/run/docker.sock", worker.DOCKER_SERVICE)
        self.assertNotIn("tcp://", worker.DOCKER_SERVICE)
        self.assertIn("--bridge=none --iptables=false --ip-forward=false", worker.DOCKER_SERVICE)

    def test_command_failure_preserves_captured_output(self):
        error = subprocess.CalledProcessError(1, ["docker", "run"], output="CUDA probe started", stderr="driver unavailable")
        details = worker.failure_details(error)
        self.assertIn("CalledProcessError", details)
        self.assertIn("Captured stdout:\nCUDA probe started", details)
        self.assertIn("Captured stderr:\ndriver unavailable", details)

    def test_timeout_output_and_plain_exception_are_supported(self):
        error = subprocess.TimeoutExpired(["docker", "load"], 10, output=b"loading layer", stderr=b"timeout\xff")
        details = worker.failure_details(error)
        self.assertIn("loading layer", details)
        self.assertIn("timeout\ufffd", details)
        self.assertIn("FileNotFoundError: docker", worker.failure_details(FileNotFoundError("docker")))

    def bundle(self, folder, files):
        archive = Path(folder) / "resume.zip"
        with zipfile.ZipFile(archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for name, data in files:
                bundle.writestr(name, data)
        return archive

    def test_valid_resume_bundle_and_hash(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = self.bundle(folder, [("checkpoint.pt", b"checkpoint"), ("replay-buffer-0002.jsonl", b"replay")])
            worker.verify_sha256(archive, hashlib.sha256(archive.read_bytes()).hexdigest())
            with self.assertRaisesRegex(RuntimeError, "SHA256"):
                worker.verify_sha256(archive, "0" * 64)
            output = Path(folder) / "unpacked"
            worker.extract_resume_bundle(archive, output)
            self.assertEqual((output / "checkpoint.pt").read_bytes(), b"checkpoint")
            self.assertEqual((output / "replay-buffer-0002.jsonl").read_bytes(), b"replay")

    def test_resume_rejects_paths_extra_files_and_symlinks(self):
        invalid_entries = ["../replay-buffer-0002.jsonl", "/replay-buffer-0002.jsonl", "nested/replay-buffer-0002.jsonl", "replay-buffer-0002\\.jsonl"]
        for name in invalid_entries:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                archive = self.bundle(folder, [("checkpoint.pt", b"data"), (name, b"data")])
                with self.assertRaises(ValueError):
                    worker.extract_resume_bundle(archive, Path(folder) / "unpacked")
        with tempfile.TemporaryDirectory() as folder:
            archive = self.bundle(folder, [("checkpoint.pt", b"data"), ("replay-buffer-0002.jsonl", b"data"), ("extra.pt", b"data")])
            with self.assertRaises(ValueError):
                worker.extract_resume_bundle(archive, Path(folder) / "unpacked")
        with tempfile.TemporaryDirectory() as folder:
            symlink = zipfile.ZipInfo("checkpoint.pt")
            symlink.create_system = 3
            symlink.external_attr = (stat.S_IFLNK | 0o777) << 16
            archive = self.bundle(folder, [(symlink, b"/etc/passwd"), ("replay-buffer-0002.jsonl", b"data")])
            with self.assertRaises(ValueError):
                worker.extract_resume_bundle(archive, Path(folder) / "unpacked")

    def test_resume_size_limit_is_uncompressed(self):
        with tempfile.TemporaryDirectory() as folder:
            archive = self.bundle(folder, [("checkpoint.pt", b"x" * 10000), ("replay-buffer-0002.jsonl", b"y" * 10000)])
            self.assertLess(archive.stat().st_size, 1000)
            with patch.object(worker, "MAX_RESUME_BYTES", 1000), self.assertRaisesRegex(ValueError, "Uncompressed"):
                worker.extract_resume_bundle(archive, Path(folder) / "unpacked")

    def test_benchmark_and_training_use_readonly_resume(self):
        config = {"image_id": "sha256:test", "pipeline_args": [], "threads": 3}
        training = worker.training_command(config, Path("/host/output"), resume=Path("/host/resume"))
        benchmark = worker.benchmark_command(config, Path("/host/output"), resume=Path("/host/resume"))
        for command in (training, benchmark):
            self.assertIn("type=bind,source=/host/resume,target=/resume,readonly", command)
            self.assertIn("--pull=never", command)
        self.assertEqual(training[training.index("--resume") + 1], "/resume/checkpoint.pt")
        self.assertEqual(benchmark[benchmark.index("--output") + 1], "/output/benchmark.json")
        self.assertEqual(benchmark[benchmark.index("--threads") + 1], "3")
        self.assertEqual(benchmark[benchmark.index("--checkpoint") + 1], "/resume/checkpoint.pt")

    def test_upload_environment_skips_bucket_permission_check(self):
        self.assertEqual(worker.cloud_environment()["CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED"], "False")

    def test_shutdown_uploads_even_if_container_stop_times_out(self):
        with patch.object(worker, "metadata", return_value='{"output_uri":"gs://test/run"}'), \
                patch.object(worker.subprocess, "run", side_effect=[subprocess.TimeoutExpired(["docker"], 25), None]) as command, \
                contextlib.redirect_stderr(io.StringIO()):
            worker.shutdown()
        upload = command.call_args_list[-1]
        self.assertEqual(upload.args[0][:3], ["gcloud", "storage", "rsync"])
        self.assertEqual(upload.kwargs["env"]["CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED"], "False")

    def test_modern_docker_resolves_expected_index_id(self):
        expected = "sha256:" + "a" * 64
        with patch.object(worker.subprocess, "run", return_value=SimpleNamespace(returncode=0, stdout=expected + "\n")) as inspect:
            result = worker.resolve_loaded_image(expected, "Loaded image: ghq-train:pilot\n")
        self.assertEqual(result, {"requested_id": expected, "resolved_id": expected, "loaded_tag": "ghq-train:pilot"})
        self.assertEqual(inspect.call_count, 1)
        self.assertEqual(inspect.call_args.args[0][-1], expected)

    def test_classic_docker_resolves_archive_tag_to_config_id(self):
        expected, classic = "sha256:" + "a" * 64, "sha256:" + "b" * 64
        with patch.object(worker.subprocess, "run", side_effect=[SimpleNamespace(returncode=1, stdout=""),
                SimpleNamespace(returncode=0, stdout=classic + "\n")]) as inspect:
            result = worker.resolve_loaded_image(expected, "Loaded image: ghq-train:pilot\n")
        self.assertEqual(result, {"requested_id": expected, "resolved_id": classic, "loaded_tag": "ghq-train:pilot"})
        self.assertEqual([call.args[0][-1] for call in inspect.call_args_list], [expected, "ghq-train:pilot"])
        self.assertTrue(all(call.args[0][:3] == ["docker", "image", "inspect"] for call in inspect.call_args_list))

    def test_archive_tag_fallback_rejects_ambiguous_or_missing_names(self):
        for output in ("Loaded image: ghq-train:pilot\nLoaded image: other:tag\n", "Loaded image ID: sha256:" + "b" * 64):
            with self.subTest(output=output), patch.object(worker.subprocess, "run", return_value=SimpleNamespace(returncode=1, stdout="")) as inspect:
                with self.assertRaisesRegex(RuntimeError, "exactly one unambiguous"):
                    worker.resolve_loaded_image("sha256:" + "a" * 64, output)
            self.assertEqual(inspect.call_count, 1)

    def test_archive_tag_fallback_requires_local_immutable_id(self):
        with patch.object(worker.subprocess, "run", side_effect=[SimpleNamespace(returncode=1, stdout=""),
                SimpleNamespace(returncode=0, stdout="ghq-train:pilot\n")]):
            with self.assertRaisesRegex(RuntimeError, "invalid immutable"):
                worker.resolve_loaded_image("sha256:" + "a" * 64, "Loaded image: ghq-train:pilot\n")


if __name__ == "__main__":
    unittest.main()
