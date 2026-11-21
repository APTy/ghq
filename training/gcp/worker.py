"""GPU VM guest runner. Only the attached bucket-scoped identity is used."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
import traceback
import urllib.request


def metadata(key: str) -> str:
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/attributes/" + key,
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode()


def training_command(config: dict, output: Path) -> list[str]:
    """Keep fresh pipeline state separate from the worker's logs and metadata."""
    return ["docker", "run", "--name=ghq-training", "--rm", "--network=none",
            "--gpus=all", "--shm-size=2g", "--cap-drop=ALL",
            "--security-opt=no-new-privileges", "--mount",
            f"type=bind,source={output},target=/output", config["image_id"],
            "--output", "/output/run", "--device", "cuda", *config["pipeline_args"]]


def main() -> int:
    config = json.loads(metadata("ghq-config"))
    output = Path("/var/lib/ghq-output")
    output.mkdir(parents=True, exist_ok=True)
    destination = config["output_uri"]
    stopped = threading.Event()
    sync_lock = threading.Lock()
    process = None

    def run(args, **kwargs):
        return subprocess.run(args, check=True, **kwargs)

    def sync() -> bool:
        with sync_lock:
            try:
                run(["gcloud", "storage", "rsync", str(output), destination,
                     "--recursive", "--exclude=.*\\.tmp$", "--quiet"], timeout=180)
                return True
            except (subprocess.SubprocessError, OSError) as error:
                print(f"Checkpoint sync failed: {error}", flush=True)
                return False

    def periodic_sync():
        while not stopped.wait(60):
            sync()

    def stop(signum, frame):
        if process is not None:
            subprocess.run(["docker", "stop", "--time", "20", "ghq-training"],
                           timeout=30, check=False)
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    thread = threading.Thread(target=periodic_sync, daemon=True)
    thread.start()
    result = {"run_id": config["run_id"], "started_at": time.time(), "exit_code": 1}
    (output / "job-config.json").write_text(json.dumps(config, indent=2))
    try:
        # The approved image must provide these tools. Do not apt-install from
        # the internet or silently fall back to CPU on a billed GPU machine.
        run(["nvidia-smi"], timeout=60)
        run(["docker", "info"], timeout=60)
        archive = Path("/opt/ghq-job/image.tar.gz")
        run(["gcloud", "storage", "cp", config["image_uri"], str(archive), "--quiet"], timeout=1800)
        digest = hashlib.sha256()
        with archive.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != config["image_sha256"]:
            raise RuntimeError("Docker archive SHA256 does not match the reviewed archive")
        run(["docker", "load", "--input", str(archive)], timeout=900)
        archive.unlink()
        # Fail before self-play if drivers/container runtime cannot access CUDA.
        run(["docker", "run", "--rm", "--network=none", "--gpus=all",
             "--entrypoint=python", config["image_id"], "-c",
             "import torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
             "print(torch.__version__, torch.cuda.get_device_name(0))"], timeout=120)
        command = training_command(config, output)
        with (output / "training.log").open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            result["exit_code"] = process.wait()
    except BaseException as error:
        result["error"] = str(error)
        (output / "worker-error.txt").write_text(traceback.format_exc())
    finally:
        stopped.set()
        result["finished_at"] = time.time()
        # Publish completion only after a successful final checkpoint upload.
        # The launcher deletes the VM upon seeing this object.
        uploaded = sync()
        for _ in range(2):
            if uploaded:
                break
            time.sleep(10)
            uploaded = sync()
        result["final_sync_succeeded"] = uploaded
        # If the upload failed, retain the VM until its hard deadline so the
        # shutdown hook has another chance; never claim the checkpoint is safe.
        status_name = "completion.json" if uploaded else "upload-failed.json"
        status_path = output / status_name
        status_path.write_text(json.dumps(result, indent=2))
        try:
            run(["gcloud", "storage", "cp", str(status_path), destination + "/" + status_name,
                 "--quiet"], timeout=60)
        except (subprocess.SubprocessError, OSError):
            traceback.print_exc()
    # Do not stop the VM here: stopping could leave a billed boot disk behind.
    # The launcher deletes it immediately; GCE's DELETE deadline is the fallback.
    return result["exit_code"]


def shutdown():
    """Best-effort final upload during host maintenance or the GCE deadline."""
    config = json.loads(metadata("ghq-config"))
    subprocess.run(["docker", "stop", "--time", "10", "ghq-training"], check=False, timeout=15)
    subprocess.run(["gcloud", "storage", "rsync", "/var/lib/ghq-output", config["output_uri"],
                    "--recursive", "--exclude=.*\\.tmp$", "--quiet"], check=False, timeout=60)


if __name__ == "__main__":
    if sys.argv[1:] == ["--shutdown"]:
        shutdown()
    else:
        raise SystemExit(main())
