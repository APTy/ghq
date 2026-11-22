"""GPU VM guest runner. Only the attached bucket-scoped identity is used."""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import signal
import stat
import subprocess
import sys
import tarfile
import threading
import time
import traceback
import urllib.request
import zipfile

MAX_RESUME_BYTES = 4 * 1024 ** 3
MAX_RUNTIME_BYTES = 256 * 1024 ** 2
MAX_DOCKER_BYTES = 512 * 1024 ** 2
MAX_SOURCE_BYTES = 64 * 1024 ** 2
RUNTIME_PACKAGES = (
    "libnvidia-container1_1.20.2-1_amd64.deb",
    "libnvidia-container-tools_1.20.2-1_amd64.deb",
    "nvidia-container-toolkit-base_1.20.2-1_amd64.deb",
    "nvidia-container-toolkit_1.20.2-1_amd64.deb",
)
RUNTIME_FILES = {"docker.tgz", "manifest.json", *RUNTIME_PACKAGES}
DOCKER_SERVICE = """[Unit]
Description=GHQ private offline Docker runtime
After=network.target
[Service]
ExecStart=/usr/local/bin/dockerd --host=unix:///var/run/docker.sock --bridge=none --iptables=false --ip-forward=false --ip-masq=false
Restart=on-failure
Delegate=yes
KillMode=process
LimitNOFILE=1048576
TasksMax=infinity
[Install]
WantedBy=multi-user.target
"""


def cloud_environment() -> dict:
    # Object User can upload objects, but cannot read bucket metadata for
    # gcloud's optional parallel-composite compatibility check.
    return {**os.environ, "CLOUDSDK_STORAGE_PARALLEL_COMPOSITE_UPLOAD_ENABLED": "False"}


def failure_details(error: BaseException) -> str:
    """Retain captured command output that Python's traceback does not show."""
    details = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    for name in ("stdout", "stderr"):
        value = getattr(error, name, None)
        if value:
            if isinstance(value, bytes):
                value = value.decode("utf-8", errors="replace")
            details += f"\nCaptured {name}:\n{value}\n"
    return details


def verify_sha256(path: Path, expected: str):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise RuntimeError(f"SHA256 does not match the reviewed archive: {path.name}")


def resolve_loaded_image(expected_id: str, load_output: str) -> dict:
    """Resolve only local images after loading the SHA-verified archive.

    Containerd can expose an index digest while classic Docker exposes the
    image-config digest. The archive's single loaded tag bridges that change.
    """
    def inspect(reference):
        result = subprocess.run(["docker", "image", "inspect", "--format={{.Id}}", reference],
                                text=True, capture_output=True, timeout=60, check=False)
        if result.returncode:
            return None
        image_id = result.stdout.strip()
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_id):
            raise RuntimeError("Docker inspect returned an invalid immutable image ID")
        return image_id

    tags = re.findall(r"^Loaded image: (\S+)$", load_output, re.MULTILINE)
    resolved = inspect(expected_id)
    loaded_tag = tags[0] if len(tags) == 1 else None
    if resolved is None:
        if len(tags) != 1 or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._/:-]*:[A-Za-z0-9_][A-Za-z0-9_.-]*", tags[0]):
            raise RuntimeError("Expected image ID unavailable and archive did not load exactly one unambiguous image tag")
        resolved = inspect(loaded_tag)
        if resolved is None:
            raise RuntimeError("Image tag from verified archive is not available locally")
    return {"requested_id": expected_id, "resolved_id": resolved, "loaded_tag": loaded_tag}


def extract_runtime_bundle(archive: Path, destination: Path) -> dict:
    if archive.stat().st_size > MAX_RUNTIME_BYTES:
        raise ValueError("Runtime archive exceeds 256 MiB")
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        if len(entries) != len(RUNTIME_FILES) or {entry.filename for entry in entries} != RUNTIME_FILES:
            raise ValueError("Runtime ZIP must contain exactly the approved flat runtime files")
        if sum(entry.file_size for entry in entries) > MAX_RUNTIME_BYTES:
            raise ValueError("Uncompressed runtime ZIP exceeds 256 MiB")
        for entry in entries:
            kind = stat.S_IFMT(entry.external_attr >> 16)
            if entry.is_dir() or kind not in (0, stat.S_IFREG) or entry.flag_bits & 1:
                raise ValueError("Runtime ZIP entries must be unencrypted regular files")
        destination.mkdir(parents=True, exist_ok=False)
        written = 0
        for entry in entries:
            with bundle.open(entry) as source, (destination / entry.filename).open("xb") as target:
                while chunk := source.read(8 * 1024 * 1024):
                    written += len(chunk)
                    if written > MAX_RUNTIME_BYTES:
                        raise ValueError("Uncompressed runtime ZIP exceeds 256 MiB")
                    target.write(chunk)
    manifest = json.loads((destination / "manifest.json").read_text())
    files = manifest.get("files", [])
    if (not isinstance(files, list) or len(files) != 5 or
            {item.get("name") for item in files} != RUNTIME_FILES - {"manifest.json"}):
        raise ValueError("Runtime manifest must identify exactly the five bundled payloads")
    if manifest.get("docker_version") != "29.10.0" or manifest.get("nvidia_container_toolkit_version") != "1.20.2-1":
        raise ValueError("Runtime manifest versions differ from the pinned runtime")
    for item in files:
        path = destination / item["name"]
        if not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))) or path.stat().st_size != item.get("bytes"):
            raise ValueError("Runtime manifest size or digest is invalid")
        verify_sha256(path, item["sha256"])
    return manifest


def extract_docker_binaries(archive: Path, destination: Path) -> list[str]:
    allowed = {"docker", "dockerd", "containerd", "containerd-shim-runc-v2", "ctr", "runc", "docker-init", "docker-proxy"}
    with tarfile.open(archive, "r:gz") as bundle:
        members = bundle.getmembers()
        files = []
        for entry in members:
            if entry.isdir() and entry.name.rstrip("/") == "docker":
                continue
            if not entry.isfile() or entry.name not in {"docker/" + name for name in allowed}:
                raise ValueError("Docker tar contains an unexpected path or non-regular file")
            files.append(entry)
        names = [Path(entry.name).name for entry in files]
        if len(names) != len(set(names)) or not {"docker", "dockerd", "containerd", "containerd-shim-runc-v2", "runc"} <= set(names):
            raise ValueError("Docker tar has missing or duplicate binaries")
        if sum(entry.size for entry in files) > MAX_DOCKER_BYTES:
            raise ValueError("Docker binaries exceed 512 MiB")
        if any((destination / name).exists() or (destination / name).is_symlink() for name in names):
            raise ValueError("Refusing to overwrite an existing runtime binary")
        destination.mkdir(parents=True, exist_ok=True)
        written = 0
        for entry in files:
            path = destination / Path(entry.name).name
            with bundle.extractfile(entry) as source, path.open("xb") as target:
                while chunk := source.read(8 * 1024 * 1024):
                    written += len(chunk)
                    if written > MAX_DOCKER_BYTES:
                        raise ValueError("Docker binaries exceed 512 MiB")
                    target.write(chunk)
            path.chmod(0o755)
    return names


def install_offline_runtime(config: dict, output: Path) -> dict:
    """Bootstrap only this fresh VM from its reviewed, private GCS archive."""
    if shutil.which("docker"):
        return {"status": "existing-docker-preserved"}
    archive = Path("/opt/ghq-job/runtime.zip")
    directory = Path("/opt/ghq-job/runtime")
    with (output / "runtime-install.log").open("w") as log:
        def run(command, timeout=180):
            return subprocess.run(command, check=True, timeout=timeout, stdout=log,
                                  stderr=subprocess.STDOUT, env=cloud_environment())
        run(["gcloud", "storage", "cp", config["runtime_bundle_uri"], str(archive), "--quiet"], timeout=300)
        verify_sha256(archive, config["runtime_bundle_sha256"])
        manifest = extract_runtime_bundle(archive, directory)
        (output / "runtime-manifest.json").write_text(json.dumps(manifest, indent=2))
        extract_docker_binaries(directory / "docker.tgz", Path("/usr/local/bin"))
        run(["dpkg", "--install", *(str(directory / name) for name in RUNTIME_PACKAGES)])
        run(["nvidia-ctk", "runtime", "configure", "--runtime=docker"])
        service = Path("/etc/systemd/system/ghq-docker.service")
        with service.open("x") as handle:
            handle.write(DOCKER_SERVICE)
        try:
            run(["systemctl", "daemon-reload"], timeout=30)
            run(["systemctl", "start", "ghq-docker.service"], timeout=30)
            deadline = time.monotonic() + 90
            while time.monotonic() < deadline:
                ready = subprocess.run(["docker", "info"], timeout=5, stdout=log, stderr=subprocess.STDOUT, check=False)
                if ready.returncode == 0:
                    break
                time.sleep(2)
            else:
                raise RuntimeError("Offline Docker daemon failed to become ready within 90 seconds")
        except (OSError, subprocess.SubprocessError, RuntimeError):
            subprocess.run(["journalctl", "--unit=ghq-docker.service", "--no-pager", "--lines=100"],
                           stdout=log, stderr=subprocess.STDOUT, timeout=30, check=False)
            raise
        run(["docker", "version"])
        run(["nvidia-ctk", "--version"])
        run(["nvidia-container-cli", "--version"])
    archive.unlink()
    return {"status": "installed-from-verified-bundle", "docker_version": manifest["docker_version"],
            "nvidia_container_toolkit_version": manifest["nvidia_container_toolkit_version"]}


def extract_source_bundle(archive: Path, destination: Path) -> dict:
    """Verify a bounded source-only bundle before overlaying the container app."""
    if archive.stat().st_size > MAX_SOURCE_BYTES:
        raise ValueError("Source ZIP exceeds 64 MiB")
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        names = [entry.filename for entry in entries]
        required = {"source-manifest.json", "ghqnet/__init__.py", "ghqnet/pipeline.py",
                    "public/engine.py", "public/search.py"}
        if len(entries) > 128 or len(set(names)) != len(names) or not required <= set(names):
            raise ValueError("Source ZIP is incomplete, duplicated, or has too many files")
        for entry in entries:
            allowed = entry.filename in {"source-manifest.json", "public/engine.py", "public/search.py"}
            allowed |= bool(re.fullmatch(r"ghqnet/[a-zA-Z_][a-zA-Z_0-9]*\.py", entry.filename))
            allowed |= bool(re.fullmatch(r"training/benchmarks/[a-zA-Z_][a-zA-Z_0-9.-]*\.manifest\.json", entry.filename))
            kind = stat.S_IFMT(entry.external_attr >> 16)
            if not allowed or entry.is_dir() or kind not in (0, stat.S_IFREG) or entry.flag_bits & 1:
                raise ValueError("Source ZIP contains an unexpected path or nonregular file")
        if sum(entry.file_size for entry in entries) > MAX_SOURCE_BYTES:
            raise ValueError("Uncompressed source ZIP exceeds 64 MiB")
        manifest = json.loads(bundle.read("source-manifest.json"))
        files = manifest.get("files") if isinstance(manifest, dict) else None
        if (not isinstance(manifest, dict) or manifest.get("version") != 1 or not isinstance(files, list) or
                not all(isinstance(item, dict) and isinstance(item.get("name"), str) for item in files) or
                len(files) != len(entries) - 1 or
                {item.get("name") for item in files} != set(names) - {"source-manifest.json"}):
            raise ValueError("Source manifest does not identify every payload exactly once")
        expected = {item["name"]: item for item in files}
        destination.mkdir(parents=True, exist_ok=False)
        written = 0
        for entry in entries:
            path = destination / entry.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            with bundle.open(entry) as source, path.open("xb") as target:
                while chunk := source.read(8 * 1024 * 1024):
                    written += len(chunk)
                    if written > MAX_SOURCE_BYTES:
                        raise ValueError("Uncompressed source ZIP exceeds 64 MiB")
                    target.write(chunk)
            if entry.filename in expected:
                item = expected[entry.filename]
                if path.stat().st_size != item.get("bytes") or not re.fullmatch(r"[0-9a-f]{64}", str(item.get("sha256", ""))):
                    raise ValueError("Invalid source manifest size or SHA256")
                verify_sha256(path, item["sha256"])
    return manifest


def extract_resume_bundle(archive: Path, destination: Path):
    """Extract just a trusted checkpoint and its bounded, flat replay buffer."""
    if archive.stat().st_size > MAX_RESUME_BYTES:
        raise ValueError("Resume archive exceeds 4 GiB")
    with zipfile.ZipFile(archive) as bundle:
        entries = bundle.infolist()
        names = [entry.filename for entry in entries]
        if not 2 <= len(entries) <= 32 or len(set(names)) != len(names) or "checkpoint.pt" not in names:
            raise ValueError("Resume ZIP must contain checkpoint.pt and bounded unique companion files")
        if len([name for name in names if re.fullmatch(r"replay-buffer-[0-9]{4,}\.jsonl", name)]) != 1:
            raise ValueError("Resume ZIP must contain exactly one replay buffer")
        for entry in entries:
            if not (entry.filename == "checkpoint.pt" or
                    re.fullmatch(r"(?:replay|validation)-buffer-[0-9]{4,}\.jsonl", entry.filename) or
                    re.fullmatch(r"curriculum-[0-9]{4,}\.json", entry.filename) or
                    re.fullmatch(r"history-[0-9a-f]{16}\.pt", entry.filename)):
                raise ValueError("Resume ZIP contains a non-flat or unexpected filename")
            kind = stat.S_IFMT(entry.external_attr >> 16)
            if entry.is_dir() or kind not in (0, stat.S_IFREG) or entry.flag_bits & 1:
                raise ValueError("Resume ZIP entries must be unencrypted regular files")
        if sum(entry.file_size for entry in entries) > MAX_RESUME_BYTES:
            raise ValueError("Uncompressed resume bundle exceeds 4 GiB")
        destination.mkdir(parents=True, exist_ok=False)
        written = 0
        for entry in entries:
            with bundle.open(entry) as source, (destination / entry.filename).open("xb") as target:
                while chunk := source.read(8 * 1024 * 1024):
                    written += len(chunk)
                    if written > MAX_RESUME_BYTES:
                        raise ValueError("Uncompressed resume bundle exceeds 4 GiB")
                    target.write(chunk)


def container_command(config: dict, output: Path, *, resume: Path | None = None) -> list[str]:
    command = ["docker", "run", "--rm", "--pull=never", "--network=none", "--gpus=all", "--shm-size=2g",
               "--cap-drop=ALL", "--security-opt=no-new-privileges", "--mount",
               f"type=bind,source={output},target=/output"]
    if resume is not None:
        command.extend(["--mount", f"type=bind,source={resume},target=/resume,readonly"])
    if config.get("source_directory"):
        command.extend(["--mount", f"type=bind,source={config['source_directory']},target=/app,readonly"])
    return command


def metadata(key: str) -> str:
    request = urllib.request.Request(
        "http://metadata.google.internal/computeMetadata/v1/instance/attributes/" + key,
        headers={"Metadata-Flavor": "Google"},
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read().decode()


def training_command(config: dict, output: Path, *, resume: Path | None = None) -> list[str]:
    """Keep fresh pipeline state separate from the worker's logs and metadata."""
    command = container_command(config, output, resume=resume)
    command.extend(["--name=ghq-training", config["image_id"], "--output", "/output/run",
                    "--device", "cuda", *config["pipeline_args"]])
    if resume is not None:
        command.extend(["--resume", "/resume/checkpoint.pt"])
    return command


def benchmark_command(config: dict, output: Path, *, resume: Path | None = None) -> list[str]:
    command = container_command(config, output, resume=resume)
    command.extend(["--name=ghq-benchmark", "--entrypoint=python", config["image_id"],
                    "-m", "ghqnet.benchmark", "--output", "/output/benchmark.json",
                    "--device", "cuda", "--repeats", "10", "--threads", str(config.get("threads", 2))])
    if resume is not None:
        command.extend(["--checkpoint", "/resume/checkpoint.pt"])
    return command


def collector_benchmark_command(config: dict, output: Path, *, resume: Path | None = None) -> list[str]:
    """A separate read-only-checkpoint workload, with no pipeline entry point."""
    if resume is None:
        raise ValueError("Collector benchmark requires the verified resume checkpoint")
    if config.get("pipeline_args"):
        raise ValueError("Collector benchmark cannot include training arguments")
    arguments = config.get("collector_benchmark_args")
    if (not isinstance(arguments, list) or not arguments or
            not all(isinstance(value, str) and value and "\x00" not in value for value in arguments)):
        raise ValueError("Collector benchmark requires reviewed argument entries")
    protected = {"--output", "--device", "--checkpoint", "--help"}
    if any(value.split("=", 1)[0] in protected for value in arguments):
        raise ValueError("Collector benchmark cannot override worker-owned paths or device")
    budget = config.get("collector_benchmark_max_seconds")
    if type(budget) not in (int, float) or not math.isfinite(budget) or budget <= 0:
        raise ValueError("Collector benchmark requires a finite positive time budget")
    if arguments.count("--max-seconds") != 1:
        raise ValueError("Collector benchmark requires one explicit max-seconds argument")
    position = arguments.index("--max-seconds")
    if position + 1 >= len(arguments) or float(arguments[position + 1]) != budget:
        raise ValueError("Collector benchmark argument and watchdog budgets differ")
    command = container_command(config, output, resume=resume)
    command.extend(["--name=ghq-collector-benchmark", "--entrypoint=python", config["image_id"],
                    "-m", "ghqnet.benchmark_collector", "/resume/checkpoint.pt", "--device", "cuda",
                    "--output", "/output/collector-benchmark.json", *arguments])
    return command


def replay_ablation_command(config: dict, output: Path, *, resume: Path | None = None) -> list[str]:
    """Offline optimizer comparison: frozen inputs, separate diagnostic outputs."""
    if resume is None:
        raise ValueError("Replay ablation requires the verified resume checkpoint")
    if config.get("pipeline_args") or config.get("collector_benchmark_args"):
        raise ValueError("Replay ablation cannot include pipeline or collector arguments")
    arguments = config.get("replay_ablation_args")
    if (not isinstance(arguments, list) or not arguments or
            not all(isinstance(value, str) and value and "\x00" not in value for value in arguments)):
        raise ValueError("Replay ablation requires reviewed argument entries")
    # The guest receives argv from the launcher, but independently reject path
    # overrides, abbreviations, duplicate flags, and malformed numeric values.
    limits = {"--epochs": (1, 4), "--batch-size": (1, 512), "--threads": (1, 64),
              "--seed": (0, 2 ** 32 - 1)}
    seen, index, argument_budget = set(), 0, None
    while index < len(arguments):
        name = arguments[index]
        if name in seen or name not in {*limits, "--learning-rates", "--max-seconds"}:
            raise ValueError(f"Unsupported or repeated replay ablation argument: {name}")
        seen.add(name)
        index += 1
        end = index
        while end < len(arguments) and not arguments[end].startswith("--"):
            end += 1
        values = arguments[index:end]
        if name == "--learning-rates":
            rates = [float(value) for value in values]
            if (not 2 <= len(rates) <= 4 or any(not math.isfinite(rate) or not 0 < rate <= 1 for rate in rates)
                    or len(set(rates)) != len(rates)):
                raise ValueError("Replay ablation requires 2–4 distinct finite learning rates in (0, 1]")
        elif name in limits:
            if len(values) != 1 or not re.fullmatch(r"[0-9]+", values[0]):
                raise ValueError(f"Replay ablation {name} requires one integer")
            lower, upper = limits[name]
            if not lower <= int(values[0]) <= upper:
                raise ValueError(f"Replay ablation {name} is outside its allowed range")
        else:
            if len(values) != 1 or not math.isfinite(float(values[0])) or float(values[0]) <= 0:
                raise ValueError("Replay ablation requires one finite positive max-seconds value")
            argument_budget = float(values[0])
        index = end
    budget = config.get("replay_ablation_max_seconds")
    if type(budget) not in (int, float) or not 0 < budget <= 13800 or not math.isfinite(budget):
        raise ValueError("Replay ablation requires a finite time budget in (0, 13800]")
    if argument_budget != budget:
        raise ValueError("Replay ablation argument and watchdog budgets differ")
    command = container_command(config, output, resume=resume)
    command.extend(["--name=ghq-replay-ablation", "--entrypoint=python", config["image_id"],
                    "-m", "ghqnet.ablate_replay", "/resume/checkpoint.pt", "--output", "/output/replay-ablation",
                    "--device", "cuda", *arguments])
    return command


def workload_container_name(config: dict) -> str:
    names = {"pipeline": "ghq-training", "collector-benchmark": "ghq-collector-benchmark",
             "replay-ablation": "ghq-replay-ablation"}
    mode = config.get("job_mode", "pipeline")
    if mode not in names:
        raise ValueError(f"Unknown job mode: {mode}")
    return names[mode]


def workload(config: dict, output: Path, *, resume: Path | None = None):
    """Select exactly one workload; legacy configurations remain training jobs."""
    mode = config.get("job_mode", "pipeline")
    if mode == "collector-benchmark":
        return (collector_benchmark_command(config, output, resume=resume),
                "collector-benchmark.log", config["collector_benchmark_max_seconds"] + 120)
    if mode == "replay-ablation":
        return (replay_ablation_command(config, output, resume=resume),
                "replay-ablation.log", config["replay_ablation_max_seconds"] + 120)
    if mode != "pipeline":
        raise ValueError(f"Unknown job mode: {mode}")
    return training_command(config, output, resume=resume), "training.log", None


def wait_for_workload(process, config: dict, timeout=None):
    """Bound standalone diagnostics even if their cooperative deadline fails."""
    if timeout is None:
        return process.wait()
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        try:
            subprocess.run(["docker", "stop", "--time", "20", workload_container_name(config)],
                           timeout=30, check=False)
        finally:
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
        raise


def main() -> int:
    config = json.loads(metadata("ghq-config"))
    output = Path("/var/lib/ghq-output")
    output.mkdir(parents=True, exist_ok=True)
    destination = config["output_uri"]
    stopped = threading.Event()
    sync_lock = threading.Lock()
    process = None
    gpu_metrics = None
    gpu_metrics_file = None

    def run(args, **kwargs):
        if args[0] == "gcloud":
            kwargs.setdefault("env", cloud_environment())
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
            subprocess.run(["docker", "stop", "--time", "20", workload_container_name(config)],
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
        gpu = run(["nvidia-smi", "--query-gpu=name,uuid,driver_version,memory.total,pci.bus_id",
                   "--format=csv"], timeout=60, capture_output=True, text=True)
        hardware = {"platform": platform.platform(), "cpu_count": os.cpu_count(),
                    "gpu": gpu.stdout, "collected_at": time.time()}
        (output / "hardware.json").write_text(json.dumps(hardware, indent=2))
        if config.get("runtime_bundle_uri"):
            hardware["runtime_install"] = install_offline_runtime(config, output)
            (output / "hardware.json").write_text(json.dumps(hardware, indent=2))
        run(["docker", "info"], timeout=60)
        archive = Path("/opt/ghq-job/image.tar.gz")
        run(["gcloud", "storage", "cp", config["image_uri"], str(archive), "--quiet"], timeout=1800)
        verify_sha256(archive, config["image_sha256"])
        loaded = run(["docker", "load", "--input", str(archive)], timeout=900, capture_output=True, text=True)
        (output / "docker-load.log").write_text(loaded.stdout + loaded.stderr)
        hardware["docker_image"] = resolve_loaded_image(config["image_id"], loaded.stdout)
        config["image_id"] = hardware["docker_image"]["resolved_id"]
        (output / "hardware.json").write_text(json.dumps(hardware, indent=2))
        archive.unlink()
        # Fail before self-play if drivers/container runtime cannot access CUDA.
        cuda = run(["docker", "run", "--rm", "--pull=never", "--network=none", "--gpus=all",
             "--entrypoint=python", config["image_id"], "-c",
             "import json,torch; assert torch.cuda.is_available(), 'CUDA unavailable'; "
             "print(json.dumps({'torch':torch.__version__,'cuda':torch.version.cuda,"
             "'gpu':torch.cuda.get_device_name(0)}))"], timeout=120, capture_output=True, text=True)
        hardware["torch_cuda"] = json.loads(cuda.stdout)
        (output / "hardware.json").write_text(json.dumps(hardware, indent=2))
        resume = None
        if config.get("source_bundle_uri"):
            source_archive = Path("/opt/ghq-job/source.zip")
            run(["gcloud", "storage", "cp", config["source_bundle_uri"], str(source_archive), "--quiet"], timeout=300)
            verify_sha256(source_archive, config["source_bundle_sha256"])
            source_directory = Path("/opt/ghq-job/source")
            source_manifest = extract_source_bundle(source_archive, source_directory)
            config["source_directory"] = str(source_directory)
            hardware["source_bundle_sha256"] = config["source_bundle_sha256"]
            (output / "source-manifest.json").write_text(json.dumps(source_manifest, indent=2))
            (output / "hardware.json").write_text(json.dumps(hardware, indent=2))
            source_archive.unlink()
        if config.get("resume_bundle_uri"):
            resume_archive = Path("/opt/ghq-job/resume.zip")
            run(["gcloud", "storage", "cp", config["resume_bundle_uri"], str(resume_archive), "--quiet"], timeout=300)
            verify_sha256(resume_archive, config["resume_bundle_sha256"])
            resume = Path("/opt/ghq-job/resume")
            extract_resume_bundle(resume_archive, resume)
            resume_archive.unlink()
        if config.get("job_mode", "pipeline") == "pipeline":
            with (output / "benchmark.log").open("w") as log:
                run(benchmark_command(config, output, resume=resume), stdout=log, stderr=subprocess.STDOUT, timeout=300)
        # Save startup proof before the longer self-play phase starts.
        sync()
        gpu_metrics_file = (output / "gpu-metrics.csv").open("w")
        gpu_metrics = subprocess.Popen([
            "nvidia-smi", "--query-gpu=timestamp,index,uuid,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu",
            "--format=csv", "--loop=5"], stdout=gpu_metrics_file, stderr=subprocess.STDOUT)
        command, log_name, timeout = workload(config, output, resume=resume)
        with (output / log_name).open("w") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            result["exit_code"] = wait_for_workload(process, config, timeout)
    except BaseException as error:
        result["error"] = str(error)
        (output / "worker-error.txt").write_text(failure_details(error))
    finally:
        stopped.set()
        if gpu_metrics is not None:
            gpu_metrics.terminate()
            try:
                gpu_metrics.wait(timeout=10)
            except subprocess.TimeoutExpired:
                gpu_metrics.kill()
                gpu_metrics.wait(timeout=10)
        if gpu_metrics_file is not None:
            gpu_metrics_file.close()
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
    try:
        subprocess.run(["docker", "stop", "--time", "10", "ghq-training", "ghq-benchmark",
                        "ghq-collector-benchmark", "ghq-replay-ablation"],
                       check=False, timeout=35)
    except (subprocess.SubprocessError, OSError):
        traceback.print_exc()
    subprocess.run(["gcloud", "storage", "rsync", "/var/lib/ghq-output", config["output_uri"],
                    "--recursive", "--exclude=.*\\.tmp$", "--quiet"], check=False, timeout=60,
                   env=cloud_environment())


if __name__ == "__main__":
    if sys.argv[1:] == ["--shutdown"]:
        shutdown()
    else:
        raise SystemExit(main())
