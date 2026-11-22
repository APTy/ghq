#!/usr/bin/env python3
"""Read-only preflight by default; --launch creates exactly one bounded GPU VM."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import shlex
import subprocess
import tempfile
import time
import uuid

DIRECTORY = Path(__file__).resolve().parent
IMAGE_PROJECT = "deeplearning-platform-release"
IMAGE_FAMILY = "common-cu129-ubuntu-2204-nvidia-580"
MAX_RESUME_BYTES = 4 * 1024 ** 3
MAX_RUNTIME_BYTES = 256 * 1024 ** 2
MAX_SOURCE_BYTES = 64 * 1024 ** 2
GPU_CONFIG = {
    "l4": {"machine_type": "g2-standard-4", "accelerator_type": "nvidia-l4"},
    "t4": {"machine_type": "n1-standard-4", "accelerator_type": "nvidia-tesla-t4"},
}


def machine_type(gpu, cpus=4):
    if cpus not in (4, 8, 16) or (gpu == "l4" and cpus != 4):
        raise ValueError("Supported machines: L4 with 4 CPUs; T4 with 4, 8, or 16 CPUs")
    return f"n1-standard-{cpus}" if gpu == "t4" else GPU_CONFIG[gpu]["machine_type"]


def load_pipeline_config(path):
    """Turn a reviewed JSON object into argv, without shell interpretation.

    Paths controlling the output, device, and resume mount belong to the worker.
    Pipeline argparse/--validate-only must validate experiment-specific options.
    """
    raw = Path(path).read_bytes()
    if len(raw) > 64 * 1024:
        raise ValueError("Pipeline configuration exceeds 64 KiB")
    config = json.loads(raw)
    if not isinstance(config, dict) or not config:
        raise ValueError("Pipeline configuration must be a nonempty JSON object")
    arguments = []
    for name, value in config.items():
        if not re.fullmatch(r"[a-z][a-z0-9-]*", name) or name in {"output", "device", "resume", "help", "validate-only"}:
            raise ValueError(f"Reserved or invalid pipeline option: {name}")
        if isinstance(value, bool):
            if value:
                arguments.append("--" + name)
            continue
        if not isinstance(value, (str, int, float)) or (isinstance(value, float) and not math.isfinite(value)):
            raise ValueError(f"Pipeline option {name} must be a finite scalar or boolean")
        if isinstance(value, str) and (not value or value.startswith("--") or "\x00" in value):
            raise ValueError(f"Invalid pipeline option value: {name}")
        arguments.extend(["--" + name, str(value)])
    return arguments, config, hashlib.sha256(raw).hexdigest()


def load_collector_benchmark_config(path):
    """Parse only bounded collector controls; paths and device stay worker-owned.

    Values become separate argv entries, never shell text. This mode deliberately
    has no training flags and requires an explicit total benchmark time budget.
    """
    raw = Path(path).read_bytes()
    if len(raw) > 64 * 1024:
        raise ValueError("Collector benchmark configuration exceeds 64 KiB")
    config = json.loads(raw)
    if not isinstance(config, dict) or not config:
        raise ValueError("Collector benchmark configuration must be a nonempty JSON object")
    integer_options = {"games": (1, None), "max-turns": (1, None), "opening-turns": (0, None),
                       "seed": (0, 2 ** 32 - 1), "threads": (1, 64), "repeats": (1, None),
                       "position-count": (1, None), "teacher-depth": (1, None),
                       "teacher-beam-width": (1, None)}
    list_options = {"simulations": (1, None), "workers": (1, 64), "batch-size": (1, 64)}
    float_options = {"max-seconds": False, "request-timeout-seconds": False,
                     "game-timeout-seconds": False, "case-max-seconds": False,
                     "teacher-ms": False, "history-fraction": True, "cpu-fraction": True}
    arguments = []
    for name, value in config.items():
        if name in integer_options or name in list_options:
            limits = (integer_options | list_options)[name]
            values = value if name in list_options and isinstance(value, list) else [value]
            if not values or len(values) > 64 or len(set(map(str, values))) != len(values):
                raise ValueError(f"Collector benchmark option {name} needs distinct nonempty values (at most 64)")
            for item in values:
                if (type(item) is not int or item < limits[0] or
                        (limits[1] is not None and item > limits[1])):
                    raise ValueError(f"Invalid integer collector benchmark option: {name}")
        elif name in float_options or name == "batch-wait-ms":
            values = value if name == "batch-wait-ms" and isinstance(value, list) else [value]
            if not values or len(values) > 64 or len(set(map(str, values))) != len(values):
                raise ValueError(f"Collector benchmark option {name} needs distinct nonempty values (at most 64)")
            for item in values:
                if (type(item) not in (int, float) or not math.isfinite(item) or item < 0 or
                        (item == 0 and not (name == "batch-wait-ms" or float_options[name])) or
                        (name in {"history-fraction", "cpu-fraction"} and item > 1)):
                    raise ValueError(f"Invalid finite numeric collector benchmark option: {name}")
            if len(set(values)) != len(values):
                raise ValueError(f"Collector benchmark option {name} needs distinct numeric values")
        elif name == "inference-protocol":
            values = value if isinstance(value, list) else [value]
            if (not values or not all(isinstance(item, str) and item in {"auto", "legacy", "encoded"} for item in values)
                    or len(set(values)) != len(values)):
                raise ValueError("Collector benchmark inference-protocol must contain distinct supported protocols")
        elif name in {"profile-inference", "profile-sample"}:
            if not isinstance(value, bool):
                raise ValueError(f"Collector benchmark {name} must be boolean")
            if value:
                arguments.append("--" + name)
            continue
        elif name in {"positions", "historical-checkpoint"}:
            pattern = (r"/resume/(?:replay|validation)-buffer-[0-9]{4,}\.jsonl" if name == "positions" else
                       r"/resume/history-[0-9a-f]{16}\.pt")
            if not isinstance(value, str) or not re.fullmatch(pattern, value):
                raise ValueError(f"Collector benchmark option {name} must reference an allowlisted resume artifact")
            values = [value]
        else:
            raise ValueError(f"Reserved or unsupported collector benchmark option: {name}")
        arguments.extend(["--" + name, *(str(item) for item in values)])
    if "max-seconds" not in config:
        raise ValueError("Collector benchmark configuration requires explicit max-seconds")
    if config.get("profile-inference") and config.get("profile-sample"):
        raise ValueError("Use only one of profile-inference or profile-sample")
    if config.get("history-fraction", 0) + config.get("cpu-fraction", 0) > 1:
        raise ValueError("Collector benchmark opponent fractions cannot sum above one")
    if config.get("history-fraction", 0) > 0 and not config.get("historical-checkpoint"):
        raise ValueError("Historical benchmark games require a resume historical-checkpoint")
    return arguments, config, hashlib.sha256(raw).hexdigest()


def load_replay_ablation_config(path):
    """Allow only bounded offline optimizer-ablation controls, as literal argv."""
    raw = Path(path).read_bytes()
    if len(raw) > 64 * 1024:
        raise ValueError("Replay ablation configuration exceeds 64 KiB")
    config = json.loads(raw)
    if not isinstance(config, dict) or not config:
        raise ValueError("Replay ablation configuration must be a nonempty JSON object")
    integers = {"epochs": (1, 4), "batch-size": (1, 512), "threads": (1, 64),
                "seed": (0, 2 ** 32 - 1)}
    arguments = []
    for name, value in config.items():
        if name == "learning-rates":
            if (not isinstance(value, list) or not 2 <= len(value) <= 4 or
                    any(type(rate) not in (int, float) or not 0 < rate <= 1 or not math.isfinite(rate)
                        for rate in value) or len(set(value)) != len(value)):
                raise ValueError("Replay ablation learning-rates requires 2–4 distinct finite rates in (0, 1]")
            values = value
        elif name in integers:
            lower, upper = integers[name]
            if type(value) is not int or not lower <= value <= upper:
                raise ValueError(f"Replay ablation {name} must be an integer from {lower} to {upper}")
            values = [value]
        elif name == "max-seconds":
            if type(value) not in (int, float) or not 0 < value <= 13800 or not math.isfinite(value):
                raise ValueError("Replay ablation max-seconds must be finite and in (0, 13800]")
            values = [value]
        else:
            raise ValueError(f"Reserved or unsupported replay ablation option: {name}")
        arguments.extend(["--" + name, *(str(item) for item in values)])
    if "max-seconds" not in config:
        raise ValueError("Replay ablation configuration requires explicit max-seconds")
    return arguments, config, hashlib.sha256(raw).hexdigest()


def check_gpu_quotas(regional, global_quotas, gpu="l4", cpus=4):
    """Legacy quota APIs can omit the global metric; absence is not zero."""
    regional = {item["metric"]: item for item in regional}
    global_quotas = {item["metric"]: item for item in global_quotas}
    kind = gpu.upper()
    selected = regional.get(f"NVIDIA_{kind}_GPUS") or regional.get(f"GPU_FAMILY:NVIDIA_{kind}")
    if selected is None or selected["limit"] - selected["usage"] < 1:
        raise RuntimeError(f"Insufficient or unavailable regional {kind} quota")
    # GPU VMs with predefined runtimes consume preemptible quota when that
    # quota has been granted, even with STANDARD provisioning.
    for metric, quota in (
        (f"PREEMPTIBLE_NVIDIA_{kind}_GPUS", regional.get(f"PREEMPTIBLE_NVIDIA_{kind}_GPUS")),
        ("GPUS_ALL_REGIONS", global_quotas.get("GPUS_ALL_REGIONS")),
    ):
        if quota is not None and quota["limit"] - quota["usage"] < 1:
            raise RuntimeError("Insufficient quota: " + metric)
    if gpu == "t4":
        cpu = regional.get("CPUS")
        if cpu is None or cpu["limit"] - cpu["usage"] < cpus:
            raise RuntimeError(f"Insufficient or unavailable regional CPUS quota for n1-standard-{cpus}")
        for metric, quota in (("PREEMPTIBLE_CPUS", regional.get("PREEMPTIBLE_CPUS")),
                              ("CPUS_ALL_REGIONS", global_quotas.get("CPUS_ALL_REGIONS"))):
            if quota is not None and quota["limit"] - quota["usage"] < cpus:
                raise RuntimeError("Insufficient quota: " + metric)
    return "reported" if "GPUS_ALL_REGIONS" in global_quotas else "unreported; enforced by Compute at creation"


def confirmed_not_found(error):
    """Recognize resource absence, never generic CLI/network/auth failures."""
    return bool(re.search(r"\bNOT_FOUND\b|\bHTTP(?:Error)?[ :]+404\b|\bstatus(?: code)?[ :=]+404\b|"
                          r"The resource ['\"][^'\"]+['\"] was not found", error, re.IGNORECASE))


def gcloud(*args, project, optional=False, timeout=120):
    command = ["gcloud", *args, "--project=" + project, "--quiet", "--format=json"]
    result = subprocess.run(command, text=True, capture_output=True, timeout=timeout)
    if result.returncode:
        if optional and confirmed_not_found(result.stderr):
            return None
        raise RuntimeError(result.stderr.strip())
    value = json.loads(result.stdout or "null")
    if optional and not isinstance(value, dict):
        raise RuntimeError("Resource lookup returned no valid object")
    return value


def delete_vm(name, *, project, zone):
    """Bound cleanup attempts and return success only after confirmed absence."""
    command = ["gcloud", "compute", "instances", "delete", name,
               "--project=" + project, "--zone=" + zone, "--quiet"]
    last_error = "Instance still exists after deletion"
    for attempt in range(3):
        try:
            result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=60)
            if result.returncode:
                last_error = result.stderr.strip() or "Delete command failed"
        except (OSError, subprocess.SubprocessError) as error:
            last_error = str(error)
        try:
            instance = gcloud("compute", "instances", "describe", name, "--zone=" + zone,
                              project=project, optional=True, timeout=15)
            if instance is None:
                print("Confirmed VM deleted: " + name, flush=True)
                return True
            last_error = "Instance still exists with status " + instance.get("status", "unknown")
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as error:
            last_error = "Could not confirm deletion: " + str(error)
        if attempt < 2:
            time.sleep(2)
    print(f"CLEANUP NOT CONFIRMED for {name}: {last_error}. "
          "The platform DELETE deadline remains the backstop. Inspect or retry: " + shlex.join(command), flush=True)
    return False


def wait_for_completion(name, output_uri, *, project, zone, deadline):
    """Monitoring errors are unknown state, not permission to end a healthy job."""
    last_warning = None

    def warn(message):
        nonlocal last_warning
        if message != last_warning:
            print("Monitoring will retry until the existing deadline: " + message, flush=True)
            last_warning = message

    while time.monotonic() < deadline:
        try:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            result = subprocess.run(["gcloud", "storage", "cat", output_uri + "/completion.json",
                                     "--project=" + project, "--quiet"], text=True, capture_output=True,
                                    timeout=min(60, remaining))
            if result.returncode == 0:
                completed = json.loads(result.stdout)
                if not isinstance(completed, dict):
                    raise ValueError("Completion record is not an object")
                print(json.dumps(completed, indent=2), flush=True)
                return completed
            # Missing completion is normal while work runs. Other failures are
            # logged, but do not imply that the VM or its work has failed.
            if not confirmed_not_found(result.stderr) and "not found" not in result.stderr.lower():
                warn("Completion read: " + (result.stderr.strip() or "command failed"))
        except (OSError, subprocess.SubprocessError, ValueError) as error:
            warn("Completion read: " + str(error))
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break
        try:
            instance = gcloud("compute", "instances", "describe", name, "--zone=" + zone,
                              project=project, optional=True, timeout=min(120, remaining))
            if instance is None:
                print("VM absence confirmed; inspect uploaded checkpoints and logs.", flush=True)
                return None
            if instance.get("status") == "TERMINATED":
                print("VM stopped unexpectedly; inspect uploaded checkpoints and logs.", flush=True)
                return None
        except (OSError, subprocess.SubprocessError, RuntimeError, ValueError) as error:
            warn("VM status read: " + str(error))
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(15, remaining))
    print("Monitoring deadline reached; cleaning up this run's VM.", flush=True)
    return None


def create_vm(command, *, name, project, zone):
    """Handle ambiguous create failures without deleting unrelated resources."""
    try:
        subprocess.run(command, check=True, timeout=300)
    except BaseException:
        try:
            instance = gcloud("compute", "instances", "describe", name,
                              "--zone=" + zone, project=project, optional=True)
            items = (instance or {}).get("metadata", {}).get("items", [])
            config = next((item["value"] for item in items if item["key"] == "ghq-config"), "{}")
            owned = json.loads(config).get("run_id") == name
            owned &= (instance or {}).get("labels", {}).get("managed-by") == "ghq-training"
            if owned:
                delete_vm(name, project=project, zone=zone)
            else:
                print("Create did not complete; no matching owned VM was confirmed for cleanup.", flush=True)
        except Exception as cleanup_error:
            print(f"Could not confirm or clean up create result: {cleanup_error}. "
                  "The requested GCE DELETE deadline remains the backstop if the VM was created.", flush=True)
        raise


def preflight(args):
    project = args.project
    details = gcloud("projects", "describe", project, project=project)
    if details.get("lifecycleState") != "ACTIVE":
        raise RuntimeError("Project is not ACTIVE")
    billing = gcloud("billing", "projects", "describe", project, project=project)
    if not billing.get("billingEnabled"):
        raise RuntimeError("Project billing is not enabled")
    services = gcloud("services", "list", "--enabled", project=project)
    enabled = {item.get("config", {}).get("name") for item in services}
    missing = {"compute.googleapis.com", "storage.googleapis.com", "iam.googleapis.com"} - enabled
    if missing:
        raise RuntimeError("Required APIs are not enabled: " + ", ".join(sorted(missing)))
    if args.service_account.endswith("-compute@developer.gserviceaccount.com"):
        raise RuntimeError("Use a dedicated training identity, not the default Compute identity")
    account = gcloud("iam", "service-accounts", "describe", args.service_account, project=project)
    if account.get("disabled"):
        raise RuntimeError("Training identity is disabled")
    policy = gcloud("projects", "get-iam-policy", project, project=project)
    principal = "serviceAccount:" + args.service_account
    roles = [b["role"] for b in policy.get("bindings", []) if principal in b.get("members", [])]
    if roles:
        raise RuntimeError("Training identity must have bucket-only permissions; project roles: " + str(roles))
    bucket_uri = "gs://" + args.bucket
    gcloud("storage", "buckets", "describe", bucket_uri, project=project)
    bucket_policy = gcloud("storage", "buckets", "get-iam-policy", bucket_uri, project=project)
    can_write = any(b["role"] == "roles/storage.objectUser" and principal in b.get("members", [])
                    and not b.get("condition") for b in bucket_policy.get("bindings", []))
    if not can_write:
        raise RuntimeError("Dedicated job bucket needs an explicit roles/storage.objectUser binding for the training identity")
    if not args.image_uri.startswith(bucket_uri + "/"):
        raise RuntimeError("Image archive must be in the dedicated job bucket")
    archive = gcloud("storage", "objects", "describe", args.image_uri, project=project)
    if args.resume_bundle_uri:
        if not args.resume_bundle_uri.startswith(bucket_uri + "/"):
            raise RuntimeError("Resume bundle must be in the dedicated job bucket")
        resume = gcloud("storage", "objects", "describe", args.resume_bundle_uri, project=project)
        if int(resume.get("size", 0)) > MAX_RESUME_BYTES:
            raise RuntimeError("Resume bundle archive exceeds 4 GiB")
    if args.runtime_bundle_uri:
        if not args.runtime_bundle_uri.startswith(bucket_uri + "/"):
            raise RuntimeError("Runtime bundle must be in the dedicated job bucket")
        runtime = gcloud("storage", "objects", "describe", args.runtime_bundle_uri, project=project)
        if int(runtime.get("size", 0)) > MAX_RUNTIME_BYTES:
            raise RuntimeError("Runtime bundle archive exceeds 256 MiB")
    if args.source_bundle_uri:
        if not args.source_bundle_uri.startswith(bucket_uri + "/"):
            raise RuntimeError("Source bundle must be in the dedicated job bucket")
        source = gcloud("storage", "objects", "describe", args.source_bundle_uri, project=project)
        if int(source.get("size", 0)) > MAX_SOURCE_BYTES:
            raise RuntimeError("Source bundle exceeds 64 MiB")
    subnet = gcloud("compute", "networks", "subnets", "describe", args.subnet,
                    "--region=" + args.zone.rsplit("-", 1)[0], project=project)
    if not subnet.get("privateIpGoogleAccess"):
        raise RuntimeError("Subnet needs Private Google Access for GCS without a public IP")
    gpu = GPU_CONFIG[args.gpu]
    machine = gcloud("compute", "machine-types", "describe", machine_type(args.gpu, args.cpus),
                     "--zone=" + args.zone, project=project)
    accelerators = gcloud("compute", "accelerator-types", "list", "--filter=zone:" + args.zone,
                          project=project)
    if not any(item.get("name") == gpu["accelerator_type"] for item in accelerators):
        raise RuntimeError("Selected zone does not advertise an NVIDIA " + args.gpu.upper())
    region = gcloud("compute", "regions", "describe", args.zone.rsplit("-", 1)[0], project=project)
    global_quota = gcloud("compute", "project-info", "describe", project=project)
    quotas = region.get("quotas", []) + global_quota.get("quotas", [])
    relevant = [q for q in quotas if "GPU" in q["metric"] or "CPUS" in q["metric"]]
    global_status = check_gpu_quotas(region.get("quotas", []), global_quota.get("quotas", []), args.gpu, args.cpus)
    image = gcloud("compute", "images", "describe-from-family", args.vm_image_family,
                   project=IMAGE_PROJECT)
    if image.get("status") != "READY":
        raise RuntimeError("GPU VM image is not READY")
    return {"project": project, "zone": args.zone, "machine_type": machine["name"],
            "gpu": "one NVIDIA " + args.gpu.upper(), "accelerator_type": gpu["accelerator_type"],
            "vm_image": image["name"], "image_project": IMAGE_PROJECT,
            "bucket": bucket_uri, "archive_generation": archive.get("generation"),
            "service_account": args.service_account, "subnet": subnet["selfLink"],
            "quota_snapshot": relevant, "max_runtime_minutes": args.max_minutes,
            "global_gpu_quota": global_status, "resume_bundle_uri": args.resume_bundle_uri,
            "runtime_bundle_uri": args.runtime_bundle_uri,
            "source_bundle_uri": args.source_bundle_uri,
            "termination_action": "DELETE", "public_ip": False,
            "note": "Quota is not a capacity reservation. Storage charges continue for retained GCS outputs."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="play-ghq")
    parser.add_argument("--zone", default="us-central1-a")
    parser.add_argument("--gpu", choices=tuple(GPU_CONFIG), default="l4", help="Explicit GPU selection; no automatic fallback")
    parser.add_argument("--cpus", type=int, choices=(4, 8, 16), default=4,
                        help="T4 worker CPU count; L4 currently supports 4 only")
    parser.add_argument("--bucket", required=True, help="Existing dedicated bucket name, without gs://")
    parser.add_argument("--subnet", required=True, help="Existing subnet with Private Google Access")
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--image-uri", required=True, help="Existing gs://.../image.tar.gz")
    parser.add_argument("--image-sha256", required=True, help="SHA256 of the compressed archive")
    parser.add_argument("--image-id", required=True, help="Immutable Docker image ID, sha256:...")
    parser.add_argument("--vm-image-family", default=IMAGE_FAMILY)
    parser.add_argument("--max-minutes", type=int, default=60)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--teacher-games", type=int, default=4)
    parser.add_argument("--teacher-ms", type=float, default=600)
    parser.add_argument("--seed", type=int, default=20261120)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--resume-bundle-uri", help="Trusted ZIP containing checkpoint.pt and its exact replay buffer")
    parser.add_argument("--resume-bundle-sha256", help="SHA256 of the trusted resume ZIP")
    parser.add_argument("--runtime-bundle-uri", help="Trusted offline Docker/NVIDIA runtime ZIP")
    parser.add_argument("--runtime-bundle-sha256", help="SHA256 of the trusted runtime ZIP")
    parser.add_argument("--source-bundle-uri", help="Reviewed training source ZIP mounted read-only over /app")
    parser.add_argument("--source-bundle-sha256", help="SHA256 of the reviewed source ZIP")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--pipeline-config", type=Path, help="Reviewed JSON pipeline options; replaces legacy training flags")
    mode.add_argument("--collector-benchmark-config", type=Path,
                      help="Reviewed JSON collector benchmark options; requires resume bundle and never trains")
    mode.add_argument("--replay-ablation-config", type=Path,
                      help="Reviewed JSON offline replay ablation; requires resume bundle and never collects games")
    parser.add_argument("--launch", action="store_true", help="Create VM, wait, then delete it")
    args = parser.parse_args()
    try:
        machine_type(args.gpu, args.cpus)
        custom_pipeline = load_pipeline_config(args.pipeline_config) if args.pipeline_config else None
        collector_benchmark = (load_collector_benchmark_config(args.collector_benchmark_config)
                               if args.collector_benchmark_config else None)
        replay_ablation = (load_replay_ablation_config(args.replay_ablation_config)
                           if args.replay_ablation_config else None)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    if not 15 <= args.max_minutes <= 240:
        parser.error("Pilot runtime must be between 15 and 240 minutes")
    if not re.fullmatch(r"[0-9a-f]{64}", args.image_sha256):
        parser.error("--image-sha256 must contain 64 lowercase hexadecimal characters")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_id):
        parser.error("--image-id must be an immutable sha256 Docker image ID")
    if bool(args.resume_bundle_uri) != bool(args.resume_bundle_sha256):
        parser.error("--resume-bundle-uri and --resume-bundle-sha256 must be provided together")
    if args.resume_bundle_sha256 and not re.fullmatch(r"[0-9a-f]{64}", args.resume_bundle_sha256):
        parser.error("--resume-bundle-sha256 must contain 64 lowercase hexadecimal characters")
    if collector_benchmark:
        if not args.resume_bundle_uri:
            parser.error("--collector-benchmark-config requires a verified --resume-bundle-uri")
        if collector_benchmark[1]["max-seconds"] > args.max_minutes * 60 - 600:
            parser.error("Collector benchmark max-seconds must leave at least 600 seconds of VM runtime for setup and upload")
    if replay_ablation:
        if not args.resume_bundle_uri:
            parser.error("--replay-ablation-config requires a verified --resume-bundle-uri")
        if replay_ablation[1]["max-seconds"] > args.max_minutes * 60 - 600:
            parser.error("Replay ablation max-seconds must leave at least 600 seconds of VM runtime for setup and upload")
    if bool(args.runtime_bundle_uri) != bool(args.runtime_bundle_sha256):
        parser.error("--runtime-bundle-uri and --runtime-bundle-sha256 must be provided together")
    if args.runtime_bundle_sha256 and not re.fullmatch(r"[0-9a-f]{64}", args.runtime_bundle_sha256):
        parser.error("--runtime-bundle-sha256 must contain 64 lowercase hexadecimal characters")
    if bool(args.source_bundle_uri) != bool(args.source_bundle_sha256):
        parser.error("--source-bundle-uri and --source-bundle-sha256 must be provided together")
    if args.source_bundle_sha256 and not re.fullmatch(r"[0-9a-f]{64}", args.source_bundle_sha256):
        parser.error("--source-bundle-sha256 must contain 64 lowercase hexadecimal characters")
    if any(getattr(args, name) < 1 for name in ("iterations", "games", "simulations", "max_turns", "epochs", "threads", "batch_size")):
        parser.error("Training counts must be positive")
    if args.teacher_games < 0 or not 0 < args.teacher_ms < float("inf"):
        parser.error("Teacher games must be nonnegative and teacher time finite and positive")
    if not 0 <= args.seed < 2 ** 32:
        parser.error("--seed must be between 0 and 2**32-1")
    plan = preflight(args)
    if custom_pipeline:
        plan["pipeline_options"] = custom_pipeline[1]
        plan["pipeline_config_sha256"] = custom_pipeline[2]
    if collector_benchmark:
        plan["job_mode"] = "collector-benchmark"
        plan["collector_benchmark_options"] = collector_benchmark[1]
        plan["collector_benchmark_config_sha256"] = collector_benchmark[2]
    if replay_ablation:
        plan["job_mode"] = "replay-ablation"
        plan["replay_ablation_options"] = replay_ablation[1]
        plan["replay_ablation_config_sha256"] = replay_ablation[2]
    prefix = "ghq-replay-ablate-" if replay_ablation else "ghq-collect-bench-" if collector_benchmark else "ghq-train-"
    name = prefix + time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    output_uri = f"gs://{args.bucket}/runs/{name}"
    plan["instance_name"] = name
    plan["output_uri"] = output_uri
    print(json.dumps(plan, indent=2), flush=True)
    if not args.launch:
        print("READ-ONLY PREFLIGHT COMPLETE. No resources changed. Add --launch to execute this configuration.")
        return 0
    pipeline_args = []
    for field in ("iterations", "games", "simulations", "max_turns", "epochs", "teacher_games",
                  "teacher_ms", "seed", "threads", "batch_size"):
        pipeline_args.extend(["--" + field.replace("_", "-"), str(getattr(args, field))])
    if custom_pipeline:
        pipeline_args = custom_pipeline[0]
    if collector_benchmark or replay_ablation:
        pipeline_args = []
    config = {"run_id": name, "image_uri": args.image_uri,
              "image_sha256": args.image_sha256, "image_id": args.image_id,
              "output_uri": output_uri, "pipeline_args": pipeline_args,
              "threads": args.threads, "resume_bundle_uri": args.resume_bundle_uri,
              "resume_bundle_sha256": args.resume_bundle_sha256,
              "runtime_bundle_uri": args.runtime_bundle_uri,
              "runtime_bundle_sha256": args.runtime_bundle_sha256,
              "source_bundle_uri": args.source_bundle_uri,
              "source_bundle_sha256": args.source_bundle_sha256,
              "pipeline_config_sha256": custom_pipeline[2] if custom_pipeline else None}
    if custom_pipeline:
        config["threads"] = custom_pipeline[1].get("threads", args.threads)
    if collector_benchmark:
        config.update(job_mode="collector-benchmark", collector_benchmark_args=collector_benchmark[0],
                      collector_benchmark_config_sha256=collector_benchmark[2],
                      collector_benchmark_max_seconds=collector_benchmark[1]["max-seconds"],
                      threads=collector_benchmark[1].get("threads", args.threads))
    if replay_ablation:
        config.update(job_mode="replay-ablation", replay_ablation_args=replay_ablation[0],
                      replay_ablation_config_sha256=replay_ablation[2],
                      replay_ablation_max_seconds=replay_ablation[1]["max-seconds"],
                      threads=replay_ablation[1].get("threads", args.threads))
    with tempfile.TemporaryDirectory(prefix="ghq-gcp-") as temporary:
        config_path = Path(temporary) / "config.json"
        config_path.write_text(json.dumps(config))
        create = ["gcloud", "compute", "instances", "create", name,
                  "--project=" + args.project, "--zone=" + args.zone,
                  "--machine-type=" + machine_type(args.gpu, args.cpus), "--image-project=" + IMAGE_PROJECT,
                  "--image=" + plan["vm_image"], "--boot-disk-size=100GB",
                  "--boot-disk-type=pd-balanced", "--boot-disk-auto-delete",
                  "--subnet=" + args.subnet, "--no-address", "--maintenance-policy=TERMINATE",
                  "--no-restart-on-failure", "--provisioning-model=STANDARD",
                  f"--max-run-duration={args.max_minutes}m", "--instance-termination-action=DELETE",
                  "--service-account=" + args.service_account, "--scopes=storage-rw",
                  "--labels=application=ghq,purpose=neural-pilot,managed-by=ghq-training",
                  "--metadata=enable-oslogin=TRUE,block-project-ssh-keys=TRUE,serial-port-enable=FALSE",
                  "--metadata-from-file=" + ",".join([
                      "startup-script=" + str(DIRECTORY / "startup.sh"),
                      "shutdown-script=" + str(DIRECTORY / "shutdown.sh"),
                      "ghq-runner=" + str(DIRECTORY / "worker.py"), "ghq-config=" + str(config_path)]),
                  "--quiet"]
        if args.gpu == "t4":
            create.append("--accelerator=type=nvidia-tesla-t4,count=1")
        print(shlex.join(create), flush=True)
        create_vm(create, name=name, project=args.project, zone=args.zone)
    deadline = time.monotonic() + args.max_minutes * 60 + 300
    completed = None
    cleaned = False
    try:
        completed = wait_for_completion(name, output_uri, project=args.project,
                                        zone=args.zone, deadline=deadline)
    finally:
        # This unique instance was created above. Never delete pre-existing VMs.
        cleaned = delete_vm(name, project=args.project, zone=args.zone)
    return 0 if cleaned and completed and completed.get("exit_code") == 0 and completed.get("final_sync_succeeded") else 1


if __name__ == "__main__":
    raise SystemExit(main())
