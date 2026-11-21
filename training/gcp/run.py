#!/usr/bin/env python3
"""Read-only preflight by default; --launch creates exactly one bounded GPU VM."""
from __future__ import annotations

import argparse
import json
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


def gcloud(*args, project, optional=False):
    command = ["gcloud", *args, "--project=" + project, "--quiet", "--format=json"]
    result = subprocess.run(command, text=True, capture_output=True, timeout=120)
    if result.returncode:
        if optional:
            return None
        raise RuntimeError(result.stderr.strip())
    return json.loads(result.stdout or "null")


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
                subprocess.run(["gcloud", "compute", "instances", "delete", name,
                                "--project=" + project, "--zone=" + zone, "--quiet"],
                               check=False, timeout=120)
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
    subnet = gcloud("compute", "networks", "subnets", "describe", args.subnet,
                    "--region=" + args.zone.rsplit("-", 1)[0], project=project)
    if not subnet.get("privateIpGoogleAccess"):
        raise RuntimeError("Subnet needs Private Google Access for GCS without a public IP")
    machine = gcloud("compute", "machine-types", "describe", "g2-standard-4",
                     "--zone=" + args.zone, project=project)
    accelerators = gcloud("compute", "accelerator-types", "list", "--filter=zone:" + args.zone,
                          project=project)
    if not any(item.get("name") == "nvidia-l4" for item in accelerators):
        raise RuntimeError("Selected zone does not advertise an NVIDIA L4")
    region = gcloud("compute", "regions", "describe", args.zone.rsplit("-", 1)[0], project=project)
    global_quota = gcloud("compute", "project-info", "describe", project=project)
    quotas = region.get("quotas", []) + global_quota.get("quotas", [])
    relevant = [q for q in quotas if "GPU" in q["metric"] or q["metric"] in {"CPUS", "G2_CPUS"}]
    for metric in ("NVIDIA_L4_GPUS", "GPUS_ALL_REGIONS"):
        entries = [q for q in quotas if q["metric"] == metric]
        if not entries or entries[0]["limit"] - entries[0]["usage"] < 1:
            raise RuntimeError("Insufficient or unavailable quota: " + metric)
    image = gcloud("compute", "images", "describe-from-family", args.vm_image_family,
                   project=IMAGE_PROJECT)
    if image.get("status") != "READY":
        raise RuntimeError("GPU VM image is not READY")
    return {"project": project, "zone": args.zone, "machine_type": machine["name"],
            "gpu": "one NVIDIA L4", "vm_image": image["name"], "image_project": IMAGE_PROJECT,
            "bucket": bucket_uri, "archive_generation": archive.get("generation"),
            "service_account": args.service_account, "subnet": subnet["selfLink"],
            "quota_snapshot": relevant, "max_runtime_minutes": args.max_minutes,
            "termination_action": "DELETE", "public_ip": False,
            "note": "Quota is not a capacity reservation. Storage charges continue for retained GCS outputs."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", default="play-ghq")
    parser.add_argument("--zone", default="us-central1-a")
    parser.add_argument("--bucket", required=True, help="Existing dedicated bucket name, without gs://")
    parser.add_argument("--subnet", required=True, help="Existing subnet with Private Google Access")
    parser.add_argument("--service-account", required=True)
    parser.add_argument("--image-uri", required=True, help="Existing gs://.../image.tar.gz")
    parser.add_argument("--image-sha256", required=True, help="SHA256 of the compressed archive")
    parser.add_argument("--image-id", required=True, help="Immutable Docker image ID, sha256:...")
    parser.add_argument("--vm-image-family", default=IMAGE_FAMILY)
    parser.add_argument("--max-minutes", type=int, default=120)
    parser.add_argument("--iterations", type=int, default=2)
    parser.add_argument("--games", type=int, default=16)
    parser.add_argument("--simulations", type=int, default=64)
    parser.add_argument("--max-turns", type=int, default=100)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--launch", action="store_true", help="Create VM, wait, then delete it")
    args = parser.parse_args()
    if not 15 <= args.max_minutes <= 240:
        parser.error("Pilot runtime must be between 15 and 240 minutes")
    if not re.fullmatch(r"[0-9a-f]{64}", args.image_sha256):
        parser.error("--image-sha256 must contain 64 lowercase hexadecimal characters")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", args.image_id):
        parser.error("--image-id must be an immutable sha256 Docker image ID")
    if any(getattr(args, name) < 1 for name in ("iterations", "games", "simulations", "max_turns", "epochs")):
        parser.error("Training counts must be positive")
    plan = preflight(args)
    name = "ghq-train-" + time.strftime("%Y%m%d-%H%M%S", time.gmtime()) + "-" + uuid.uuid4().hex[:6]
    output_uri = f"gs://{args.bucket}/runs/{name}"
    plan["instance_name"] = name
    plan["output_uri"] = output_uri
    print(json.dumps(plan, indent=2), flush=True)
    if not args.launch:
        print("READ-ONLY PREFLIGHT COMPLETE. No resources changed. Add --launch to execute this configuration.")
        return 0
    pipeline_args = []
    for field in ("iterations", "games", "simulations", "max_turns", "epochs"):
        pipeline_args.extend(["--" + field.replace("_", "-"), str(getattr(args, field))])
    config = {"run_id": name, "image_uri": args.image_uri,
              "image_sha256": args.image_sha256, "image_id": args.image_id,
              "output_uri": output_uri, "pipeline_args": pipeline_args}
    with tempfile.TemporaryDirectory(prefix="ghq-gcp-") as temporary:
        config_path = Path(temporary) / "config.json"
        config_path.write_text(json.dumps(config))
        create = ["gcloud", "compute", "instances", "create", name,
                  "--project=" + args.project, "--zone=" + args.zone,
                  "--machine-type=g2-standard-4", "--image-project=" + IMAGE_PROJECT,
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
        print(shlex.join(create), flush=True)
        create_vm(create, name=name, project=args.project, zone=args.zone)
    deadline = time.monotonic() + args.max_minutes * 60 + 300
    completed = None
    try:
        while time.monotonic() < deadline:
            result = subprocess.run(["gcloud", "storage", "cat", output_uri + "/completion.json",
                                     "--project=" + args.project, "--quiet"],
                                    text=True, capture_output=True, timeout=60)
            if result.returncode == 0:
                completed = json.loads(result.stdout)
                print(json.dumps(completed, indent=2), flush=True)
                break
            instance = gcloud("compute", "instances", "describe", name,
                              "--zone=" + args.zone, project=args.project, optional=True)
            if instance is None:
                print("VM no longer exists; inspect uploaded checkpoints and logs.")
                break
            if instance.get("status") == "TERMINATED":
                print("VM stopped unexpectedly; inspect uploaded checkpoints and logs.")
                break
            time.sleep(15)
    finally:
        # This unique instance was created above. Never delete pre-existing VMs.
        subprocess.run(["gcloud", "compute", "instances", "delete", name,
                        "--project=" + args.project, "--zone=" + args.zone, "--quiet"], check=False)
    return 0 if completed and completed.get("exit_code") == 0 and completed.get("final_sync_succeeded") else 1


if __name__ == "__main__":
    raise SystemExit(main())
