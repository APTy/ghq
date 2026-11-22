"""Download pinned official Docker/NVIDIA packages and build an offline ZIP.

Run on an internet-connected workstation. This utility neither installs the
packages nor uploads anything. SHA256 pins were checked against Docker's
published .sha256 file and NVIDIA's Packages index on 2026-11-21. Updating a
version requires reviewing new vendor checksums and the VM worker's allowlist.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import stat
import urllib.request
import zipfile

DOCKER_VERSION = "29.10.0"
NVIDIA_VERSION = "1.20.2-1"
DOCKER_URL = f"https://download.docker.com/linux/static/stable/x86_64/docker-{DOCKER_VERSION}.tgz"
NVIDIA_BASE = "https://nvidia.github.io/libnvidia-container/stable/deb/amd64/"
NVIDIA_PACKAGES_URL = NVIDIA_BASE + "Packages"
PAYLOADS = (
    {"name": "docker.tgz", "url": DOCKER_URL,
     "sha256": "7c1ebadc65c716cdc6d35fc78759ed0e4317debf43c21dd12f066f2006999def", "bytes": 78755098},
    {"name": "libnvidia-container-tools_1.20.2-1_amd64.deb", "package": "libnvidia-container-tools",
     "sha256": "a0ded4571083019155661eb53b5fd967d9b83bf7c68516d7e6f489c80caf6946", "bytes": 20808},
    {"name": "libnvidia-container1_1.20.2-1_amd64.deb", "package": "libnvidia-container1",
     "sha256": "466e48e38325bd72ebda9d91653c36f44e71ef9bc261f949689132c892835a4b", "bytes": 1237564},
    {"name": "nvidia-container-toolkit-base_1.20.2-1_amd64.deb", "package": "nvidia-container-toolkit-base",
     "sha256": "a6bd7f3a94b0b53b9748872709e70e810473081de0e836a59405073012c49368", "bytes": 6002004},
    {"name": "nvidia-container-toolkit_1.20.2-1_amd64.deb", "package": "nvidia-container-toolkit",
     "sha256": "3ea74794dc7fd88526da6fece9fb142e976efac232892fa4c870a96a9708316b", "bytes": 1531284},
)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(payload, destination):
    item = dict(payload)
    if "package" in item:
        item.update(url=NVIDIA_BASE + item["name"], version=NVIDIA_VERSION)
    path = destination / item["name"]
    digest, size = hashlib.sha256(), 0
    with urllib.request.urlopen(item["url"], timeout=180) as source, path.open("xb") as target:
        while chunk := source.read(1024 * 1024):
            size += len(chunk)
            if size > item["bytes"]:
                raise RuntimeError(f"Download exceeds pinned size: {item['name']}")
            target.write(chunk)
            digest.update(chunk)
    if size != item["bytes"] or digest.hexdigest() != item["sha256"]:
        raise RuntimeError(f"Vendor-pinned SHA256/size mismatch: {item['name']}")
    print(json.dumps({"downloaded": item["name"], "bytes": size, "sha256": item["sha256"]}), flush=True)
    return item


def build(output):
    output = Path(output).resolve()
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be a new or empty directory; existing bundles/downloads are never overwritten")
    output.mkdir(parents=True, exist_ok=True)
    downloads = output / "downloads"
    downloads.mkdir()
    with ThreadPoolExecutor(max_workers=5) as pool:
        files = list(pool.map(lambda item: download(item, downloads), PAYLOADS))
    manifest = {"docker_version": DOCKER_VERSION,
                "nvidia_container_toolkit_version": NVIDIA_VERSION,
                "docker_checksum_source": DOCKER_URL + ".sha256",
                "nvidia_packages_manifest": NVIDIA_PACKAGES_URL,
                "files": files}
    manifest_path = downloads / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    archive = output / "runtime.zip"
    # Fixed member ordering/timestamps/permissions make identical payloads
    # produce identical ZIP bytes on subsequent builds in fresh directories.
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_STORED) as bundle:
        for name in sorted([item["name"] for item in files] + ["manifest.json"]):
            info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
            info.create_system = 3
            info.external_attr = (stat.S_IFREG | 0o644) << 16
            with (downloads / name).open("rb") as source, bundle.open(info, "w") as target:
                shutil.copyfileobj(source, target, length=1024 * 1024)
    artifact = {"path": str(archive), "sha256": sha256(archive), "bytes": archive.stat().st_size,
                "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path),
                "docker_version": DOCKER_VERSION, "nvidia_container_toolkit_version": NVIDIA_VERSION}
    with (output / "artifact.json").open("x") as handle:
        handle.write(json.dumps(artifact, indent=2) + "\n")
    return artifact


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="New/empty directory for downloads/, runtime.zip, and artifact.json")
    args = parser.parse_args()
    try:
        artifact = build(args.output)
    except ValueError as error:
        parser.error(str(error))
    print(json.dumps(artifact, indent=2), flush=True)


if __name__ == "__main__":
    main()
