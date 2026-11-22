#!/usr/bin/env python3
"""Build a portable, hash-verified resume ZIP from a trusted checkpoint.

Example: python training/gcp/build_resume.py --checkpoint RUN/checkpoint-0001.pt
         --output resume.zip

The ZIP contains only checkpoint.pt and its exact referenced companion files.
The JSON manifest is written beside the ZIP, not inside it. This command never
uploads files, changes a checkpoint, or starts training. Only use trusted local
checkpoints; tensor/basic-data loading uses torch.load(weights_only=True).
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import zipfile

MAX_BYTES = 4 * 1024 ** 3
MAX_FILES = 32
CHUNK_BYTES = 8 * 1024 ** 2
REFERENCE_PATTERNS = {
    "replay_buffer": r"replay-buffer-[0-9]{4,}\.jsonl",
    "validation_buffer": r"validation-buffer-[0-9]{4,}\.jsonl",
    "curriculum_state": r"curriculum-[0-9]{4,}\.json",
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def regular_size(path):
    try:
        info = Path(path).lstat()
    except FileNotFoundError as error:
        raise ValueError(f"Required resume file is missing: {path}") from error
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"Resume input must be a regular file, not a symlink/directory: {path}")
    return info.st_size


def _digest(value, label):
    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
        raise ValueError(f"Missing or invalid SHA256 for {label}")
    return value


def collect_files(checkpoint, *, max_bytes=MAX_BYTES):
    """Resolve only named references; never glob neighbouring experiment files."""
    import torch

    checkpoint = Path(checkpoint).absolute()
    size = regular_size(checkpoint)
    if size > max_bytes:
        raise ValueError("Checkpoint exceeds the 4 GiB resume limit")
    digest = sha256(checkpoint)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or not isinstance(payload.get("metadata"), dict):
        raise ValueError("Checkpoint lacks resume metadata")
    metadata = payload["metadata"]
    files = {"checkpoint.pt": {"path": checkpoint, "bytes": size, "sha256": digest, "role": "checkpoint"}}

    def add(name, expected, role, pattern):
        if not isinstance(name, str) or not re.fullmatch(pattern, name):
            raise ValueError(f"Invalid flat filename for {role}: {name!r}")
        expected = _digest(expected, role)
        path = checkpoint.parent / name
        length = regular_size(path)
        previous = files.get(name)
        if previous is not None:
            if previous["sha256"] != expected:
                raise ValueError(f"Conflicting hashes for resume file {name}")
            return path
        if sum(item["bytes"] for item in files.values()) + length > max_bytes:
            raise ValueError("Uncompressed resume bundle exceeds 4 GiB")
        if sha256(path) != expected:
            raise ValueError(f"SHA256 mismatch for {name}")
        files[name] = {"path": path, "bytes": length, "sha256": expected, "role": role}
        if len(files) > MAX_FILES:
            raise ValueError("Resume bundle exceeds the worker's 32-file limit")
        return path

    def reference(key):
        return add(metadata.get(key), metadata.get(key + "_sha256"), key, REFERENCE_PATTERNS[key])

    reference("replay_buffer")
    curriculum_fields = ("validation_buffer", "validation_buffer_sha256", "curriculum_state", "curriculum_state_sha256")
    run_config = metadata.get("run_config", {})
    is_curriculum = any(key in metadata for key in curriculum_fields) or (
        isinstance(run_config, dict) and run_config.get("curriculum") is True)
    if is_curriculum:
        if not all(key in metadata for key in curriculum_fields):
            raise ValueError("Curriculum resume requires exact validation and curriculum-state references with hashes")
        reference("validation_buffer")
        curriculum_path = reference("curriculum_state")
        curriculum = json.loads(curriculum_path.read_text())
        if not isinstance(curriculum, dict) or not isinstance(curriculum.get("history"), list):
            raise ValueError("Curriculum state lacks a historical-checkpoint list")
        for entry in curriculum["history"]:
            if not isinstance(entry, dict):
                raise ValueError("Invalid historical-checkpoint reference")
            expected = _digest(entry.get("sha256"), "history")
            name = entry.get("file")
            if name != f"history-{expected[:16]}.pt":
                raise ValueError("History filename must match its checkpoint hash prefix")
            add(name, expected, "history", r"history-[0-9a-f]{16}\.pt")
    # Detect a checkpoint changed while its metadata was being inspected.
    if regular_size(checkpoint) != size or sha256(checkpoint) != digest:
        raise ValueError("Checkpoint changed during resume inspection")
    return files, {"kind": "curriculum" if is_curriculum else "legacy", "iteration": metadata.get("iteration"),
                   "seed": metadata.get("seed"), "rules_sha256": payload.get("rules_sha256"),
                   "checkpoint_version": payload.get("checkpoint_version")}


class _BoundedWriter:
    def __init__(self, source, limit):
        self.source, self.limit = source, limit

    def write(self, data):
        if self.source.tell() + len(data) > self.limit:
            raise ValueError("Compressed resume archive exceeds 4 GiB")
        return self.source.write(data)

    def __getattr__(self, name):
        return getattr(self.source, name)


def build(checkpoint, output, manifest_path=None, *, max_bytes=MAX_BYTES):
    """Build deterministic entries; publish without overwriting either output.

    ``max_bytes`` exists for bounded unit tests and may only lower the 4 GiB cap.
    The checkpoint and every companion are re-hashed while writing the archive,
    so changed source files cannot silently produce a mismatched portable set.
    """
    if not isinstance(max_bytes, int) or not 1 <= max_bytes <= MAX_BYTES:
        raise ValueError("Invalid resume size limit")
    output = Path(output).absolute()
    manifest_path = Path(manifest_path).absolute() if manifest_path else Path(str(output) + ".manifest.json")
    if output == manifest_path:
        raise ValueError("Archive and manifest must have different paths")
    for path in (output, manifest_path):
        if path.exists() or path.is_symlink():
            raise ValueError(f"Resume output already exists: {path}")
    files, identity = collect_files(checkpoint, max_bytes=max_bytes)
    output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary, published = [], []
    try:
        with tempfile.NamedTemporaryFile(dir=output.parent, prefix=".resume-", suffix=".zip.tmp", delete=False) as raw:
            archive_temp = Path(raw.name)
            temporary.append(archive_temp)
            with zipfile.ZipFile(_BoundedWriter(raw, max_bytes), "w", compression=zipfile.ZIP_DEFLATED) as bundle:
                for name, item in sorted(files.items()):
                    if regular_size(item["path"]) != item["bytes"]:
                        raise ValueError(f"Resume input changed: {name}")
                    entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                    entry.compress_type = zipfile.ZIP_DEFLATED
                    entry.external_attr = (stat.S_IFREG | 0o644) << 16
                    entry.file_size = item["bytes"]
                    digest, copied = hashlib.sha256(), 0
                    with item["path"].open("rb") as source, bundle.open(entry, "w", force_zip64=True) as target:
                        for chunk in iter(lambda: source.read(CHUNK_BYTES), b""):
                            copied += len(chunk)
                            if copied > item["bytes"]:
                                raise ValueError(f"Resume input grew while packing: {name}")
                            digest.update(chunk)
                            target.write(chunk)
                    if copied != item["bytes"] or digest.hexdigest() != item["sha256"]:
                        raise ValueError(f"Resume input changed while packing: {name}")
        compressed = archive_temp.stat().st_size
        if compressed > max_bytes:
            raise ValueError("Compressed resume archive exceeds 4 GiB")
        manifest = {"version": 1, **identity, "checkpoint_entry": "checkpoint.pt",
                    "archive": str(output), "sha256": sha256(archive_temp), "bytes": compressed,
                    "uncompressed_bytes": sum(item["bytes"] for item in files.values()),
                    "files": [{"name": name, "bytes": item["bytes"], "sha256": item["sha256"], "role": item["role"]}
                              for name, item in sorted(files.items())]}
        with tempfile.NamedTemporaryFile(dir=manifest_path.parent, prefix=".resume-", suffix=".json.tmp", mode="w", delete=False) as target:
            manifest_temp = Path(target.name)
            temporary.append(manifest_temp)
            json.dump(manifest, target, indent=2)
            target.write("\n")
        # link() provides exclusive publication even if another process creates
        # a destination after our initial existence check. No file is replaced.
        for source, destination in ((archive_temp, output), (manifest_temp, manifest_path)):
            os.link(source, destination)
            published.append(destination)
        return {"path": str(output), "sha256": manifest["sha256"], "bytes": compressed,
                "uncompressed_bytes": manifest["uncompressed_bytes"], "files": len(files), "kind": identity["kind"],
                "manifest": str(manifest_path), "manifest_sha256": sha256(manifest_path)}
    except BaseException:
        for path in published:
            path.unlink(missing_ok=True)
        raise
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Trusted local checkpoint with exact resume metadata")
    parser.add_argument("--output", type=Path, required=True, help="New local ZIP path")
    parser.add_argument("--manifest", type=Path, help="New JSON path; defaults to OUTPUT.manifest.json")
    args = parser.parse_args()
    try:
        artifact = build(args.checkpoint, args.output, args.manifest)
    except (ValueError, OSError, RuntimeError) as error:
        parser.exit(1, f"{error}\n")
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
