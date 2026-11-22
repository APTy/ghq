#!/usr/bin/env python3
"""Package reviewed training sources for a reusable, pinned PyTorch image.

Only ghqnet/*.py, the two authoritative rules/search files, and reviewed
evaluation exclusion manifests are included.
This never installs dependencies, uploads data, or includes experiment outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path
import stat
import zipfile


def build(root, output):
    root, output = Path(root).resolve(), Path(output)
    files = (sorted((root / "ghqnet").glob("*.py")) +
             [root / "public/engine.py", root / "public/search.py"] +
             sorted((root / "training/benchmarks").glob("*.manifest.json")))
    if not (root / "ghqnet/pipeline.py").is_file():
        raise ValueError("Repository root must contain ghqnet/pipeline.py")
    if output.exists():
        raise ValueError("Source bundle already exists; choose a unique output path")
    payload = {}
    for path in files:
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Source must be a regular file: {path}")
        payload[path.relative_to(root).as_posix()] = path.read_bytes()
    manifest = {"version": 1, "files": [{"name": name, "bytes": len(data),
                "sha256": hashlib.sha256(data).hexdigest()} for name, data in sorted(payload.items())]}
    payload["source-manifest.json"] = (json.dumps(manifest, indent=2) + "\n").encode()
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "x", compression=zipfile.ZIP_DEFLATED) as bundle:
        for name, data in sorted(payload.items()):
            entry = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
            entry.compress_type = zipfile.ZIP_DEFLATED
            entry.external_attr = (stat.S_IFREG | 0o644) << 16
            bundle.writestr(entry, data)
    return {"path": str(output.resolve()), "bytes": output.stat().st_size,
            "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "manifest": manifest}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root, args.output), indent=2))


if __name__ == "__main__":
    main()
