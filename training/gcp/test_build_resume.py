import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

try:
    import torch
except ImportError:
    raise unittest.SkipTest("Resume packager tests require torch")

from training.gcp import build_resume as builder
from training.gcp import worker


def checkpoint_fixture(root, *, curriculum=True, histories=2):
    root = Path(root)
    replay = root / "replay-buffer-0001.jsonl"
    replay.write_text('{"example": "exact replay bytes"}\n')
    metadata = {"iteration": 1, "seed": 17, "replay_buffer": replay.name,
                "replay_buffer_sha256": builder.sha256(replay)}
    if curriculum:
        validation = root / "validation-buffer-0001.jsonl"
        validation.write_text('{"example": "held-out validation bytes"}\n')
        history = []
        for index in range(histories):
            value = f"historical checkpoint {index}".encode()
            digest = hashlib.sha256(value).hexdigest()
            path = root / f"history-{digest[:16]}.pt"
            path.write_bytes(value)
            history.append({"file": path.name, "sha256": digest, "iteration": index})
        state = root / "curriculum-0001.json"
        state.write_text(json.dumps({"version": 1, "history": history, "partition": {"seed": 17}}))
        metadata.update(validation_buffer=validation.name, validation_buffer_sha256=builder.sha256(validation),
                        curriculum_state=state.name, curriculum_state_sha256=builder.sha256(state),
                        run_config={"curriculum": True})
    checkpoint = root / "checkpoint-0001.pt"
    torch.save({"checkpoint_version": 1, "rules_sha256": "a" * 64,
                "model_state": {"weight": torch.zeros(2)}, "metadata": metadata}, checkpoint)
    return checkpoint, metadata


class ResumeBundleTests(unittest.TestCase):
    def test_curriculum_exact_files_determinism_manifest_and_worker_extraction(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint, metadata = checkpoint_fixture(root)
            (root / "unrelated-secret.json").write_text("must never be globbed")
            original = checkpoint.read_bytes()
            with patch.object(torch, "load", wraps=torch.load) as loader:
                result = builder.build(checkpoint, root / "resume.zip")
            loader.assert_called_once_with(checkpoint.absolute(), map_location="cpu", weights_only=True)
            self.assertEqual(result["files"], 6)
            self.assertEqual(result["kind"], "curriculum")
            manifest = json.loads(Path(result["manifest"]).read_text())
            self.assertEqual(manifest["sha256"], builder.sha256(root / "resume.zip"))
            with zipfile.ZipFile(result["path"]) as archive:
                self.assertEqual(set(archive.namelist()), {entry["name"] for entry in manifest["files"]})
                self.assertEqual(archive.read("checkpoint.pt"), original)
                self.assertNotIn("unrelated-secret.json", archive.namelist())
                self.assertTrue(all(stat.S_IFMT(entry.external_attr >> 16) == stat.S_IFREG for entry in archive.infolist()))
            worker.extract_resume_bundle(Path(result["path"]), root / "extracted")
            restored = torch.load(root / "extracted/checkpoint.pt", map_location="cpu", weights_only=True)
            self.assertEqual(restored["metadata"], metadata)
            second = builder.build(checkpoint, root / "again.zip")
            self.assertEqual(second["sha256"], result["sha256"])
            self.assertEqual(checkpoint.read_bytes(), original)

    def test_legacy_pair_and_no_overwrite_for_archive_or_manifest(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint, _ = checkpoint_fixture(root, curriculum=False)
            result = builder.build(checkpoint, root / "legacy.zip")
            self.assertEqual(result["kind"], "legacy")
            self.assertEqual(result["files"], 2)
            with self.assertRaisesRegex(ValueError, "already exists"):
                builder.build(checkpoint, root / "legacy.zip")
            (root / "taken.json").write_text("keep")
            with self.assertRaisesRegex(ValueError, "already exists"):
                builder.build(checkpoint, root / "new.zip", root / "taken.json")
            self.assertFalse((root / "new.zip").exists())
            self.assertEqual((root / "taken.json").read_text(), "keep")

    def test_missing_corrupt_and_symlink_references_fail_before_publication(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint, _ = checkpoint_fixture(root)
            replay = root / "replay-buffer-0001.jsonl"
            original = replay.read_bytes()
            replay.write_text("corrupted")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                builder.build(checkpoint, root / "bad.zip")
            replay.unlink()
            with self.assertRaisesRegex(ValueError, "missing"):
                builder.build(checkpoint, root / "bad.zip")
            target = root / "elsewhere"; target.write_bytes(original); replay.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "regular file"):
                builder.build(checkpoint, root / "bad.zip")
            self.assertFalse((root / "bad.zip").exists())
            self.assertFalse(list(root.glob(".resume-*")))

    def test_path_traversal_partial_curriculum_and_history_hash_are_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint, _ = checkpoint_fixture(root)
            payload = torch.load(checkpoint, weights_only=True)
            payload["metadata"]["replay_buffer"] = "../replay-buffer-0001.jsonl"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "flat filename"):
                builder.build(checkpoint, root / "bad.zip")
            checkpoint, _ = checkpoint_fixture(root)
            payload = torch.load(checkpoint, weights_only=True)
            del payload["metadata"]["validation_buffer_sha256"]
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "requires exact"):
                builder.build(checkpoint, root / "bad.zip")
            checkpoint, _ = checkpoint_fixture(root)
            history = next(root.glob("history-*.pt")); history.write_bytes(b"wrong history")
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                builder.build(checkpoint, root / "bad.zip")

    def test_both_size_caps_and_worker_file_count_limit_are_enforced(self):
        stream = io.BytesIO()
        limited = builder._BoundedWriter(stream, 4)
        limited.write(b"1234")
        with self.assertRaisesRegex(ValueError, "Compressed"):
            limited.write(b"5")
        self.assertEqual(stream.getvalue(), b"1234")
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            checkpoint, _ = checkpoint_fixture(root, curriculum=False)
            with self.assertRaisesRegex(ValueError, "Uncompressed"):
                builder.build(checkpoint, root / "small.zip", max_bytes=checkpoint.stat().st_size)
            self.assertFalse((root / "small.zip").exists())
            with self.assertRaisesRegex(ValueError, "Invalid resume size limit"):
                builder.build(checkpoint, root / "large.zip", max_bytes=builder.MAX_BYTES + 1)
            checkpoint, _ = checkpoint_fixture(root, histories=29)
            with self.assertRaisesRegex(ValueError, "32-file"):
                builder.build(checkpoint, root / "too-many.zip")


if __name__ == "__main__":
    unittest.main()
