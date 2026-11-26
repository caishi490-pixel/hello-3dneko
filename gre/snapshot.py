# SPDX-License-Identifier: Apache-2.0
"""Content-addressed P2 snapshots and version-checked local .blend transactions."""

import hashlib
import json
import os
import re
import shutil
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from uuid import uuid4

import numpy as np


@dataclass(frozen=True)
class MeshSnapshot:
    V: np.ndarray
    vid: np.ndarray
    triangles: np.ndarray
    triangle_loops: np.ndarray
    triangle_fid: np.ndarray
    uv_layers: dict[str, np.ndarray]
    hashes: dict[str, object]
    unit_scale: float = 1.0


def file_hash(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path: Path, data: dict) -> None:
    encoded = json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(encoded + "\n")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@contextmanager
def _lock(path: Path):
    descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(descriptor)
    try:
        yield
    finally:
        path.unlink()


def _snapshot_paths(root: Path, asset: str, digest: str) -> tuple[Path, Path, Path]:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", asset) or not re.fullmatch(r"[a-f0-9]{64}", digest):
        raise ValueError("Invalid asset ID or source SHA-256")
    folder = root / "snapshots" / asset
    return tuple(folder / f"{digest}{suffix}" for suffix in (".npz", ".meta.json", ".blend"))


def save_snapshot(root: Path, asset: str, data: MeshSnapshot, source: Path, expected_source_hash: str) -> dict:
    """Write a snapshot once, then verify immutable contents on subsequent reads."""
    archive, metadata, backup = _snapshot_paths(root, asset, expected_source_hash)
    archive.parent.mkdir(parents=True, exist_ok=True)
    with _lock(archive.with_suffix(".lock")):
        if file_hash(source) != expected_source_hash:
            raise ValueError("Source changed during snapshot capture")
        if metadata.exists():
            saved, prior = load_snapshot(root, asset, expected_source_hash)
            if saved.hashes != data.hashes or saved.V.tobytes() != data.V.tobytes():
                raise ValueError("The source snapshot already records a different mesh")
            return prior
        with TemporaryDirectory(prefix=".gre-snapshot-", dir=archive.parent) as scratch:
            stage = Path(scratch)
            stage_npz, stage_meta, stage_blend = (stage / path.name for path in (archive, metadata, backup))
            np.savez_compressed(stage_npz, V=data.V, vid=data.vid, triangles=data.triangles,
                                triangle_loops=data.triangle_loops, triangle_fid=data.triangle_fid,
                                **{f"uv_{i}": values for i, values in enumerate(data.uv_layers.values())})
            shutil.copyfile(source, stage_blend)
            if file_hash(stage_blend) != expected_source_hash:
                raise ValueError("Source changed while capturing the native .blend")
            meta = {"schema_version": 1, "asset_id": asset, "source_file_sha256": expected_source_hash,
                    "snapshot_npz_sha256": file_hash(stage_npz), "object_hashes": data.hashes,
                    "unit_scale": data.unit_scale, "uv_layers": list(data.uv_layers)}
            write_json(stage_meta, meta)
            published = []
            try:
                for staged, target in ((stage_npz, archive), (stage_blend, backup), (stage_meta, metadata)):
                    os.link(staged, target)
                    published.append(target)
            except OSError:
                for path in published:
                    path.unlink()
                raise
            return meta


def load_snapshot(root: Path, asset: str, digest: str) -> tuple[MeshSnapshot, dict]:
    archive, metadata, backup = _snapshot_paths(root, asset, digest)
    meta = read_json(metadata)
    if (meta.get("schema_version") != 1 or meta.get("asset_id") != asset or meta.get("source_file_sha256") != digest
            or file_hash(archive) != meta.get("snapshot_npz_sha256") or file_hash(backup) != digest):
        raise ValueError("Snapshot metadata or content hash mismatch")
    with np.load(archive, allow_pickle=False) as values:
        data = MeshSnapshot(values["V"], values["vid"], values["triangles"], values["triangle_loops"],
                            values["triangle_fid"], {name: values[f"uv_{i}"] for i, name in enumerate(meta["uv_layers"])},
                            meta["object_hashes"], meta["unit_scale"])
    return data, meta


def _working(root: Path, name: str) -> Path:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*\.blend", name):
        raise ValueError("Working file must be a simple .blend filename")
    folder = root / "working"
    if folder.resolve() != folder:
        raise ValueError("Working directory must stay inside the workspace")
    path = folder / name
    if path.is_symlink():
        raise ValueError("Working file must not be a symlink")
    return path


def commit(root: Path, asset: str, source_hash: str, staged: Path, working_name: str,
           decision: str, review_note: str = "") -> dict:
    note = review_note.strip()
    if decision != "accepted" and not (decision == "needs_review" and note):
        raise ValueError("Commit requires accepted QA or an explicit note for needs_review")
    _, metadata = load_snapshot(root, asset, source_hash)
    root = root.resolve()
    working = _working(root, working_name)
    working.parent.mkdir(parents=True, exist_ok=True)
    audit = root / "audit"
    audit.mkdir(parents=True, exist_ok=True)
    candidate_hash = file_hash(staged)
    if candidate_hash == source_hash:
        raise ValueError("The staged .blend is already identical to the source")
    receipt = {"audit_id": uuid4().hex, "asset_id": asset, "working_file": working_name,
               "snapshot_npz_sha256": metadata["snapshot_npz_sha256"], "source_file_sha256": source_hash,
               "committed_file_sha256": candidate_hash, "qa_decision": decision, "review_note": note or None}
    with _lock(working.with_suffix(".lock")), TemporaryDirectory(prefix=".gre-commit-", dir=working.parent) as scratch:
        temporary, record = Path(scratch) / "working.blend", Path(scratch) / "receipt.json"
        shutil.copyfile(staged, temporary)
        if file_hash(temporary) != candidate_hash:
            raise ValueError("Staged .blend changed before commit")
        write_json(record, receipt)
        os.link(temporary, working)
        try:
            os.link(record, audit / f"{receipt['audit_id']}.json")
        except OSError:
            working.unlink()
            raise
    return receipt


def rollback(root: Path, audit_id: str, expected_current_hash: str) -> dict:
    if not re.fullmatch(r"[a-f0-9]{32}", audit_id):
        raise ValueError("Invalid audit ID")
    root = root.resolve()
    audit = root / "audit"
    receipt = read_json(audit / f"{audit_id}.json")
    if receipt["audit_id"] != audit_id or receipt["committed_file_sha256"] != expected_current_hash:
        raise ValueError("Rollback version mismatch")
    working = _working(root, receipt["working_file"])
    original_hash = receipt["source_file_sha256"]
    _, meta = load_snapshot(root, receipt["asset_id"], original_hash)
    if meta["snapshot_npz_sha256"] != receipt["snapshot_npz_sha256"]:
        raise ValueError("Audit snapshot hash does not match the stored snapshot")
    _, _, backup = _snapshot_paths(root, receipt["asset_id"], original_hash)
    record = {"audit_id": audit_id, "working_file": receipt["working_file"], "previous_file_sha256": expected_current_hash,
              "restored_file_sha256": original_hash}
    with _lock(working.with_suffix(".lock")), TemporaryDirectory(prefix=".gre-rollback-", dir=working.parent) as scratch:
        if file_hash(working) != expected_current_hash:
            raise ValueError("Working .blend changed since commit; rollback refused")
        tmp = Path(scratch)
        staged, old, proof = tmp / "restored.blend", tmp / "committed.blend", tmp / "rollback.json"
        shutil.copyfile(backup, staged)
        if file_hash(staged) != original_hash or file_hash(working) != expected_current_hash:
            raise ValueError("Rollback source or working file changed")
        write_json(proof, record)
        # Retain a link to the committed inode until the restore record is published.
        os.link(working, old)
        os.replace(staged, working)
        try:
            os.link(proof, audit / f"{audit_id}.rollback.json")
        except OSError:
            os.replace(old, working)
            raise
    return record
