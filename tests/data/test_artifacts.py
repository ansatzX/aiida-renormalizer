"""Tests for external artifact helpers."""
from __future__ import annotations

import hashlib
from pathlib import Path


class _DummyDumpable:
    """Small stand-in for a Renormalizer object with a dump API."""

    def __init__(self, payload: bytes):
        self.payload = payload

    def dump(self, path: str) -> None:
        Path(f"{path}.npz").write_bytes(self.payload)


def test_resolve_artifact_path_joins_base_and_relative(tmp_path):
    from aiida_renormalizer.data.artifacts import resolve_artifact_path

    resolved = resolve_artifact_path("posix", str(tmp_path), "states/gs.npz")

    assert resolved == tmp_path / "states" / "gs.npz"


def test_write_external_artifact_persists_payload_and_metadata(tmp_path):
    from aiida_renormalizer.data.artifacts import write_external_artifact

    payload = b"wavefunction-bytes"
    artifact = write_external_artifact(
        _DummyDumpable(payload),
        storage_backend="posix",
        storage_base=str(tmp_path),
        relative_path="states/final_state.npz",
        artifact_format="mps.npz",
    )

    artifact_path = tmp_path / "states" / "final_state.npz"
    assert artifact_path.exists()
    assert artifact_path.read_bytes() == payload
    assert artifact["storage_backend"] == "posix"
    assert artifact["storage_base"] == str(tmp_path)
    assert artifact["relative_path"] == "states/final_state.npz"
    assert artifact["artifact_format"] == "mps.npz"
    assert artifact["artifact_size"] == len(payload)
    assert artifact["content_hash"] == hashlib.sha256(payload).hexdigest()


def test_build_artifact_manifest_contains_lightweight_reference_data():
    from aiida_renormalizer.data.artifacts import build_artifact_manifest

    manifest = build_artifact_manifest(
        node_uuid="uuid-123",
        artifact={
            "storage_backend": "posix",
            "storage_base": "/data/archive",
            "relative_path": "states/gs.npz",
            "content_hash": "abc123",
            "artifact_size": 42,
            "artifact_format": "mps.npz",
        },
        summary={"bond_dims": [4, 8, 4], "n_sites": 2},
    )

    assert manifest["node_uuid"] == "uuid-123"
    assert manifest["artifact"]["relative_path"] == "states/gs.npz"
    assert manifest["summary"]["bond_dims"] == [4, 8, 4]
    assert "storage_base" in manifest["artifact"]
