"""Helpers for external wavefunction artifact storage."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import shutil
import tempfile
import typing as t
from pathlib import Path


def resolve_artifact_path(
    storage_backend: str,
    storage_base: str,
    relative_path: str,
) -> Path:
    """Resolve a logical artifact location to a concrete filesystem path."""
    if storage_backend != "posix":
        raise ValueError(f"Unsupported storage backend: {storage_backend}")
    return Path(storage_base).expanduser() / relative_path


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_external_artifact(
    obj: t.Any,
    *,
    storage_backend: str,
    storage_base: str,
    relative_path: str,
    artifact_format: str,
) -> dict[str, t.Any]:
    """Dump an object to external storage and return artifact metadata."""
    artifact_path = resolve_artifact_path(storage_backend, storage_base, relative_path)
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        dump_stem = Path(tmpdir) / artifact_path.stem
        obj.dump(str(dump_stem))
        dump_path = dump_stem.with_suffix(".npz")
        if not dump_path.exists():
            dump_path = dump_stem
        os.replace(dump_path, artifact_path)

    return {
        "storage_backend": storage_backend,
        "storage_base": storage_base,
        "relative_path": relative_path,
        "artifact_format": artifact_format,
        "artifact_size": artifact_path.stat().st_size,
        "content_hash": _hash_file(artifact_path),
    }


def build_artifact_manifest(
    *,
    node_uuid: str,
    artifact: dict[str, t.Any],
    summary: dict[str, t.Any],
) -> dict[str, t.Any]:
    """Build a lightweight manifest for repository-side metadata."""
    return {
        "node_uuid": node_uuid,
        "artifact": dict(artifact),
        "summary": dict(summary),
    }


def export_publication_bundle(
    *,
    node_uuid: str,
    artifact: dict[str, t.Any],
    summary: dict[str, t.Any] | None = None,
    output_dir: str | Path,
) -> dict[str, t.Any]:
    """Copy an artifact into a publication-ready bundle directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    source_path = resolve_artifact_path(
        artifact["storage_backend"],
        artifact["storage_base"],
        artifact["relative_path"],
    )
    if not source_path.exists():
        raise FileNotFoundError(f"Artifact not found: {source_path}")

    summary = dict(summary or {})
    artifact_type = str(artifact.get("artifact_format", "artifact")).split(".", 1)[0]
    suffix = source_path.suffix or ".npz"
    bundle_filename = f"{artifact_type}-{node_uuid[:8]}{suffix}"
    bundle_relative_path = Path("artifacts") / bundle_filename
    bundled_path = output_path / bundle_relative_path
    bundled_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, bundled_path)

    summary_path = output_path / "metadata" / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True))

    exported_at = dt.datetime.now(dt.UTC).isoformat()
    manifest = {
        "node_uuid": node_uuid,
        "exported_at": exported_at,
        "source_artifact": dict(artifact),
        "bundle_relative_path": str(bundle_relative_path),
        "summary": summary,
    }
    (output_path / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    (output_path / "README.md").write_text(
        "\n".join(
            [
                "# Publication Bundle",
                "",
                "This directory contains a publication-oriented export of an aiida-renormalizer artifact.",
                "",
                f"- Node UUID: `{node_uuid}`",
                f"- Exported at: `{exported_at}`",
                f"- Artifact file: `{bundle_relative_path.as_posix()}`",
                "- Manifest: `manifest.json`",
                "- Summary metadata: `metadata/summary.json`",
                "",
                "The large tensor payload is stored under `artifacts/`, while machine-readable metadata is split between the manifest and summary files.",
                "",
            ]
        )
    )
    return manifest
