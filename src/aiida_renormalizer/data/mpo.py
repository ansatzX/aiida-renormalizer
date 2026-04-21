"""MPOData node for Renormalizer MPO persistence."""
from __future__ import annotations

import tempfile
import typing as t

from aiida.orm import Data

if t.TYPE_CHECKING:
    from renormalizer.mps import Mpo

    from aiida_renormalizer.data.model import ModelData

from aiida_renormalizer.data.artifacts import (
    build_artifact_manifest,
    resolve_artifact_path,
    write_external_artifact,
)
from aiida_renormalizer.data.utils import get_linked_node, write_json_to_repository


class MPOData(Data):
    """AiiDA Data node storing a Renormalizer Mpo."""

    @classmethod
    def from_mpo(
        cls,
        mpo: Mpo,
        model_data: ModelData,
        *,
        storage_backend: str = "posix",
        storage_base: str | None = None,
        relative_path: str | None = None,
    ) -> MPOData:
        """Create an MPOData node from a Reno Mpo."""
        node = cls()
        if storage_base is None:
            storage_base = str(tempfile.gettempdir())
        if relative_path is None:
            relative_path = f"aiida-renormalizer/mpo/{node.uuid}.npz"

        node.base.attributes.set("n_sites", len(mpo))
        node.base.attributes.set("bond_dims", [int(d) for d in mpo.bond_dims])
        node.base.attributes.set("dtype", str(mpo.dtype))
        node.base.attributes.set("model_data_uuid", str(model_data.uuid))

        artifact = write_external_artifact(
            mpo,
            storage_backend=storage_backend,
            storage_base=storage_base,
            relative_path=relative_path,
            artifact_format="mpo.npz",
        )
        for key, value in artifact.items():
            node.base.attributes.set(key, value)

        manifest = build_artifact_manifest(
            node_uuid=str(node.uuid),
            artifact=artifact,
            summary={
                "n_sites": node.base.attributes.get("n_sites"),
                "bond_dims": node.base.attributes.get("bond_dims"),
            },
        )
        write_json_to_repository(node, manifest, "artifact_manifest.json")

        return node

    @property
    def model_data(self) -> ModelData:
        """Retrieve the linked ModelData node."""
        return get_linked_node(self.base.attributes.get("model_data_uuid"))

    @property
    def artifact_metadata(self) -> dict[str, t.Any]:
        """Return current artifact metadata, allowing mutable overrides via extras."""
        metadata = {
            "storage_backend": self.base.attributes.get("storage_backend"),
            "storage_base": self.base.attributes.get("storage_base"),
            "relative_path": self.base.attributes.get("relative_path"),
            "artifact_format": self.base.attributes.get("artifact_format"),
            "artifact_size": self.base.attributes.get("artifact_size"),
            "content_hash": self.base.attributes.get("content_hash"),
        }
        for key in ("storage_backend", "storage_base", "relative_path"):
            extra_key = f"artifact_{key}"
            if extra_key in self.base.extras.all:
                metadata[key] = self.base.extras.get(extra_key)
        return metadata

    def load_mpo(self, model_data: ModelData | None = None) -> Mpo:
        """Reconstruct the Mpo from stored data."""
        if model_data is None:
            model_data = self.model_data
        model = model_data.load_model()

        from renormalizer.mps import Mpo

        artifact_path = resolve_artifact_path(
            self.artifact_metadata["storage_backend"],
            self.artifact_metadata["storage_base"],
            self.artifact_metadata["relative_path"],
        )
        if not artifact_path.exists():
            raise FileNotFoundError(f"MPO artifact not found: {artifact_path}")

        return Mpo.load(model, str(artifact_path))

    def relink_artifact(self, storage_base: str, relative_path: str) -> None:
        """Update the logical artifact location."""
        self.base.extras.set("artifact_storage_base", storage_base)
        self.base.extras.set("artifact_relative_path", relative_path)
