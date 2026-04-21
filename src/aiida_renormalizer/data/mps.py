"""MPSData node for Renormalizer MPS/MpDm persistence."""
from __future__ import annotations

import typing as t

import numpy as np
from aiida.orm import Data

if t.TYPE_CHECKING:
    from renormalizer.mps import MpDm, Mps

    from aiida_renormalizer.data.model import ModelData
    from aiida_renormalizer.data.tensor_network_layout import TensorNetworkLayoutData

from aiida_renormalizer.data.artifacts import (
    build_artifact_manifest,
    resolve_artifact_path,
    write_external_artifact,
)
from aiida_renormalizer.data.utils import get_linked_node, write_json_to_repository


class MPSData(Data):
    """AiiDA Data node storing a Renormalizer Mps or MpDm."""

    @classmethod
    def from_mps(
        cls,
        mps_obj: Mps,
        model_data: ModelData,
        tn_layout_data: TensorNetworkLayoutData | None = None,
        *,
        storage_backend: str,
        storage_base: str,
        relative_path: str,
    ) -> MPSData:
        """Create an MPSData node from a Reno Mps/MpDm."""
        from renormalizer.mps import MpDm

        node = cls()

        # Attributes
        node.base.attributes.set("n_sites", len(mps_obj))
        node.base.attributes.set("bond_dims", [int(d) for d in mps_obj.bond_dims])
        node.base.attributes.set(
            "qntot",
            mps_obj.qntot.tolist() if isinstance(mps_obj.qntot, np.ndarray) else mps_obj.qntot,
        )
        node.base.attributes.set(
            "qnidx", int(mps_obj.qnidx) if mps_obj.qnidx is not None else None
        )
        node.base.attributes.set("dtype", str(mps_obj.dtype))
        node.base.attributes.set("is_mpdm", isinstance(mps_obj, MpDm))
        node.base.attributes.set("model_data_uuid", str(model_data.uuid))
        if tn_layout_data is not None:
            node.base.attributes.set("tn_layout_data_uuid", str(tn_layout_data.uuid))

        artifact = write_external_artifact(
            mps_obj,
            storage_backend=storage_backend,
            storage_base=storage_base,
            relative_path=relative_path,
            artifact_format="mps.npz",
        )
        for key, value in artifact.items():
            node.base.attributes.set(key, value)

        manifest = build_artifact_manifest(
            node_uuid=str(node.uuid),
            artifact=artifact,
            summary={
                "n_sites": node.base.attributes.get("n_sites"),
                "bond_dims": node.base.attributes.get("bond_dims"),
                "is_mpdm": node.base.attributes.get("is_mpdm"),
            },
        )
        write_json_to_repository(node, manifest, "artifact_manifest.json")

        return node

    @property
    def model_data(self) -> ModelData:
        """Retrieve the linked ModelData node."""
        return get_linked_node(self.base.attributes.get("model_data_uuid"))

    @property
    def tn_layout_data(self) -> TensorNetworkLayoutData | None:
        """Retrieve the linked TensorNetworkLayoutData node, if present."""
        uuid = self.base.attributes.get("tn_layout_data_uuid")
        if not uuid:
            return None
        return get_linked_node(uuid)

    @property
    def artifact_metadata(self) -> dict[str, t.Any]:
        """Return the current artifact metadata, allowing mutable overrides via extras."""
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

    def load_mps(self, model_data: ModelData | None = None) -> Mps:
        """Reconstruct the Mps/MpDm from stored data.

        Args:
            model_data: Optional ModelData to use. If None, loads from stored UUID.

        Raises:
            ValueError: If the provided model has a different number of sites
                than the stored MPS.
        """
        if model_data is None:
            model_data = self.model_data
        model = model_data.load_model()

        # Validate model compatibility: number of sites must match
        stored_n_sites = self.base.attributes.get("n_sites")
        if stored_n_sites is not None and model.nsite != stored_n_sites:
            raise ValueError(
                f"Model has {model.nsite} sites but the stored MPS has "
                f"{stored_n_sites} sites. Cannot load with a mismatched model."
            )

        if self.base.attributes.get("is_mpdm"):
            from renormalizer.mps import MpDm

            loader_class = MpDm
        else:
            from renormalizer.mps import Mps

            loader_class = Mps

        artifact_path = resolve_artifact_path(
            self.artifact_metadata["storage_backend"],
            self.artifact_metadata["storage_base"],
            self.artifact_metadata["relative_path"],
        )
        if not artifact_path.exists():
            raise FileNotFoundError(f"MPS artifact not found: {artifact_path}")

        mps_loaded = loader_class.load(model, str(artifact_path))
        return mps_loaded

    def relink_artifact(self, storage_base: str, relative_path: str) -> None:
        """Update the logical artifact location."""
        self.base.extras.set("artifact_storage_base", storage_base)
        self.base.extras.set("artifact_relative_path", relative_path)
