"""TTNSData node for Renormalizer TTNS persistence."""
from __future__ import annotations

import typing as t

import numpy as np
from aiida.orm import Data

if t.TYPE_CHECKING:
    from renormalizer.tn.tree import TTNS

    from aiida_renormalizer.data.basis_tree import BasisTreeData

from aiida_renormalizer.data.artifacts import (
    build_artifact_manifest,
    resolve_artifact_path,
    write_external_artifact,
)
from aiida_renormalizer.data.utils import get_linked_node, write_json_to_repository


class TTNSData(Data):
    """AiiDA Data node storing a Renormalizer TTNS (Tree Tensor Network State)."""

    @classmethod
    def from_ttns(
        cls,
        ttns_obj: TTNS,
        basis_tree_data: BasisTreeData,
        *,
        storage_backend: str,
        storage_base: str,
        relative_path: str,
    ) -> TTNSData:
        """Create a TTNSData node from a Reno TTNS."""
        node = cls()

        # Attributes
        node.base.attributes.set("n_nodes", len(ttns_obj))
        node.base.attributes.set("bond_dims", [int(d) for d in ttns_obj.bond_dims])
        node.base.attributes.set(
            "qntot",
            ttns_obj.qntot.tolist() if isinstance(ttns_obj.qntot, np.ndarray) else ttns_obj.qntot,
        )
        node.base.attributes.set(
            "dtype",
            str(np.result_type(*(nd.tensor.dtype for nd in ttns_obj.node_list))),
        )
        node.base.attributes.set("basis_tree_data_uuid", str(basis_tree_data.uuid))
        node.base.attributes.set("coeff", float(ttns_obj.coeff))

        artifact = write_external_artifact(
            ttns_obj,
            storage_backend=storage_backend,
            storage_base=storage_base,
            relative_path=relative_path,
            artifact_format="ttns.npz",
        )
        for key, value in artifact.items():
            node.base.attributes.set(key, value)

        manifest = build_artifact_manifest(
            node_uuid=str(node.uuid),
            artifact=artifact,
            summary={
                "n_nodes": node.base.attributes.get("n_nodes"),
                "bond_dims": node.base.attributes.get("bond_dims"),
                "coeff": node.base.attributes.get("coeff"),
            },
        )
        write_json_to_repository(node, manifest, "artifact_manifest.json")

        return node

    @property
    def basis_tree_data(self) -> BasisTreeData:
        """Retrieve the linked BasisTreeData node."""
        return get_linked_node(self.base.attributes.get("basis_tree_data_uuid"))

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

    def load_ttns(self, basis_tree_data: BasisTreeData | None = None) -> TTNS:
        """Reconstruct the TTNS from stored data.

        Args:
            basis_tree_data: Optional BasisTreeData to use. If None, loads from stored UUID.

        Returns:
            The Renormalizer TTNS object.

        """
        if basis_tree_data is None:
            basis_tree_data = self.basis_tree_data
        basis_tree = basis_tree_data.load_basis_tree()

        from renormalizer.tn.tree import TTNS

        artifact_path = resolve_artifact_path(
            self.artifact_metadata["storage_backend"],
            self.artifact_metadata["storage_base"],
            self.artifact_metadata["relative_path"],
        )
        if not artifact_path.exists():
            raise FileNotFoundError(f"TTNS artifact not found: {artifact_path}")

        ttns_loaded = TTNS.load(basis_tree, str(artifact_path))

        # Restore coefficient
        ttns_loaded.coeff = self.base.attributes.get("coeff")

        return ttns_loaded

    def relink_artifact(self, storage_base: str, relative_path: str) -> None:
        """Update the logical artifact location."""
        self.base.extras.set("artifact_storage_base", storage_base)
        self.base.extras.set("artifact_relative_path", relative_path)
