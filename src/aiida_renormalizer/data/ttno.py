"""TTNOData node for Renormalizer TTNO persistence."""
from __future__ import annotations

import tempfile
import typing as t

import numpy as np
from aiida.orm import Data

if t.TYPE_CHECKING:
    from renormalizer.tn.tree import TTNO

    from aiida_renormalizer.data.basis_tree import BasisTreeData

from aiida_renormalizer.data.artifacts import (
    build_artifact_manifest,
    resolve_artifact_path,
    write_external_artifact,
)
from aiida_renormalizer.data.utils import get_linked_node, write_json_to_repository


class TTNOData(Data):
    """AiiDA Data node storing a Renormalizer TTNO (Tree Tensor Network Operator)."""

    @classmethod
    def from_ttno(
        cls,
        ttno_obj: TTNO,
        basis_tree_data: BasisTreeData,
        *,
        storage_backend: str = "posix",
        storage_base: str | None = None,
        relative_path: str | None = None,
    ) -> TTNOData:
        """Create a TTNOData node from a Reno TTNO."""
        node = cls()
        if storage_base is None:
            storage_base = str(tempfile.gettempdir())
        if relative_path is None:
            relative_path = f"aiida-renormalizer/ttno/{node.uuid}.npz"

        # Attributes
        node.base.attributes.set("n_nodes", len(ttno_obj))
        node.base.attributes.set("bond_dims", [int(d) for d in ttno_obj.bond_dims])
        node.base.attributes.set(
            "dtype",
            str(np.result_type(*(nd.tensor.dtype for nd in ttno_obj.node_list))),
        )
        node.base.attributes.set("basis_tree_data_uuid", str(basis_tree_data.uuid))

        artifact = write_external_artifact(
            ttno_obj,
            storage_backend=storage_backend,
            storage_base=storage_base,
            relative_path=relative_path,
            artifact_format="ttno.npz",
        )
        for key, value in artifact.items():
            node.base.attributes.set(key, value)

        manifest = build_artifact_manifest(
            node_uuid=str(node.uuid),
            artifact=artifact,
            summary={
                "n_nodes": node.base.attributes.get("n_nodes"),
                "bond_dims": node.base.attributes.get("bond_dims"),
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

    def load_ttno(self, basis_tree_data: BasisTreeData | None = None) -> TTNO:
        """Reconstruct the TTNO from stored data.

        Uses TTNBase.load to avoid TTNO.__init__ requiring ``terms``
        (the operator terms are not persisted; only the numeric tensors are).

        Args:
            basis_tree_data: Optional BasisTreeData to use. If None, loads from stored UUID.

        Returns:
            The Renormalizer TTNO object.
        """
        if basis_tree_data is None:
            basis_tree_data = self.basis_tree_data
        basis_tree = basis_tree_data.load_basis_tree()

        from renormalizer.tn.tree import TTNO, TTNBase

        artifact_path = resolve_artifact_path(
            self.artifact_metadata["storage_backend"],
            self.artifact_metadata["storage_base"],
            self.artifact_metadata["relative_path"],
        )
        if not artifact_path.exists():
            raise FileNotFoundError(f"TTNO artifact not found: {artifact_path}")

        # Load via TTNBase.load to avoid TTNO.__init__ requiring terms
        ttno_loaded = TTNBase.load(basis_tree, str(artifact_path), other_attrs=[])

        # Patch class and terms so the object behaves like a proper TTNO
        ttno_loaded.__class__ = TTNO
        ttno_loaded.terms = []

        return ttno_loaded

    def relink_artifact(self, storage_base: str, relative_path: str) -> None:
        """Update the logical artifact location."""
        self.base.extras.set("artifact_storage_base", storage_base)
        self.base.extras.set("artifact_relative_path", relative_path)
