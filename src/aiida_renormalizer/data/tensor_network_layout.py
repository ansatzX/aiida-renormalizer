"""Tensor-network layout metadata shared by MPS/MPO and TTNS/TTNO."""
from __future__ import annotations

import typing as t

from aiida.orm import Data

from aiida_renormalizer.data.utils import read_json_from_repository, write_json_to_repository

if t.TYPE_CHECKING:
    from aiida_renormalizer.data.basis_tree import BasisTreeData


class TensorNetworkLayoutData(Data):
    """Store reusable tensor-network layout metadata.

    This node captures expensive/common structural metadata so state/operator nodes
    can reference a shared layout instead of rebuilding equivalent descriptors.
    """

    @classmethod
    def from_chain(
        cls,
        dof_order: list[str],
        *,
        labels: list[str] | None = None,
    ) -> TensorNetworkLayoutData:
        """Create a layout node for 1D chain networks (MPS/MPO)."""
        node = cls()
        labels = labels or dof_order
        if len(labels) != len(dof_order):
            raise ValueError("labels must match dof_order length")
        node.base.attributes.set("network_kind", "chain")
        node.base.attributes.set("n_sites", len(dof_order))
        write_json_to_repository(
            node,
            {"dof_order": [str(d) for d in dof_order], "labels": [str(v) for v in labels]},
            "layout.json",
        )
        return node

    @classmethod
    def from_basis_tree_data(
        cls,
        basis_tree_data: BasisTreeData,
    ) -> TensorNetworkLayoutData:
        """Create a layout node from stored tree topology (TTNS/TTNO)."""
        node = cls()
        tree = read_json_from_repository(basis_tree_data, "tree_structure.json")
        nodes = tree.get("nodes", [])
        node.base.attributes.set("network_kind", "tree")
        node.base.attributes.set("n_nodes", len(nodes))
        layout_nodes: list[dict[str, t.Any]] = []
        for idx, entry in enumerate(nodes):
            layout_nodes.append(
                {
                    "node_idx": int(entry.get("node_idx", idx)),
                    "children_indices": [int(i) for i in entry.get("children_indices", [])],
                    "n_basis_sets": len(entry.get("basis_sets", [])),
                }
            )
        write_json_to_repository(node, {"nodes": layout_nodes}, "layout.json")
        return node

    def load_layout(self) -> dict[str, t.Any]:
        """Return the stored layout payload."""
        return read_json_from_repository(self, "layout.json")
