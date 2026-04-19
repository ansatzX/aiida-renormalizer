"""BasisTreeData node for Renormalizer BasisTree persistence."""
from __future__ import annotations

import typing as t

from aiida.orm import Data

if t.TYPE_CHECKING:
    from renormalizer.tn.treebase import BasisTree

from aiida_renormalizer.data.basis_registry import deserialize_basis, serialize_basis
from aiida_renormalizer.data.utils import read_json_from_repository, write_json_to_repository


class BasisTreeData(Data):
    """AiiDA Data node storing a Renormalizer BasisTree (tree topology + basis grouping).

    The BasisTree defines the tree structure for TTN calculations, including:
    - Tree topology (parent-child relationships)
    - Basis sets grouped at each tree node
    - Quantum number consistency
    """

    @classmethod
    def from_basis_tree(cls, basis_tree: BasisTree) -> BasisTreeData:
        """Create a BasisTreeData node from a Reno BasisTree.

        Args:
            basis_tree: The Renormalizer BasisTree object.

        Returns:
            BasisTreeData node storing the tree structure and basis sets.
        """
        node = cls()

        # Attributes (lightweight metadata)
        node.base.attributes.set("n_nodes", len(basis_tree))
        node.base.attributes.set("qn_size", basis_tree.qn_size)
        node.base.attributes.set("dof_list", [str(dof) for dof in basis_tree.dof_list])

        # Repository: tree_structure.json (topology and basis sets)
        tree_data = _serialize_basis_tree(basis_tree)
        write_json_to_repository(node, tree_data, "tree_structure.json")

        return node

    def load_basis_tree(self) -> BasisTree:
        """Reconstruct the BasisTree from stored data.

        Returns:
            The Renormalizer BasisTree object.
        """
        from renormalizer.tn.node import TreeNodeBasis
        from renormalizer.tn.treebase import BasisTree as RenoBasisTree

        tree_data = read_json_from_repository(self, "tree_structure.json")

        # Reconstruct tree nodes
        node_list = []
        for node_data in tree_data["nodes"]:
            basis_sets = [deserialize_basis(b_data) for b_data in node_data["basis_sets"]]
            tree_node = TreeNodeBasis(basis_sets)
            node_list.append(tree_node)

        # Reconstruct parent-child relationships
        for i, node_data in enumerate(tree_data["nodes"]):
            for child_idx in node_data["children_indices"]:
                node_list[i].add_child(node_list[child_idx])

        # Find root (node with no parent)
        root = None
        for tree_node in node_list:
            if tree_node.parent is None:
                root = tree_node
                break

        return RenoBasisTree(root)


def _serialize_basis_tree(basis_tree: BasisTree) -> dict:
    """Serialize a BasisTree to a JSON-safe dict.

    The structure is:
    {
        "nodes": [
            {
                "basis_sets": [serialized_basis1, ...],
                "children_indices": [0, 1, ...],  # indices of children in the node_list
                "node_idx": 0  # index in the preorder traversal
            },
            ...
        ]
    }

    Args:
        basis_tree: The Renormalizer BasisTree object.

    Returns:
        JSON-serializable dict representing the tree.
    """
    nodes_data = []

    for i, tree_node in enumerate(basis_tree.node_list):
        # Serialize basis sets at this node
        basis_sets_data = [serialize_basis(b) for b in tree_node.basis_sets]

        # Get indices of children in the preorder list
        children_indices = [basis_tree.node_idx[child] for child in tree_node.children]

        node_data = {
            "basis_sets": basis_sets_data,
            "children_indices": children_indices,
            "node_idx": i,
        }
        nodes_data.append(node_data)

    return {"nodes": nodes_data}
