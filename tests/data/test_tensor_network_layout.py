"""Tests for TensorNetworkLayoutData."""
from __future__ import annotations


class TestTensorNetworkLayoutData:
    def test_chain_layout_roundtrip(self, aiida_profile):
        from aiida_renormalizer.data.tensor_network_layout import TensorNetworkLayoutData

        node = TensorNetworkLayoutData.from_chain(["spin", "v0", "v1"])
        node.store()

        assert node.base.attributes.get("network_kind") == "chain"
        assert node.base.attributes.get("n_sites") == 3
        payload = node.load_layout()
        assert payload["dof_order"] == ["spin", "v0", "v1"]

    def test_tree_layout_from_basis_tree(self, aiida_profile, sho_basis):
        from renormalizer.tn.treebase import BasisTree

        from aiida_renormalizer.data.basis_tree import BasisTreeData
        from aiida_renormalizer.data.tensor_network_layout import TensorNetworkLayoutData

        basis_tree = BasisTree.binary(sho_basis)
        basis_tree_data = BasisTreeData.from_basis_tree(basis_tree)
        basis_tree_data.store()

        node = TensorNetworkLayoutData.from_basis_tree_data(basis_tree_data)
        node.store()

        assert node.base.attributes.get("network_kind") == "tree"
        assert node.base.attributes.get("n_nodes") == len(basis_tree)
        payload = node.load_layout()
        assert len(payload["nodes"]) == len(basis_tree)
