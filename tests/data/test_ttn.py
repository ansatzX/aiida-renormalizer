"""Tests for TTN data nodes (BasisTreeData, TTNSData, TTNOData)."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest


class TestBasisTreeData:
    """Tests for BasisTreeData serialization."""

    def test_roundtrip_linear_tree(self, aiida_profile, sho_basis):
        from renormalizer.tn.treebase import BasisTree

        from aiida_renormalizer.data.basis_tree import BasisTreeData

        basis_tree = BasisTree.linear(sho_basis)
        node = BasisTreeData.from_basis_tree(basis_tree)
        node.store()

        assert node.base.attributes.get("n_nodes") == 2
        assert node.base.attributes.get("qn_size") == basis_tree.qn_size

        restored = node.load_basis_tree()
        assert len(restored) == len(basis_tree)
        assert restored.qn_size == basis_tree.qn_size

    def test_basis_tree_compiled_cache_available(self, aiida_profile, sho_basis, tmp_path):
        from renormalizer.tn.treebase import BasisTree

        from aiida_renormalizer.data.basis_tree import BasisTreeData

        basis_tree = BasisTree.binary(sho_basis)
        node = BasisTreeData.from_basis_tree(basis_tree)
        node.store()

        out_path = tmp_path / "basis_tree_cached.npz"
        with out_path.open("wb") as handle:
            wrote = node.write_cached_pickle(handle)

        assert wrote is True
        assert out_path.exists()
        assert out_path.stat().st_size > 0


class TestTTNSData:
    """Tests for TTNSData serialization."""

    def test_roundtrip_external_artifact(self, aiida_profile, sho_basis, tmp_path):
        from renormalizer.tn.tree import TTNS
        from renormalizer.tn.treebase import BasisTree

        from aiida_renormalizer.data.basis_tree import BasisTreeData
        from aiida_renormalizer.data.ttns import TTNSData

        basis_tree = BasisTree.binary(sho_basis)
        TTNS = TTNS.random(basis_tree, qntot=0, m_max=10)

        basis_tree_node = BasisTreeData.from_basis_tree(basis_tree)
        basis_tree_node.store()

        artifact_base = tmp_path / "ttns-artifacts"
        ttns_node = TTNSData.from_ttns(
            TTNS,
            basis_tree_node,
            storage_backend="posix",
            storage_base=str(artifact_base),
            relative_path="states/final_ttns.npz",
        )
        ttns_node.store()

        artifact_path = artifact_base / "states" / "final_ttns.npz"
        assert artifact_path.exists()
        assert ttns_node.artifact_metadata["storage_base"] == str(artifact_base)
        assert ttns_node.artifact_metadata["relative_path"] == "states/final_ttns.npz"
        assert ttns_node.artifact_metadata["content_hash"] == hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest()

        restored_TTNS = ttns_node.load_ttns()
        assert len(restored_TTNS) == len(TTNS)
        assert list(restored_TTNS.bond_dims) == list(TTNS.bond_dims)
        assert np.allclose(restored_TTNS.qntot, TTNS.qntot)
        assert restored_TTNS.coeff == TTNS.coeff

    def test_roundtrip_preserves_expectation(self, aiida_profile, sho_basis, tmp_path):
        from renormalizer import Op
        from renormalizer.model.op import OpSum
        from renormalizer.tn.tree import TTNO, TTNS
        from renormalizer.tn.treebase import BasisTree

        from aiida_renormalizer.data.basis_tree import BasisTreeData
        from aiida_renormalizer.data.ttno import TTNOData
        from aiida_renormalizer.data.ttns import TTNSData

        basis_tree = BasisTree.binary(sho_basis)
        TTNS = TTNS.random(basis_tree, qntot=0, m_max=10)
        ham_terms = OpSum([Op("b^\\dagger b", "v0", 1.0), Op("b^\\dagger b", "v1", 1.5)])
        TTNO = TTNO(basis_tree, ham_terms)

        e_orig = TTNS.expectation(TTNO)

        basis_tree_node = BasisTreeData.from_basis_tree(basis_tree)
        basis_tree_node.store()

        ttns_node = TTNSData.from_ttns(
            TTNS,
            basis_tree_node,
            storage_backend="posix",
            storage_base=str(tmp_path / "archive"),
            relative_path="states/ttns.npz",
        )
        ttns_node.store()

        ttno_node = TTNOData.from_ttno(TTNO, basis_tree_node)
        ttno_node.store()

        restored_TTNS = ttns_node.load_ttns()
        restored_TTNO = ttno_node.load_ttno()

        e_restored = restored_TTNS.expectation(restored_TTNO)
        np.testing.assert_allclose(e_restored, e_orig, rtol=1e-10)

    def test_missing_external_artifact_raises(self, aiida_profile, sho_basis, tmp_path):
        from renormalizer.tn.tree import TTNS
        from renormalizer.tn.treebase import BasisTree

        from aiida_renormalizer.data.basis_tree import BasisTreeData
        from aiida_renormalizer.data.ttns import TTNSData

        basis_tree = BasisTree.binary(sho_basis)
        TTNS = TTNS.random(basis_tree, qntot=0, m_max=10)

        basis_tree_node = BasisTreeData.from_basis_tree(basis_tree)
        basis_tree_node.store()

        ttns_node = TTNSData.from_ttns(
            TTNS,
            basis_tree_node,
            storage_backend="posix",
            storage_base=str(tmp_path / "missing"),
            relative_path="states/ttns.npz",
        )
        ttns_node.store()

        artifact_path = tmp_path / "missing" / "states" / "ttns.npz"
        artifact_path.unlink()

        with pytest.raises(FileNotFoundError):
            ttns_node.load_ttns()

    def test_basis_tree_link(self, aiida_profile, sho_basis, tmp_path):
        from renormalizer.tn.tree import TTNS
        from renormalizer.tn.treebase import BasisTree

        from aiida_renormalizer.data.basis_tree import BasisTreeData
        from aiida_renormalizer.data.tensor_network_layout import TensorNetworkLayoutData
        from aiida_renormalizer.data.ttns import TTNSData

        basis_tree = BasisTree.binary(sho_basis)
        TTNS = TTNS.random(basis_tree, qntot=0, m_max=10)

        basis_tree_node = BasisTreeData.from_basis_tree(basis_tree)
        basis_tree_node.store()
        layout_node = TensorNetworkLayoutData.from_basis_tree_data(basis_tree_node)
        layout_node.store()

        ttns_node = TTNSData.from_ttns(
            TTNS,
            basis_tree_node,
            layout_node,
            storage_backend="posix",
            storage_base=str(tmp_path / "artifacts"),
            relative_path="states/ttns.npz",
        )
        ttns_node.store()

        retrieved_basis_node = ttns_node.basis_tree_data
        assert retrieved_basis_node.uuid == basis_tree_node.uuid
        assert ttns_node.tn_layout_data is not None
        assert ttns_node.tn_layout_data.uuid == layout_node.uuid


class TestTTNOData:
    """Tests for TTNOData serialization."""

    def test_roundtrip(self, aiida_profile, sho_basis, tmp_path):
        from renormalizer import Op
        from renormalizer.model.op import OpSum
        from renormalizer.tn.tree import TTNO
        from renormalizer.tn.treebase import BasisTree

        from aiida_renormalizer.data.basis_tree import BasisTreeData
        from aiida_renormalizer.data.ttno import TTNOData

        basis_tree = BasisTree.binary(sho_basis)
        ham_terms = OpSum([Op("b^\\dagger b", "v0", 1.0), Op("b^\\dagger b", "v1", 1.5)])
        TTNO = TTNO(basis_tree, ham_terms)

        basis_tree_node = BasisTreeData.from_basis_tree(basis_tree)
        basis_tree_node.store()

        ttno_node = TTNOData.from_ttno(
            TTNO,
            basis_tree_node,
            storage_backend="posix",
            storage_base=str(tmp_path / "ttno-artifacts"),
            relative_path="operators/ttno.npz",
        )
        ttno_node.store()

        restored_TTNO = ttno_node.load_ttno()
        assert len(restored_TTNO) == len(TTNO)
        assert list(restored_TTNO.bond_dims) == list(TTNO.bond_dims)
