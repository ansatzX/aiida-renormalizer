"""Tests for TTN data nodes (BasisTreeData, TTNSData, TtnoData)."""
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


class TestTTNSData:
    """Tests for TTNSData serialization."""

    def test_roundtrip_external_artifact(self, aiida_profile, sho_basis, tmp_path):
        from renormalizer.tn.tree import TTNS
        from renormalizer.tn.treebase import BasisTree

        from aiida_renormalizer.data.basis_tree import BasisTreeData
        from aiida_renormalizer.data.ttns import TTNSData

        basis_tree = BasisTree.binary(sho_basis)
        ttns = TTNS.random(basis_tree, qntot=0, m_max=10)

        basis_tree_node = BasisTreeData.from_basis_tree(basis_tree)
        basis_tree_node.store()

        artifact_base = tmp_path / "ttns-artifacts"
        ttns_node = TTNSData.from_ttns(
            ttns,
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

        restored = ttns_node.load_ttns()
        assert len(restored) == len(ttns)
        assert list(restored.bond_dims) == list(ttns.bond_dims)
        assert np.allclose(restored.qntot, ttns.qntot)
        assert restored.coeff == ttns.coeff

    def test_roundtrip_preserves_expectation(self, aiida_profile, sho_basis, tmp_path):
        from renormalizer import Op
        from renormalizer.model.op import OpSum
        from renormalizer.tn.tree import TTNO, TTNS
        from renormalizer.tn.treebase import BasisTree

        from aiida_renormalizer.data.basis_tree import BasisTreeData
        from aiida_renormalizer.data.ttno import TtnoData
        from aiida_renormalizer.data.ttns import TTNSData

        basis_tree = BasisTree.binary(sho_basis)
        ttns = TTNS.random(basis_tree, qntot=0, m_max=10)
        ham_terms = OpSum([Op("b^\\dagger b", "v0", 1.0), Op("b^\\dagger b", "v1", 1.5)])
        ttno = TTNO(basis_tree, ham_terms)

        e_orig = ttns.expectation(ttno)

        basis_tree_node = BasisTreeData.from_basis_tree(basis_tree)
        basis_tree_node.store()

        ttns_node = TTNSData.from_ttns(
            ttns,
            basis_tree_node,
            storage_backend="posix",
            storage_base=str(tmp_path / "archive"),
            relative_path="states/ttns.npz",
        )
        ttns_node.store()

        ttno_node = TtnoData.from_ttno(ttno, basis_tree_node)
        ttno_node.store()

        restored_ttns = ttns_node.load_ttns()
        restored_ttno = ttno_node.load_ttno()

        e_restored = restored_ttns.expectation(restored_ttno)
        np.testing.assert_allclose(e_restored, e_orig, rtol=1e-10)

    def test_missing_external_artifact_raises(self, aiida_profile, sho_basis, tmp_path):
        from renormalizer.tn.tree import TTNS
        from renormalizer.tn.treebase import BasisTree

        from aiida_renormalizer.data.basis_tree import BasisTreeData
        from aiida_renormalizer.data.ttns import TTNSData

        basis_tree = BasisTree.binary(sho_basis)
        ttns = TTNS.random(basis_tree, qntot=0, m_max=10)

        basis_tree_node = BasisTreeData.from_basis_tree(basis_tree)
        basis_tree_node.store()

        ttns_node = TTNSData.from_ttns(
            ttns,
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
        from aiida_renormalizer.data.ttns import TTNSData

        basis_tree = BasisTree.binary(sho_basis)
        ttns = TTNS.random(basis_tree, qntot=0, m_max=10)

        basis_tree_node = BasisTreeData.from_basis_tree(basis_tree)
        basis_tree_node.store()

        ttns_node = TTNSData.from_ttns(
            ttns,
            basis_tree_node,
            storage_backend="posix",
            storage_base=str(tmp_path / "artifacts"),
            relative_path="states/ttns.npz",
        )
        ttns_node.store()

        retrieved_basis_node = ttns_node.basis_tree_data
        assert retrieved_basis_node.uuid == basis_tree_node.uuid


class TestTtnoData:
    """Tests for TtnoData serialization."""

    def test_roundtrip(self, aiida_profile, sho_basis, tmp_path):
        from renormalizer import Op
        from renormalizer.model.op import OpSum
        from renormalizer.tn.tree import TTNO
        from renormalizer.tn.treebase import BasisTree

        from aiida_renormalizer.data.basis_tree import BasisTreeData
        from aiida_renormalizer.data.ttno import TtnoData

        basis_tree = BasisTree.binary(sho_basis)
        ham_terms = OpSum([Op("b^\\dagger b", "v0", 1.0), Op("b^\\dagger b", "v1", 1.5)])
        ttno = TTNO(basis_tree, ham_terms)

        basis_tree_node = BasisTreeData.from_basis_tree(basis_tree)
        basis_tree_node.store()

        ttno_node = TtnoData.from_ttno(
            ttno,
            basis_tree_node,
            storage_backend="posix",
            storage_base=str(tmp_path / "ttno-artifacts"),
            relative_path="operators/ttno.npz",
        )
        ttno_node.store()

        restored = ttno_node.load_ttno()
        assert len(restored) == len(ttno)
        assert list(restored.bond_dims) == list(ttno.bond_dims)
