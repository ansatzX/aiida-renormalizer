"""Tests for MPOData."""
from __future__ import annotations

import hashlib

import numpy as np
import pytest


class TestMPOData:
    def test_roundtrip_external_artifact(self, aiida_profile, sho_model, sho_mpo, tmp_path):
        from aiida_renormalizer.data.model import ModelData
        from aiida_renormalizer.data.mpo import MPOData

        model_node = ModelData.from_model(sho_model)
        model_node.store()

        artifact_base = tmp_path / "mpo-artifacts"
        mpo_node = MPOData.from_mpo(
            sho_mpo,
            model_node,
            storage_backend="posix",
            storage_base=str(artifact_base),
            relative_path="operators/hamiltonian.npz",
        )
        mpo_node.store()

        artifact_path = artifact_base / "operators" / "hamiltonian.npz"
        assert artifact_path.exists()
        assert mpo_node.artifact_metadata["storage_base"] == str(artifact_base)
        assert mpo_node.artifact_metadata["content_hash"] == hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest()

        restored = mpo_node.load_mpo()
        assert len(restored) == len(sho_mpo)

    def test_expectation_consistency(self, aiida_profile, sho_model, sho_mps, sho_mpo, tmp_path):
        from aiida_renormalizer.data.model import ModelData
        from aiida_renormalizer.data.mpo import MPOData

        model_node = ModelData.from_model(sho_model)
        model_node.store()

        mpo_node = MPOData.from_mpo(
            sho_mpo,
            model_node,
            storage_backend="posix",
            storage_base=str(tmp_path / "archive"),
            relative_path="operators/mpo.npz",
        )
        mpo_node.store()

        e_orig = sho_mps.expectation(sho_mpo)
        e_restored = sho_mps.expectation(mpo_node.load_mpo())

        np.testing.assert_allclose(e_restored, e_orig, rtol=1e-10)

    def test_missing_external_artifact_raises(self, aiida_profile, sho_model, sho_mpo, tmp_path):
        from aiida_renormalizer.data.model import ModelData
        from aiida_renormalizer.data.mpo import MPOData

        model_node = ModelData.from_model(sho_model)
        model_node.store()

        mpo_node = MPOData.from_mpo(
            sho_mpo,
            model_node,
            storage_backend="posix",
            storage_base=str(tmp_path / "missing"),
            relative_path="operators/mpo.npz",
        )
        mpo_node.store()

        artifact_path = tmp_path / "missing" / "operators" / "mpo.npz"
        artifact_path.unlink()

        with pytest.raises(FileNotFoundError):
            mpo_node.load_mpo()
