"""Tests for MPSData."""
from __future__ import annotations

import hashlib
import shutil

import numpy as np
import pytest


class TestMPSData:
    def test_roundtrip_external_artifact(self, aiida_profile, sho_model, sho_mps, tmp_path):
        from aiida_renormalizer.data.model import ModelData
        from aiida_renormalizer.data.mps import MPSData

        model_node = ModelData.from_model(sho_model)
        model_node.store()

        artifact_base = tmp_path / "artifacts"
        mps_node = MPSData.from_mps(
            sho_mps,
            model_node,
            storage_backend="posix",
            storage_base=str(artifact_base),
            relative_path="states/ground_state.npz",
        )
        mps_node.store()

        artifact_path = artifact_base / "states" / "ground_state.npz"
        assert artifact_path.exists()
        assert mps_node.artifact_metadata["storage_backend"] == "posix"
        assert mps_node.artifact_metadata["storage_base"] == str(artifact_base)
        assert mps_node.artifact_metadata["relative_path"] == "states/ground_state.npz"
        assert mps_node.artifact_metadata["artifact_size"] == artifact_path.stat().st_size
        assert mps_node.artifact_metadata["content_hash"] == hashlib.sha256(
            artifact_path.read_bytes()
        ).hexdigest()

        restored_MPS = mps_node.load_mps()
        assert len(restored_MPS) == len(sho_mps)
        assert list(restored_MPS.bond_dims) == list(sho_mps.bond_dims)

    def test_roundtrip_preserves_expectation(
        self, aiida_profile, sho_model, sho_mps, sho_mpo, tmp_path
    ):
        from aiida_renormalizer.data.model import ModelData
        from aiida_renormalizer.data.mps import MPSData

        e_orig = sho_mps.expectation(sho_mpo)

        model_node = ModelData.from_model(sho_model)
        model_node.store()

        mps_node = MPSData.from_mps(
            sho_mps,
            model_node,
            storage_backend="posix",
            storage_base=str(tmp_path / "archive"),
            relative_path="states/final.npz",
        )
        mps_node.store()

        restored_model = model_node.load_model()
        restored_MPS = mps_node.load_mps()

        from renormalizer.mps import Mpo

        mpo_restored = Mpo(restored_model)
        e_restored = restored_MPS.expectation(mpo_restored)

        np.testing.assert_allclose(e_restored, e_orig, rtol=1e-10)

    def test_relink_artifact_updates_logical_location(
        self, aiida_profile, sho_model, sho_mps, tmp_path
    ):
        from aiida_renormalizer.data.model import ModelData
        from aiida_renormalizer.data.mps import MPSData

        model_node = ModelData.from_model(sho_model)
        model_node.store()

        first_base = tmp_path / "first"
        second_base = tmp_path / "second"
        mps_node = MPSData.from_mps(
            sho_mps,
            model_node,
            storage_backend="posix",
            storage_base=str(first_base),
            relative_path="states/mps.npz",
        )
        mps_node.store()

        source_path = first_base / "states" / "mps.npz"
        target_path = second_base / "published" / "mps.npz"
        target_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, target_path)

        mps_node.relink_artifact(str(second_base), "published/mps.npz")

        assert mps_node.artifact_metadata["storage_base"] == str(second_base)
        assert mps_node.artifact_metadata["relative_path"] == "published/mps.npz"
        restored_MPS = mps_node.load_mps()
        assert list(restored_MPS.bond_dims) == list(sho_mps.bond_dims)

    def test_missing_external_artifact_raises(self, aiida_profile, sho_model, sho_mps, tmp_path):
        from aiida_renormalizer.data.model import ModelData
        from aiida_renormalizer.data.mps import MPSData

        model_node = ModelData.from_model(sho_model)
        model_node.store()

        mps_node = MPSData.from_mps(
            sho_mps,
            model_node,
            storage_backend="posix",
            storage_base=str(tmp_path / "missing"),
            relative_path="states/mps.npz",
        )
        mps_node.store()

        artifact_path = tmp_path / "missing" / "states" / "mps.npz"
        artifact_path.unlink()

        with pytest.raises(FileNotFoundError):
            mps_node.load_mps()

    def test_model_data_link(self, aiida_profile, sho_model, sho_mps, tmp_path):
        from aiida_renormalizer.data.model import ModelData
        from aiida_renormalizer.data.mps import MPSData

        model_node = ModelData.from_model(sho_model)
        model_node.store()

        mps_node = MPSData.from_mps(
            sho_mps,
            model_node,
            storage_backend="posix",
            storage_base=str(tmp_path / "artifacts"),
            relative_path="states/linked.npz",
        )
        mps_node.store()

        retrieved_model_node = mps_node.model_data
        assert retrieved_model_node.uuid == model_node.uuid
