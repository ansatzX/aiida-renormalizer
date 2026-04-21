"""End-to-end integration test: Model → ModelData → MPS → MPSData → roundtrip."""
from __future__ import annotations

import numpy as np
import pytest


class TestEndToEnd:
    def test_full_roundtrip_workflow(self, aiida_profile, artifact_storage_base):
        """Simulate a typical data flow: build model, create MPS, store, reload, compute."""
        from renormalizer.model import Model, Op
        from renormalizer.model.basis import BasisHalfSpin, BasisSHO
        from renormalizer.model.op import OpSum
        from renormalizer.mps import Mpo, Mps

        from aiida_renormalizer.data import ModelData, MPOData, MPSData, OpData

        # 1. Build a spin-boson model
        basis = [BasisHalfSpin("spin"), BasisSHO("v0", omega=1.0, nbas=6)]
        ham = OpSum([
            Op("sigma_z", "spin", 1.0),
            Op("b^\\dagger b", "v0", 1.0),
            Op("sigma_x", "spin", 0.5),
        ])
        model = Model(basis, ham)

        # 2. Store model
        model_node = ModelData.from_model(model)
        model_node.store()

        # 3. Create MPS and MPO
        MPS = Mps.random(model, qntot=0, m_max=10)
        MPO = Mpo(model)

        # 4. Compute expectation before storing
        e_before = MPS.expectation(MPO)

        # 5. Store MPS and MPO
        mps_node = MPSData.from_mps(
            MPS,
            model_node,
            storage_backend="posix",
            storage_base=str(artifact_storage_base),
            relative_path="integration/full_roundtrip.npz",
        )
        mps_node.store()
        mpo_node = MPOData.from_mpo(MPO, model_node)
        mpo_node.store()

        # 6. Store an observable operator
        obs_op = Op("sigma_z", "spin", 1.0)
        op_node = OpData.from_op(obs_op)
        op_node.store()

        # 7. Reload everything from stored nodes
        restored_MPS = mps_node.load_mps()
        restored_MPO = mpo_node.load_mpo()
        restored_obs = op_node.load_op()

        # 8. Verify expectation consistency
        e_after = restored_MPS.expectation(restored_MPO)
        np.testing.assert_allclose(e_after, e_before, rtol=1e-10)

        # 9. Verify model structure
        restored_model = model_node.load_model()
        assert len(restored_model.basis) == 2
        assert type(restored_model.basis[0]).__name__ == "BasisHalfSpin"
        assert type(restored_model.basis[1]).__name__ == "BasisSHO"

        # 10. Verify observable
        assert restored_obs.symbol == "sigma_z"

    def test_multiple_mps_share_model(self, aiida_profile, sho_model, artifact_storage_base):
        """Multiple MPS snapshots from the same model all link to the same ModelData."""
        from renormalizer.mps import Mps

        from aiida_renormalizer.data import ModelData, MPSData

        model_node = ModelData.from_model(sho_model)
        model_node.store()

        MPS_list = [Mps.random(sho_model, qntot=0, m_max=10) for _ in range(3)]
        mps_nodes = [
            MPSData.from_mps(
                m,
                model_node,
                storage_backend="posix",
                storage_base=str(artifact_storage_base),
                relative_path=f"integration/mps_{idx}.npz",
            )
            for idx, m in enumerate(MPS_list)
        ]
        for n in mps_nodes:
            n.store()

        # All point to same ModelData
        for n in mps_nodes:
            assert n.model_data.uuid == model_node.uuid

        # Each can be independently loaded and matches original bond dims
        for orig, node in zip(MPS_list, mps_nodes):
            restored_MPS = node.load_mps()
            assert list(restored_MPS.bond_dims) == list(orig.bond_dims)
