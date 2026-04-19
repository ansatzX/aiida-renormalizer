"""Tests for ModelData."""
from __future__ import annotations

import pytest


class TestModelData:
    def test_roundtrip_simple(self, aiida_profile, sho_model):
        from aiida_renormalizer.data.model import ModelData

        node = ModelData.from_model(sho_model)
        node.store()

        assert node.base.attributes.get("n_sites") == 2
        assert node.base.attributes.get("dof_list") == ["v0", "v1"]
        assert node.base.attributes.get("basis_types") == ["BasisSHO", "BasisSHO"]

        restored = node.load_model()
        assert len(restored.basis) == 2
        assert restored.basis[0].dof == "v0"
        assert restored.basis[0].omega == pytest.approx(1.0)
        assert restored.basis[1].omega == pytest.approx(1.5)

        # Verify Hamiltonian terms roundtrip
        assert len(restored.ham_terms) == len(sho_model.ham_terms)
        for orig, rest in zip(sho_model.ham_terms, restored.ham_terms):
            assert orig.symbol == rest.symbol
            assert list(orig.dofs) == list(rest.dofs)
            assert orig.factor == pytest.approx(rest.factor)

    def test_roundtrip_with_dipole(self, aiida_profile):
        from renormalizer.model import Model, Op
        from renormalizer.model.basis import BasisHalfSpin, BasisSHO
        from renormalizer.model.op import OpSum

        from aiida_renormalizer.data.model import ModelData

        basis = [BasisHalfSpin("spin_0"), BasisSHO("v0", omega=1.0, nbas=4)]
        ham = OpSum([Op("sigma_z", "spin_0", 1.0), Op("b^\\dagger b", "v0", 1.0)])
        dipole = OpSum([Op("sigma_x", "spin_0", 1.0)])
        model = Model(basis, ham, dipole=dipole)

        node = ModelData.from_model(model)
        node.store()

        restored = node.load_model()
        assert restored.dipole is not None
        assert len(restored.dipole) == 1
        assert restored.dipole[0].symbol == "sigma_x"

    def test_restored_model_produces_identical_mpo(self, aiida_profile, sho_model, sho_mps):
        """MPO built from restored Model gives identical expectation on the SAME MPS."""
        import numpy as np
        from renormalizer.mps import Mpo

        from aiida_renormalizer.data.model import ModelData

        mpo_orig = Mpo(sho_model)
        e_orig = sho_mps.expectation(mpo_orig)

        node = ModelData.from_model(sho_model)
        node.store()
        restored_model = node.load_model()

        mpo_restored = Mpo(restored_model)
        e_restored = sho_mps.expectation(mpo_restored)

        np.testing.assert_allclose(e_restored, e_orig, rtol=1e-10)
