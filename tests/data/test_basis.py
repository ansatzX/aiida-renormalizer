"""Tests for BasisSetData."""
from __future__ import annotations

import pytest


class TestBasisSetData:
    def test_from_basis_list_roundtrip(self, aiida_profile, sho_basis):
        from aiida_renormalizer.data.basis import BasisSetData

        node = BasisSetData.from_basis_list(sho_basis)
        node.store()

        assert node.base.attributes.get("n_sites") == 2
        assert node.base.attributes.get("basis_types") == ["BasisSHO", "BasisSHO"]

        restored = node.load_basis_list()
        assert len(restored) == 2
        assert type(restored[0]).__name__ == "BasisSHO"
        assert restored[0].dof == "v0"
        assert restored[0].omega == pytest.approx(1.0)
        assert restored[1].dof == "v1"
        assert restored[1].omega == pytest.approx(1.5)

    def test_mixed_basis_types(self, aiida_profile):
        from renormalizer.model.basis import BasisHalfSpin, BasisSHO

        from aiida_renormalizer.data.basis import BasisSetData

        basis_list = [BasisHalfSpin("spin_0"), BasisSHO("v0", omega=1.0, nbas=4)]
        node = BasisSetData.from_basis_list(basis_list)
        node.store()

        assert node.base.attributes.get("basis_types") == ["BasisHalfSpin", "BasisSHO"]

        restored = node.load_basis_list()
        assert type(restored[0]).__name__ == "BasisHalfSpin"
        assert type(restored[1]).__name__ == "BasisSHO"
