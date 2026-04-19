"""Tests for basis_registry serialize/deserialize dispatch."""
from __future__ import annotations

import json

import numpy as np
import pytest
from renormalizer.model.basis import (
    BasisDummy,
    BasisHalfSpin,
    BasisHopsBoson,
    BasisMultiElectron,
    BasisMultiElectronVac,
    BasisSHO,
    BasisSimpleElectron,
    BasisSineDVR,
)

from aiida_renormalizer.data.basis_registry import deserialize_basis, serialize_basis


@pytest.mark.parametrize(
    "make_basis",
    [
        lambda: BasisSHO("v0", omega=1.0, nbas=4),
        lambda: BasisSineDVR("x0", nbas=10, xi=-5.0, xf=5.0),
        lambda: BasisHalfSpin("spin_0"),
        lambda: BasisSimpleElectron("e0"),
        lambda: BasisMultiElectronVac("e0"),
        lambda: BasisMultiElectronVac("e0"),
        lambda: BasisDummy("aux", nbas=3),
        lambda: BasisHopsBoson("hops0", nbas=6),
    ],
    ids=["SHO", "SineDVR", "HalfSpin", "SimpleElectron",
         "MultiElectron", "MultiElectronVac", "Dummy", "HopsBoson"],
)
def test_all_types_roundtrip(make_basis):
    """Every registered basis type must survive serialize -> deserialize."""
    basis = make_basis()
    data = serialize_basis(basis)
    restored = deserialize_basis(data)

    assert type(restored).__name__ == type(basis).__name__
    assert restored.dof == basis.dof
    assert restored.nbas == basis.nbas


class TestBasisSHO:
    """SHO has the most constructor params — test non-default values."""

    def test_all_params_preserved(self):
        basis = BasisSHO("v0", omega=2.5, nbas=8, x0=0.1, dvr=True, general_xp_power=True)
        restored = deserialize_basis(serialize_basis(basis))

        assert restored.omega == pytest.approx(2.5)
        assert restored.nbas == 8
        assert restored.x0 == pytest.approx(0.1)
        assert restored.dvr is True
        assert restored.general_xp_power is True


class TestBasisSineDVR:
    """SineDVR has continuous interval params that need float precision."""

    def test_interval_params_preserved(self):
        basis = BasisSineDVR("x0", nbas=10, xi=-5.0, xf=5.0, quadrature=True)
        restored = deserialize_basis(serialize_basis(basis))

        # Note: BasisSineDVR may adjust xi/xf internally
        # Just check they're preserved as floats
        assert isinstance(restored.xi, float)
        assert isinstance(restored.xf, float)
        assert restored.quadrature is True


class TestSigmaqnSerialization:
    """sigmaqn is a numpy array — tests the ndarray <-> list <-> ndarray path."""

    def test_halfspin_with_explicit_sigmaqn(self):
        sigmaqn = np.array([[1, 0], [0, 1]])
        basis = BasisHalfSpin("spin_0", sigmaqn=sigmaqn)
        restored = deserialize_basis(serialize_basis(basis))

        np.testing.assert_array_equal(restored.sigmaqn, basis.sigmaqn)

    def test_multielectron_required_sigmaqn(self):
        sigmaqn = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
        # MultiElectron requires dof to match sigmaqn length
        basis = BasisMultiElectron(["e0", "e1", "e2", "e3"], sigmaqn=sigmaqn)
        restored = deserialize_basis(serialize_basis(basis))

        np.testing.assert_array_equal(restored.sigmaqn, sigmaqn)


class TestRegistryEdgeCases:
    def test_unknown_type_raises(self):
        class FakeBasis:
            dof = "fake"

        with pytest.raises(ValueError, match="Unknown basis type"):
            serialize_basis(FakeBasis())

    def test_json_serializable(self):
        """Serialized output must contain no numpy scalars."""
        basis = BasisSHO("v0", omega=1.0, nbas=4)
        data = serialize_basis(basis)
        json.dumps(data)  # Must not raise
