"""Shared test fixtures for aiida-renormalizer."""
from __future__ import annotations

from pathlib import Path

import pytest

pytest_plugins = ["aiida.tools.pytest_fixtures"]


@pytest.fixture
def sho_basis():
    """Two-site SHO basis list."""
    from renormalizer.model.basis import BasisSHO

    return [BasisSHO("v0", omega=1.0, nbas=4), BasisSHO("v1", omega=1.5, nbas=4)]


@pytest.fixture
def sho_ham(sho_basis):
    """Simple harmonic Hamiltonian: H = ω₀ a†a₀ + ω₁ a†a₁."""
    from renormalizer.model import Op
    from renormalizer.model.op import OpSum

    return OpSum([Op("b^\\dagger b", "v0", 1.0), Op("b^\\dagger b", "v1", 1.5)])


@pytest.fixture
def sho_model(sho_basis, sho_ham):
    """Minimal 2-site SHO Model."""
    from renormalizer.model import Model

    return Model(sho_basis, sho_ham)


@pytest.fixture
def sho_mps(sho_model):
    """Random MPS for the SHO model."""
    from renormalizer.mps import Mps

    return Mps.random(sho_model, qntot=0, m_max=10)


@pytest.fixture
def sho_mpo(sho_model):
    """Hamiltonian MPO for the SHO model."""
    from renormalizer.mps import Mpo

    return Mpo(sho_model)


@pytest.fixture
def fixture_model_data(sho_model):
    """AiiDA ModelData fixture."""
    from aiida_renormalizer.data import ModelData
    return ModelData.from_model(sho_model)


@pytest.fixture
def fixture_code(aiida_code_installed):
    """AiiDA Code fixture pointing to Python with renormalizer."""
    from aiida import orm

    # Use the aiida_code_installed fixture from aiida
    # This creates a code that runs 'python'
    code = aiida_code_installed(
        label='python-renormalizer',
        computer=orm.load_computer('localhost'),
        filepath_executable='/usr/bin/python3',
    )
    return code


@pytest.fixture
def artifact_storage_base(tmp_path) -> Path:
    """Base directory for external wavefunction artifacts in tests."""
    return tmp_path / "artifacts"
