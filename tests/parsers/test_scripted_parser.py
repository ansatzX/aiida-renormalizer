"""Focused behavioral tests for ScriptedParser."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from aiida.orm import CalcJobNode

from aiida_renormalizer.data import ModelData, MPSData, TensorNetworkLayoutData
from aiida_renormalizer.parsers.scripted import ScriptedParser


class _Inputs(dict):
    def __getattr__(self, key):
        return self[key]


@pytest.fixture
def scripted_parser(aiida_profile):
    node = CalcJobNode()
    node.set_option("resources", {"num_machines": 1})
    return ScriptedParser(node=node)


@pytest.fixture
def simple_model():
    from renormalizer.model import Model
    from renormalizer.model.basis import BasisSHO
    from renormalizer.model.op import Op

    basis = [BasisSHO(f"v{i}", omega=1.0, nbas=4) for i in range(2)]
    ham_terms = [Op(r"b^\dagger b", f"v{i}", 1.0) for i in range(2)]
    return ModelData.from_model(Model(basis, ham_terms))


@pytest.fixture
def simple_MPS(simple_model, tmp_path):
    from renormalizer.mps import Mps

    model = simple_model.load_model()
    MPS = Mps.random(model, qntot=0, m_max=8)
    return MPSData.from_mps(
        MPS,
        simple_model,
        storage_backend="posix",
        storage_base=str(tmp_path / "fixture-artifacts"),
        relative_path="states/simple_mps.npz",
    )


def test_validate_physical_constraints_detects_nan(scripted_parser):
    result = scripted_parser._validate_physical_constraints({"energy": float("nan")})
    assert result["passed"] is False


def test_parse_mps_file_returns_mpsdata(scripted_parser, simple_model, simple_MPS, tmp_path):
    model = simple_model.load_model()
    MPS = simple_MPS.load_mps(simple_model)
    npz_path = Path(tmp_path) / "output_mps.npz"
    MPS.dump(str(npz_path.with_suffix("")))
    assert npz_path.exists()

    mock_retrieved = MagicMock()
    mock_retrieved.open = MagicMock(side_effect=lambda _f, mode: open(npz_path, mode))
    artifact_base = Path(tmp_path) / "parsed-artifacts"
    mock_node = MagicMock()
    mock_node.get_option.side_effect = lambda key, default=None: {
        "artifact_storage_backend": "posix",
        "artifact_storage_base": str(artifact_base),
    }.get(key, default)

    with patch.object(type(scripted_parser), "retrieved", new_callable=lambda: property(lambda self: mock_retrieved)), \
         patch.object(type(scripted_parser), "node", new_callable=lambda: property(lambda self: mock_node)):
        parsed = scripted_parser._parse_mps_file("output_mps.npz", simple_model)

    assert isinstance(parsed, MPSData)
    loaded = parsed.load_mps(simple_model)
    assert len(loaded) == model.nsite


def test_resolve_chain_layout_prefers_input_layout(scripted_parser, simple_model):
    provided_layout = TensorNetworkLayoutData.from_chain(["x0", "x1"])
    mock_node = MagicMock()
    mock_node.inputs = _Inputs(tn_layout=provided_layout)

    with patch.object(type(scripted_parser), "node", new_callable=lambda: property(lambda self: mock_node)):
        resolved = scripted_parser._resolve_chain_layout(simple_model)

    assert resolved.uuid == provided_layout.uuid
