"""Focused tests for RenoBaseParser validation and artifact routing."""
from __future__ import annotations

from types import SimpleNamespace

from renormalizer.tn.treebase import BasisTree

from aiida_renormalizer.data import BasisTreeData, ModelData, TensorNetworkLayoutData
from aiida_renormalizer.parsers.reno_base import RenoBaseParser


def test_validate_physical_constraints_rejects_nan():
    params = {"energy": float("nan"), "bond_dims": [10, 20]}
    result = RenoBaseParser._validate_physical_constraints(None, params)
    assert result["passed"] is False


def test_validate_physical_constraints_rejects_energy_increase():
    params = {
        "calc_type": "optimization",
        "energy_trajectory": [-10.0, -9.9],
        "bond_dims": [10, 20],
    }
    result = RenoBaseParser._validate_physical_constraints(None, params)
    assert result["passed"] is False
    assert "Energy increased" in result["reason"]


def test_get_artifact_location_uses_node_options():
    mock_node = type("Node", (), {})()
    mock_node.uuid = "abc123"
    mock_node.get_option = lambda key: {
        "artifact_storage_backend": "posix",
        "artifact_storage_base": "/tmp/artifacts",
        "artifact_relative_path": "custom/path.npz",
    }.get(key)

    parser = object.__new__(RenoBaseParser)
    from unittest.mock import patch

    with patch.object(type(parser), "node", new_callable=lambda: property(lambda self: mock_node)):
        backend, base, relative = RenoBaseParser._get_artifact_location(parser, "ignored.npz")
    assert backend == "posix"
    assert base == "/tmp/artifacts"
    assert relative == "custom/path.npz"


class _Inputs(dict):
    def __getattr__(self, key):
        return self[key]


def test_resolve_chain_layout_prefers_input_layout(aiida_profile, sho_model):
    model_data = ModelData.from_model(sho_model)
    provided_layout = TensorNetworkLayoutData.from_chain(["custom0", "custom1"])

    parser = object.__new__(RenoBaseParser)
    mock_node = SimpleNamespace(inputs=_Inputs(tn_layout=provided_layout))

    from unittest.mock import patch

    with patch.object(type(parser), "node", new_callable=lambda: property(lambda self: mock_node)):
        resolved = RenoBaseParser._resolve_chain_layout(parser, model_data)

    assert resolved.uuid == provided_layout.uuid


def test_resolve_tree_layout_builds_from_basis_tree(aiida_profile, sho_basis):
    basis_tree_data = BasisTreeData.from_basis_tree(BasisTree.binary(sho_basis))
    parser = object.__new__(RenoBaseParser)
    mock_node = SimpleNamespace(inputs=_Inputs())

    from unittest.mock import patch

    with patch.object(type(parser), "node", new_callable=lambda: property(lambda self: mock_node)):
        resolved = RenoBaseParser._resolve_tree_layout(parser, basis_tree_data)

    payload = resolved.load_layout()
    assert resolved.base.attributes.get("network_kind") == "tree"
    assert resolved.base.attributes.get("n_nodes") == len(payload["nodes"])
