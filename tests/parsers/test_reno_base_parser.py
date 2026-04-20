"""Focused tests for RenoBaseParser validation and artifact routing."""
from __future__ import annotations

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


def test_validate_physical_constraints_accepts_normal_case():
    params = {"energy": -10.5, "bond_dims": [8, 16, 8], "converged": True}
    result = RenoBaseParser._validate_physical_constraints(None, params)
    assert result["passed"] is True


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
