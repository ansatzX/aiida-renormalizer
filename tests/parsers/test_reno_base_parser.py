"""Tests for RenoBaseParser."""
from __future__ import annotations

import json

import numpy as np
import pytest
from aiida import orm

from aiida_renormalizer.data import ModelData


class TestRenoBaseParser:
    def test_parse_output_parameters(self, aiida_profile, sho_model):
        """Parse output_parameters.json into orm.Dict."""
        from aiida_renormalizer.parsers.reno_base import RenoBaseParser

        # Create a fake retrieved folder
        # (In real tests, this would come from a CalcJob execution)
        # For unit test, we'll mock it
        # This is a placeholder — actual test needs CalcJob infrastructure
        pass

    def test_exit_code_310_physical_validation(self, aiida_profile):
        """NaN in observables → exit code 310."""
        # Test that parser detects NaN and returns correct exit code
        pass

    def test_validate_physical_constraints_nan(self):
        """NaN in observables should fail validation."""
        from aiida_renormalizer.parsers.reno_base import RenoBaseParser

        params = {
            'energy': float('nan'),
            'bond_dims': [10, 20, 30],
        }

        result = RenoBaseParser._validate_physical_constraints(None, params)
        assert result['passed'] is False
        assert 'NaN' in result['reason']

    def test_validate_physical_constraints_inf(self):
        """Inf in observables should fail validation."""
        from aiida_renormalizer.parsers.reno_base import RenoBaseParser

        params = {
            'energy': float('inf'),
            'bond_dims': [10, 20, 30],
        }

        result = RenoBaseParser._validate_physical_constraints(None, params)
        assert result['passed'] is False
        assert 'Inf' in result['reason']

    def test_validate_physical_constraints_bond_dim_collapse(self):
        """Bond dimension collapse (all dims = 1) should fail validation."""
        from aiida_renormalizer.parsers.reno_base import RenoBaseParser

        params = {
            'energy': -10.5,
            'bond_dims': [1, 1, 1, 1],
        }

        result = RenoBaseParser._validate_physical_constraints(None, params)
        assert result['passed'] is False
        assert 'bond dimension' in result['reason'].lower()

    def test_validate_physical_constraints_spin_violation(self):
        """Spin constraint violation should fail validation."""
        from aiida_renormalizer.parsers.reno_base import RenoBaseParser

        params = {
            'energy': -10.5,
            'bond_dims': [10, 20, 30],
            'sigma_z': 1.5,  # Violates |<σ_z>| ≤ 1
        }

        result = RenoBaseParser._validate_physical_constraints(None, params)
        assert result['passed'] is False
        assert 'σ_z' in result['reason'] or 'sigma_z' in result['reason']

    def test_validate_physical_constraints_energy_monotonicity(self):
        """Energy increase during optimization should fail validation."""
        from aiida_renormalizer.parsers.reno_base import RenoBaseParser

        params = {
            'calc_type': 'optimization',
            'energy_trajectory': [-10.5, -10.3, -10.4],  # Energy increased
            'bond_dims': [10, 20, 30],
        }

        result = RenoBaseParser._validate_physical_constraints(None, params)
        assert result['passed'] is False
        assert 'Energy' in result['reason']

    def test_validate_physical_constraints_energy_conservation(self):
        """Energy drift during real-time evolution should fail validation."""
        from aiida_renormalizer.parsers.reno_base import RenoBaseParser

        params = {
            'calc_type': 'real_time_evolution',
            'energy_trajectory': [-10.5, -10.5, -10.3],  # Energy drifted
            'bond_dims': [10, 20, 30],
        }

        result = RenoBaseParser._validate_physical_constraints(None, params)
        assert result['passed'] is False
        assert 'Energy drift' in result['reason']

    def test_validate_physical_constraints_pass(self):
        """Valid parameters should pass validation."""
        from aiida_renormalizer.parsers.reno_base import RenoBaseParser

        params = {
            'energy': -10.5,
            'bond_dims': [10, 20, 30],
            'converged': True,
        }

        result = RenoBaseParser._validate_physical_constraints(None, params)
        assert result['passed'] is True

    def test_get_artifact_location_uses_node_options(self):
        """Artifact location should read CalcJobNode options, not input links."""
        from aiida_renormalizer.parsers.reno_base import RenoBaseParser

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
