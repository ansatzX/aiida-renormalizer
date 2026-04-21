"""Tests for ModelFromSymbolicSpecCalcJob."""
from __future__ import annotations

from aiida import orm

from aiida_renormalizer.calculations.basic.model_from_symbolic import ModelFromSymbolicSpecCalcJob


def test_inputs_outputs():
    from unittest.mock import Mock

    spec = Mock()
    spec.input = Mock()
    spec.output = Mock()
    spec.exit_code = Mock()
    spec.options = {}

    ModelFromSymbolicSpecCalcJob.define(spec)
    input_names = [call[0][0] for call in spec.input.call_args_list]
    output_names = [call[0][0] for call in spec.output.call_args_list]

    assert "symbolic_inputs" in input_names
    assert "output_model" in output_names


def test_symbolic_validator():
    good = orm.Dict(
        dict={
            "basis": [{"kind": "half_spin", "dof": "spin", "sigmaqn": [0, 0]}],
            "hamiltonian": [{"symbol": "sigma_x", "dofs": "spin", "factor": 1.0}],
        }
    )
    bad = orm.Dict(dict={"basis": [], "hamiltonian": []})

    assert ModelFromSymbolicSpecCalcJob._validate_symbolic_inputs(good, None) is None
    assert ModelFromSymbolicSpecCalcJob._validate_symbolic_inputs(bad, None) is not None
