"""Tests for SbmSymbolicSpecFromModesCalcJob."""
from __future__ import annotations

import numpy as np
from aiida import orm

from aiida_renormalizer.calculations.bath.sbm_symbolic_spec_from_modes import (
    SbmSymbolicSpecFromModesCalcJob,
)


def _make_calcjob(cls, inputs_dict):
    """Create a CalcJob instance without triggering plumpy Process.__init__."""
    from plumpy.utils import AttributesFrozendict

    calcjob = object.__new__(cls)
    calcjob._parsed_inputs = AttributesFrozendict(inputs_dict)
    return calcjob


def _array_data(name: str, values) -> orm.ArrayData:
    node = orm.ArrayData()
    node.set_array(name, np.asarray(values, dtype=float))
    return node


def test_inputs_outputs():
    from unittest.mock import Mock

    spec = Mock()
    spec.input = Mock()
    spec.output = Mock()
    spec.exit_code = Mock()
    spec.options = {}

    SbmSymbolicSpecFromModesCalcJob.define(spec)

    input_names = [call[0][0] for call in spec.input.call_args_list]
    assert "omega_k" in input_names
    assert "c_j2" in input_names
    assert "delta_eff" in input_names
    assert "symbol_map" in input_names

    output_names = [call[0][0] for call in spec.output.call_args_list]
    assert "output_parameters" in output_names


def test_retrieve_list_strips_wavefunction_artifacts():
    calc = _make_calcjob(
        SbmSymbolicSpecFromModesCalcJob,
        {
            "omega_k": _array_data("omega_k", [0.5, 1.0]),
            "c_j2": _array_data("c_j2", [0.1, 0.2]),
            "delta_eff": orm.Float(0.3),
        },
    )
    retrieve_list = calc._get_retrieve_list()
    assert "output_parameters.json" in retrieve_list
    assert "output_mps.npz" not in retrieve_list
    assert "output_mpo.npz" not in retrieve_list
