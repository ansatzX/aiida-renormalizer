"""Tests for ThermalPropCalcJob."""
from __future__ import annotations

import pytest
from aiida import orm
from aiida.common import AttributeDict

from aiida_renormalizer.data import ModelData, MPOData


def _make_calcjob(cls, inputs_dict):
    """Create a CalcJob instance without triggering plumpy Process.__init__."""
    from plumpy.utils import AttributesFrozendict
    calcjob = object.__new__(cls)
    calcjob._parsed_inputs = AttributesFrozendict(inputs_dict)
    return calcjob


class TestThermalPropCalcJob:
    """Tests for finite-temperature state preparation."""

    def test_thermal_prop_inputs_outputs(self):
        """ThermalPropCalcJob should define correct inputs/outputs."""
        from aiida_renormalizer.calculations.composite.thermal_prop import ThermalPropCalcJob
        from unittest.mock import Mock

        spec = Mock()
        spec.input = Mock()
        spec.output = Mock()
        spec.exit_code = Mock()
        spec.options = {}

        ThermalPropCalcJob.define(spec)

        # Check inputs
        input_calls = [call for call in spec.input.call_args_list]
        input_names = [call[0][0] for call in input_calls]

        assert "mpo" in input_names
        assert "temperature" in input_names
        assert "n_iterations" in input_names
        assert "is_mpdm" in input_names

        # Check outputs
        output_calls = [call for call in spec.output.call_args_list]
        output_names = [call[0][0] for call in output_calls]

        assert "output_mps" in output_names

    def test_thermal_prop_template_context(self, sho_model, sho_mpo):
        """ThermalPropCalcJob should provide correct template context."""
        from aiida_renormalizer.calculations.composite.thermal_prop import ThermalPropCalcJob

        model_data = ModelData.from_model(sho_model)
        mpo_data = MPOData.from_mpo(sho_mpo, model_data)

        calcjob = _make_calcjob(ThermalPropCalcJob, {
            "model": model_data,
            "mpo": mpo_data,
            "temperature": orm.Float(1.0),
            "n_iterations": orm.Int(10),
        })

        context = calcjob._get_template_context()

        assert "n_iterations" in context
        assert context["n_iterations"] == 10
