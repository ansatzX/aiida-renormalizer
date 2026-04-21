"""Tests for ImagTimeCalcJob."""
from __future__ import annotations

import pytest
from aiida import orm
from aiida.common import AttributeDict

from aiida_renormalizer.data import ModelData, MPSData, MPOData


def _make_calcjob(cls, inputs_dict):
    """Create a CalcJob instance without triggering plumpy Process.__init__."""
    from plumpy.utils import AttributesFrozendict
    calcjob = object.__new__(cls)
    calcjob._parsed_inputs = AttributesFrozendict(inputs_dict)
    return calcjob


class TestImagTimeCalcJob:
    """Tests for imaginary time evolution."""

    def test_imag_time_calcjob_defined(self):
        """ImagTimeCalcJob should be properly defined."""
        from aiida_renormalizer.calculations.composite.imag_time import ImagTimeCalcJob

        assert hasattr(ImagTimeCalcJob, "_template_name")
        assert ImagTimeCalcJob._template_name == "imag_time_driver.py.jinja"

    def test_imag_time_inputs_outputs(self):
        """ImagTimeCalcJob should define correct inputs/outputs."""
        from aiida_renormalizer.calculations.composite.imag_time import ImagTimeCalcJob
        from unittest.mock import Mock

        spec = Mock()
        spec.input = Mock()
        spec.output = Mock()
        spec.exit_code = Mock()
        spec.options = {}

        ImagTimeCalcJob.define(spec)

        # Check inputs
        input_calls = [call for call in spec.input.call_args_list]
        input_names = [call[0][0] for call in input_calls]

        assert "mpo" in input_names
        assert "initial_mps" in input_names
        assert "beta" in input_names
        assert "dt" in input_names

        # Check outputs
        output_calls = [call for call in spec.output.call_args_list]
        output_names = [call[0][0] for call in output_calls]

        assert "output_mps" in output_names

    def test_imag_time_template_context(self, sho_model, sho_mpo):
        """ImagTimeCalcJob should provide correct template context."""
        from aiida_renormalizer.calculations.composite.imag_time import ImagTimeCalcJob

        model_data = ModelData.from_model(sho_model)
        mpo_data = MPOData.from_mpo(sho_mpo, model_data)

        calcjob = _make_calcjob(ImagTimeCalcJob, {
            "model": model_data,
            "mpo": mpo_data,
            "beta": orm.Float(10.0),
        })

        context = calcjob._get_template_context()

        assert "has_initial_mps" in context
        assert "has_dt" in context
        assert context["has_initial_mps"] is False
        assert context["has_dt"] is False

    def test_imag_time_with_dt(self, sho_model, sho_mpo):
        """ImagTimeCalcJob should handle custom time step."""
        from aiida_renormalizer.calculations.composite.imag_time import ImagTimeCalcJob

        model_data = ModelData.from_model(sho_model)
        mpo_data = MPOData.from_mpo(sho_mpo, model_data)

        calcjob = _make_calcjob(ImagTimeCalcJob, {
            "model": model_data,
            "mpo": mpo_data,
            "beta": orm.Float(10.0),
            "dt": orm.Float(0.05),
        })

        context = calcjob._get_template_context()

        assert context["has_dt"] is True

    def test_imag_time_template_exists(self):
        """ImagTimeCalcJob template file should exist."""
        from pathlib import Path

        import aiida_renormalizer
        pkg_dir = Path(aiida_renormalizer.__file__).parent
        template_path = pkg_dir / "templates" / "imag_time_driver.py.jinja"

        assert template_path.exists(), f"Template not found: {template_path}"
