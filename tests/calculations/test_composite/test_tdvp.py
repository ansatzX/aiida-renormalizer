"""Tests for TDVPCalcJob."""
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


class TestTDVPCalcJob:
    """Tests for TDVP real-time evolution."""

    def test_tdvp_inputs_outputs(self):
        """TDVPCalcJob should define correct inputs/outputs."""
        from aiida_renormalizer.calculations.composite.tdvp import TDVPCalcJob
        from unittest.mock import Mock

        spec = Mock()
        spec.input = Mock()
        spec.output = Mock()
        spec.exit_code = Mock()
        spec.options = {}

        TDVPCalcJob.define(spec)

        # Check inputs
        input_calls = [call for call in spec.input.call_args_list]
        input_names = [call[0][0] for call in input_calls]

        assert "mpo" in input_names
        assert "initial_mps" in input_names
        assert "total_time" in input_names
        assert "dt" in input_names
        assert "trajectory_interval" in input_names
        assert "observables" in input_names

        # Check outputs
        output_calls = [call for call in spec.output.call_args_list]
        output_names = [call[0][0] for call in output_calls]

        assert "output_mps" in output_names
        assert "trajectory" in output_names

    def test_tdvp_template_context(self, sho_model, sho_mpo, sho_mps, artifact_storage_base):
        """TDVPCalcJob should provide correct template context."""
        from aiida_renormalizer.calculations.composite.tdvp import TDVPCalcJob

        model_data = ModelData.from_model(sho_model)
        mpo_data = MPOData.from_mpo(sho_mpo, model_data)
        mps_data = MPSData.from_mps(
            sho_mps,
            model_data,
            storage_backend="posix",
            storage_base=str(artifact_storage_base),
            relative_path="composite/tdvp_context.npz",
        )

        calcjob = _make_calcjob(TDVPCalcJob, {
            "model": model_data,
            "mpo": mpo_data,
            "initial_mps": mps_data,
            "total_time": orm.Float(10.0),
            "dt": orm.Float(0.1),
            "trajectory_interval": orm.Int(10),
        })

        context = calcjob._get_template_context()

        assert "has_observables" in context
        assert context["has_observables"] is False

    def test_tdvp_with_observables(self, sho_model, sho_mpo, sho_mps, artifact_storage_base):
        """TDVPCalcJob should handle observables input."""
        from aiida_renormalizer.calculations.composite.tdvp import TDVPCalcJob

        model_data = ModelData.from_model(sho_model)
        mpo_data = MPOData.from_mpo(sho_mpo, model_data)
        mps_data = MPSData.from_mps(
            sho_mps,
            model_data,
            storage_backend="posix",
            storage_base=str(artifact_storage_base),
            relative_path="composite/tdvp_observables.npz",
        )

        calcjob = _make_calcjob(TDVPCalcJob, {
            "model": model_data,
            "mpo": mpo_data,
            "initial_mps": mps_data,
            "total_time": orm.Float(10.0),
            "dt": orm.Float(0.1),
            "trajectory_interval": orm.Int(10),
            "observables": orm.List(list=["uuid1", "uuid2"]),
        })

        context = calcjob._get_template_context()

        assert context["has_observables"] is True
