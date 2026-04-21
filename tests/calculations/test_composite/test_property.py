"""Tests for PropertyCalcJob."""
from __future__ import annotations

import pytest
from aiida import orm
from aiida.common import AttributeDict

from aiida_renormalizer.data import ModelData, MPSData, MPOData, OpData


def _make_calcjob(cls, inputs_dict):
    """Create a CalcJob instance without triggering plumpy Process.__init__."""
    from plumpy.utils import AttributesFrozendict
    calcjob = object.__new__(cls)
    calcjob._parsed_inputs = AttributesFrozendict(inputs_dict)
    return calcjob


class TestPropertyCalcJob:
    """Tests for multi-observable property scanning."""

    def test_property_calcjob_defined(self):
        """PropertyCalcJob should be properly defined."""
        from aiida_renormalizer.calculations.composite.property import PropertyCalcJob

        assert hasattr(PropertyCalcJob, "_template_name")
        assert PropertyCalcJob._template_name == "property_driver.py.jinja"

    def test_property_inputs_outputs(self):
        """PropertyCalcJob should define correct inputs/outputs."""
        from aiida_renormalizer.calculations.composite.property import PropertyCalcJob
        from unittest.mock import Mock

        spec = Mock()
        spec.input = Mock()
        spec.input_namespace = Mock()
        spec.output = Mock()
        spec.options = {}

        PropertyCalcJob.define(spec)

        # Check inputs
        input_calls = [call for call in spec.input.call_args_list]
        input_names = [call[0][0] for call in input_calls]

        assert "mps" in input_names

        # Check input_namespace was called for observables
        namespace_calls = [call for call in spec.input_namespace.call_args_list]
        namespace_names = [call[0][0] for call in namespace_calls]

        assert "observables" in namespace_names

    def test_property_template_exists(self):
        """PropertyCalcJob template file should exist."""
        from pathlib import Path

        import aiida_renormalizer
        pkg_dir = Path(aiida_renormalizer.__file__).parent
        template_path = pkg_dir / "templates" / "property_driver.py.jinja"

        assert template_path.exists(), f"Template not found: {template_path}"

    def test_property_observable_namespace(self, sho_model, sho_mps, sho_mpo, artifact_storage_base):
        """PropertyCalcJob should accept multiple observables."""
        from aiida_renormalizer.calculations.composite.property import PropertyCalcJob

        model_data = ModelData.from_model(sho_model)
        mps_data = MPSData.from_mps(
            sho_mps,
            model_data,
            storage_backend="posix",
            storage_base=str(artifact_storage_base),
            relative_path="composite/property_mps.npz",
        )
        mpo_data = MPOData.from_mpo(sho_mpo, model_data)

        # Create additional observable
        from renormalizer.model import Op
        op = Op("b^\\dagger b", "v0", 1.0)
        op_data = OpData.from_op(op)

        calcjob = _make_calcjob(PropertyCalcJob, {
            "model": model_data,
            "mps": mps_data,
            "observables": AttributeDict({
                "hamiltonian": mpo_data,
                "n0": op_data,
            }),
        })

        # Just test that inputs are accepted
        assert "observables" in calcjob.inputs
        assert len(calcjob.inputs["observables"]) == 2
