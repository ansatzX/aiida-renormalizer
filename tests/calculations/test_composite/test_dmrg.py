"""Tests for DMRGCalcJob."""
from __future__ import annotations

import pytest
from aiida import orm
from aiida.common import AttributeDict

from aiida_renormalizer.data import ModelData, MPSData, MPOData, ConfigData


def _make_calcjob(cls, inputs_dict):
    """Create a CalcJob instance without triggering plumpy Process.__init__.

    The Process constructor validates inputs (including required ``resources``),
    which makes it impossible to set ``inputs`` after construction in unit tests.
    Using ``object.__new__`` bypasses ``__init__`` and we set ``_parsed_inputs``
    directly so the ``inputs`` property returns our test data.
    """
    from plumpy.utils import AttributesFrozendict
    calcjob = object.__new__(cls)
    calcjob._parsed_inputs = AttributesFrozendict(inputs_dict)
    return calcjob


class TestDMRGCalcJob:
    """Tests for DMRG variational optimization."""

    def test_dmrg_calcjob_defined(self):
        """DMRGCalcJob should be properly defined."""
        from aiida_renormalizer.calculations.composite.dmrg import DMRGCalcJob

        assert hasattr(DMRGCalcJob, "_template_name")
        assert DMRGCalcJob._template_name == "dmrg_driver.py.jinja"

    def test_dmrg_inputs_outputs(self):
        """DMRGCalcJob should define correct inputs/outputs."""
        from aiida_renormalizer.calculations.composite.dmrg import DMRGCalcJob
        from unittest.mock import Mock

        spec = Mock()
        spec.input = Mock()
        spec.output = Mock()
        spec.exit_code = Mock()
        spec.options = {}

        DMRGCalcJob.define(spec)

        # Check inputs
        input_calls = [call for call in spec.input.call_args_list]
        input_names = [call[0][0] for call in input_calls]

        assert "mpo" in input_names
        assert "initial_mps" in input_names
        assert "omega" in input_names

        # Check outputs
        output_calls = [call for call in spec.output.call_args_list]
        output_names = [call[0][0] for call in output_calls]

        assert "output_mps" in output_names

    def test_dmrg_template_context(self, sho_model, sho_mpo):
        """DMRGCalcJob should provide correct template context."""
        from aiida_renormalizer.calculations.composite.dmrg import DMRGCalcJob

        # Create minimal inputs
        model_data = ModelData.from_model(sho_model)
        mpo_data = MPOData.from_mpo(sho_mpo, model_data)

        # Create CalcJob instance bypassing Process.__init__
        calcjob = _make_calcjob(DMRGCalcJob, {
            "model": model_data,
            "mpo": mpo_data,
        })

        context = calcjob._get_template_context()

        assert "has_initial_mps" in context
        assert "has_omega" in context
        assert context["has_initial_mps"] is False
        assert context["has_omega"] is False

    def test_dmrg_with_initial_mps(self, sho_model, sho_mpo, sho_mps, artifact_storage_base):
        """DMRGCalcJob should handle initial MPS input."""
        from aiida_renormalizer.calculations.composite.dmrg import DMRGCalcJob

        model_data = ModelData.from_model(sho_model)
        mpo_data = MPOData.from_mpo(sho_mpo, model_data)
        mps_data = MPSData.from_mps(
            sho_mps,
            model_data,
            storage_backend="posix",
            storage_base=str(artifact_storage_base),
            relative_path="composite/dmrg_initial.npz",
        )

        calcjob = _make_calcjob(DMRGCalcJob, {
            "model": model_data,
            "mpo": mpo_data,
            "initial_mps": mps_data,
        })

        context = calcjob._get_template_context()

        assert context["has_initial_mps"] is True

    def test_dmrg_with_omega(self, sho_model, sho_mpo):
        """DMRGCalcJob should handle omega input for excited states."""
        from aiida_renormalizer.calculations.composite.dmrg import DMRGCalcJob

        model_data = ModelData.from_model(sho_model)
        mpo_data = MPOData.from_mpo(sho_mpo, model_data)

        calcjob = _make_calcjob(DMRGCalcJob, {
            "model": model_data,
            "mpo": mpo_data,
            "omega": orm.Float(1.0),
        })

        context = calcjob._get_template_context()

        assert context["has_omega"] is True

    def test_dmrg_template_exists(self):
        """DMRGCalcJob template file should exist."""
        from pathlib import Path

        # Find template relative to package
        import aiida_renormalizer
        pkg_dir = Path(aiida_renormalizer.__file__).parent
        template_path = pkg_dir / "templates" / "dmrg_driver.py.jinja"

        assert template_path.exists(), f"Template not found: {template_path}"
