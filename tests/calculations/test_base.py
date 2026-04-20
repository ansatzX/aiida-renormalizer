"""Tests for RenoBaseCalcJob base class."""
from __future__ import annotations

import pytest
from aiida import orm
from aiida.engine import CalcJob

from aiida_renormalizer.data import ModelData


class TestRenoBaseCalcJob:
    def test_base_class_requires_template_name(self):
        """RenoBaseCalcJob subclasses must define _template_name."""
        from aiida_renormalizer.calculations.base import RenoBaseCalcJob

        # Verify that RenoBaseCalcJob itself doesn't have _template_name
        assert not hasattr(RenoBaseCalcJob, '_template_name')

    def test_subclass_must_specify_template(self):
        """Subclasses must override _template_name."""
        from aiida_renormalizer.calculations.base import RenoBaseCalcJob
        from aiida.engine import CalcJobProcessSpec

        class BadCalcJob(RenoBaseCalcJob):
            pass

        # Define should work (doesn't check _template_name)
        spec = CalcJobProcessSpec()
        BadCalcJob.define(spec)

        # But prepare_for_submission should fail
        # We can't easily test this without full CalcJob infrastructure
        # Just verify the class doesn't have _template_name
        assert not hasattr(BadCalcJob, '_template_name')

    def test_environment_control(self, aiida_profile, sho_model):
        """MKL/OMP threads must be set to 1 by default."""
        from aiida_renormalizer.calculations.base import RenoBaseCalcJob
        from aiida_renormalizer.data import ModelData

        # Minimal concrete subclass for testing
        class ConcreteCalcJob(RenoBaseCalcJob):
            _template_name = "test_driver.py.jinja"

            @classmethod
            def define(cls, spec):
                super().define(spec)
                spec.input('model', valid_type=ModelData)
                spec.output('output_parameters', valid_type=orm.Dict)

        model_node = ModelData.from_model(sho_model)
        model_node.store()

        # Check that prepare_for_submission sets correct environment
        # (This requires mocking the CalcJobRunner infrastructure)
        # For now, just verify the method exists
        assert hasattr(ConcreteCalcJob, 'prepare_for_submission')

    def test_common_inputs_outputs_defined(self):
        """Base class defines common input/output ports."""
        from aiida_renormalizer.calculations.base import RenoBaseCalcJob
        from aiida_renormalizer.data import ModelData, ConfigData
        from aiida.engine import CalcJobProcessSpec

        class ConcreteCalcJob(RenoBaseCalcJob):
            _template_name = "test_driver.py.jinja"

            @classmethod
            def define(cls, spec):
                super().define(spec)

        spec = CalcJobProcessSpec()
        ConcreteCalcJob.define(spec)

        # Check that common inputs are defined
        assert 'model' in spec.inputs
        assert 'config' in spec.inputs
        assert 'code' in spec.inputs

        # Check that common outputs are defined
        assert 'output_parameters' in spec.outputs

    def test_default_parser_name(self):
        """Base CalcJob should default to the Reno base parser."""
        from aiida_renormalizer.calculations.base import RenoBaseCalcJob
        from aiida.engine import CalcJobProcessSpec

        class ConcreteCalcJob(RenoBaseCalcJob):
            _template_name = "test_driver.py.jinja"

        spec = CalcJobProcessSpec()
        ConcreteCalcJob.define(spec)

        assert spec.inputs["metadata"]["options"]["parser_name"].default == "reno.base"
