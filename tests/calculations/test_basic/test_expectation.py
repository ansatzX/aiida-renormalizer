"""Tests for ExpectationCalcJob."""
from __future__ import annotations

import pytest
from aiida import orm

from aiida_renormalizer.data import ModelData, MpsData, MpoData


class TestExpectationCalcJob:
    def test_expectation_calcjob_defined(self):
        """ExpectationCalcJob should be properly defined."""
        from aiida_renormalizer.calculations.basic.expectation import ExpectationCalcJob

        assert hasattr(ExpectationCalcJob, '_template_name')
        assert ExpectationCalcJob._template_name == 'expectation_driver.py.jinja'

    def test_expectation_inputs_outputs(self):
        """ExpectationCalcJob should define correct inputs/outputs."""
        from aiida_renormalizer.calculations.basic.expectation import ExpectationCalcJob
        from unittest.mock import Mock

        spec = Mock()
        spec.input = Mock()
        spec.output = Mock()
        spec.options = {}

        ExpectationCalcJob.define(spec)

        # Check inputs
        input_calls = [call for call in spec.input.call_args_list]
        input_names = [call[0][0] for call in input_calls]

        assert 'mps' in input_names
        assert 'mpo' in input_names

        # output_parameters is inherited from base
