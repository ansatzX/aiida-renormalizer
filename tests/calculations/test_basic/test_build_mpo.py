"""Tests for BuildMPOCalcJob."""
from __future__ import annotations

import pytest
from aiida import orm

from aiida_renormalizer.data import ModelData, OpData


class TestBuildMPOCalcJob:
    def test_build_mpo_calcjob_defined(self):
        """BuildMPOCalcJob should be properly defined."""
        from aiida_renormalizer.calculations.basic.build_mpo import BuildMPOCalcJob

        assert hasattr(BuildMPOCalcJob, '_template_name')
        assert BuildMPOCalcJob._template_name == 'build_mpo_driver.py.jinja'

    def test_build_mpo_inputs_outputs(self):
        """BuildMPOCalcJob should define correct inputs/outputs."""
        from aiida_renormalizer.calculations.basic.build_mpo import BuildMPOCalcJob
        from unittest.mock import Mock

        spec = Mock()
        spec.input = Mock()
        spec.output = Mock()
        spec.options = {}

        BuildMPOCalcJob.define(spec)

        # Check inputs
        input_calls = [call for call in spec.input.call_args_list]
        input_names = [call[0][0] for call in input_calls]

        assert 'op' in input_names

        # Check outputs
        output_calls = [call for call in spec.output.call_args_list]
        output_names = [call[0][0] for call in output_calls]

        assert 'output_mpo' in output_names
