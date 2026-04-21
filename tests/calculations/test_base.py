"""Focused contract tests for RenoBaseCalcJob."""
from __future__ import annotations

from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob


class _ConcreteCalcJob(RenoBaseCalcJob):
    _template_name = "dummy_driver.py.jinja"


def test_base_calcjob_ports_and_defaults():
    spec = CalcJobProcessSpec()
    _ConcreteCalcJob.define(spec)

    assert "model" in spec.inputs
    assert "config" in spec.inputs
    assert "tn_layout" in spec.inputs
    assert "code" in spec.inputs
    assert "output_parameters" in spec.outputs
    assert "output_tn_layout" in spec.outputs


def test_base_calcjob_shared_exit_codes_present():
    spec = CalcJobProcessSpec()
    _ConcreteCalcJob.define(spec)

    assert "ERROR_OUTPUT_MISSING" in spec.exit_codes
    assert "ERROR_OUTPUT_PARSING" in spec.exit_codes
