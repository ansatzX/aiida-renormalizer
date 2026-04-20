"""Focused contract tests for RenoScriptCalcJob spec."""
from __future__ import annotations

from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.scripted import RenoScriptCalcJob


def test_scripted_calcjob_uses_scripted_parser_by_default():
    spec = CalcJobProcessSpec()
    RenoScriptCalcJob.define(spec)
    assert spec.inputs["metadata"]["options"]["parser_name"].default == "reno.scripted"


def test_scripted_calcjob_declares_required_outputs():
    spec = CalcJobProcessSpec()
    RenoScriptCalcJob.define(spec)
    assert "output_parameters" in spec.outputs
    assert "output_mps" in spec.outputs
    assert "output_mpo" in spec.outputs
