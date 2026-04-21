"""Tests for MaxEntangledMpdmCalcJob."""
from __future__ import annotations

from aiida import orm


class TestMaxEntangledMpdmCalcJob:
    def test_inputs_outputs(self):
        from aiida_renormalizer.calculations.basic.max_entangled_mpdm import MaxEntangledMpdmCalcJob
        from unittest.mock import Mock

        spec = Mock()
        spec.input = Mock()
        spec.output = Mock()
        spec.exit_code = Mock()
        spec.options = {}

        MaxEntangledMpdmCalcJob.define(spec)

        input_names = [call[0][0] for call in spec.input.call_args_list]
        assert "model" in input_names
        assert "space" in input_names

        output_names = [call[0][0] for call in spec.output.call_args_list]
        assert "output_mps" in output_names

    def test_space_validator(self):
        from aiida_renormalizer.calculations.basic.max_entangled_mpdm import MaxEntangledMpdmCalcJob

        assert MaxEntangledMpdmCalcJob._validate_space(orm.Str("GS"), None) is None
        assert MaxEntangledMpdmCalcJob._validate_space(orm.Str("EX"), None) is None
        assert MaxEntangledMpdmCalcJob._validate_space(orm.Str("bad"), None) is not None
