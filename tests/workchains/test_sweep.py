"""Tests for parameter sweep WorkChains."""
import pytest
from unittest.mock import Mock, MagicMock, patch

import numpy as np

from aiida import orm
from aiida.engine import WorkChain

from aiida_renormalizer.workchains.sweep import (
    ParameterSweepWorkChain,
    TemperatureSweepWorkChain,
    BondDimensionSweepWorkChain,
    FrequencySweepWorkChain,
)


class TestParameterSweepWorkChain:
    """Test cases for ParameterSweepWorkChain."""

    def test_workchain_class_exists(self):
        """Test that ParameterSweepWorkChain can be imported."""
        from aiida_renormalizer.workchains import ParameterSweepWorkChain as PSC
        assert PSC is ParameterSweepWorkChain

    def test_workchain_inherits_from_workchain(self):
        """Test that ParameterSweepWorkChain inherits from WorkChain."""
        assert issubclass(ParameterSweepWorkChain, WorkChain)

    def test_workchain_has_define_method(self):
        """Test that ParameterSweepWorkChain has define method."""
        assert hasattr(ParameterSweepWorkChain, 'define')

    def test_workchain_has_outline(self):
        """Test that ParameterSweepWorkChain has outline."""
        spec = ParameterSweepWorkChain.spec()
        assert spec.get_outline() is not None

    def test_workchain_has_exit_codes(self):
        """Test that ParameterSweepWorkChain has exit codes defined."""
        assert hasattr(ParameterSweepWorkChain, 'exit_codes')
        exit_codes = ParameterSweepWorkChain.exit_codes

        # Check for specific exit codes
        assert hasattr(exit_codes, 'ERROR_INVALID_SWEEP_PARAMETERS')
        assert hasattr(exit_codes, 'ERROR_CALCULATION_FAILED')
        assert hasattr(exit_codes, 'ERROR_AGGREGATION_FAILED')

    def test_exit_codes_have_correct_status(self):
        """Test that exit codes have correct status values."""
        exit_codes = ParameterSweepWorkChain.exit_codes

        assert exit_codes.ERROR_INVALID_SWEEP_PARAMETERS.status == 380
        assert exit_codes.ERROR_CALCULATION_FAILED.status == 381
        assert exit_codes.ERROR_AGGREGATION_FAILED.status == 382

    def test_workchain_has_required_methods(self):
        """Test that ParameterSweepWorkChain has required methods."""
        required_methods = [
            'setup',
            'launch_sweep',
            'collect_results',
            'aggregate_results',
        ]

        for method in required_methods:
            assert hasattr(ParameterSweepWorkChain, method), f"Missing method: {method}"

    def test_workchain_inputs_defined(self):
        """Test that WorkChain has expected inputs defined."""
        spec = ParameterSweepWorkChain.spec()

        # Check inputs exist
        inputs = spec.inputs
        assert 'base_inputs' in inputs
        assert 'parameter_name' in inputs
        assert 'parameter_values' in inputs
        assert 'workchain_class' in inputs

    def test_workchain_outputs_defined(self):
        """Test that WorkChain has expected outputs defined."""
        spec = ParameterSweepWorkChain.spec()

        # Check outputs exist
        outputs = spec.outputs
        assert 'sweep_results' in outputs
        assert 'output_parameters' in outputs


class TestTemperatureSweepWorkChain:
    """Test cases for TemperatureSweepWorkChain."""

    def test_workchain_class_exists(self):
        """Test that TemperatureSweepWorkChain can be imported."""
        from aiida_renormalizer.workchains import TemperatureSweepWorkChain as TSC
        assert TSC is TemperatureSweepWorkChain

    def test_workchain_inherits_from_parameter_sweep(self):
        """Test that TemperatureSweepWorkChain inherits from ParameterSweepWorkChain."""
        assert issubclass(TemperatureSweepWorkChain, ParameterSweepWorkChain)

    def test_workchain_has_define_method(self):
        """Test that TemperatureSweepWorkChain has define method."""
        assert hasattr(TemperatureSweepWorkChain, 'define')

    def test_workchain_has_required_methods(self):
        """Test that TemperatureSweepWorkChain has required methods."""
        required_methods = [
            'setup',
            'aggregate_results',
        ]

        for method in required_methods:
            assert hasattr(TemperatureSweepWorkChain, method), f"Missing method: {method}"

    def test_workchain_has_temperatures_input(self):
        """Test that TemperatureSweepWorkChain has temperatures input."""
        spec = TemperatureSweepWorkChain.spec()

        # Check inputs exist
        inputs = spec.inputs
        assert 'temperatures' in inputs


class TestBondDimensionSweepWorkChain:
    """Test cases for BondDimensionSweepWorkChain."""

    def test_workchain_class_exists(self):
        """Test that BondDimensionSweepWorkChain can be imported."""
        from aiida_renormalizer.workchains import BondDimensionSweepWorkChain as BDSC
        assert BDSC is BondDimensionSweepWorkChain

    def test_workchain_inherits_from_parameter_sweep(self):
        """Test that BondDimensionSweepWorkChain inherits from ParameterSweepWorkChain."""
        assert issubclass(BondDimensionSweepWorkChain, ParameterSweepWorkChain)

    def test_workchain_has_define_method(self):
        """Test that BondDimensionSweepWorkChain has define method."""
        assert hasattr(BondDimensionSweepWorkChain, 'define')

    def test_workchain_has_m_values_input(self):
        """Test that BondDimensionSweepWorkChain has m_values input."""
        spec = BondDimensionSweepWorkChain.spec()

        # Check inputs exist
        inputs = spec.inputs
        assert 'm_values' in inputs

    def test_workchain_has_aggregate_results_method(self):
        """Test that BondDimensionSweepWorkChain has aggregate_results method."""
        assert hasattr(BondDimensionSweepWorkChain, 'aggregate_results')


class TestFrequencySweepWorkChain:
    """Test cases for FrequencySweepWorkChain."""

    def test_workchain_class_exists(self):
        """Test that FrequencySweepWorkChain can be imported."""
        from aiida_renormalizer.workchains import FrequencySweepWorkChain as FSC
        assert FSC is FrequencySweepWorkChain

    def test_workchain_inherits_from_parameter_sweep(self):
        """Test that FrequencySweepWorkChain inherits from ParameterSweepWorkChain."""
        assert issubclass(FrequencySweepWorkChain, ParameterSweepWorkChain)

    def test_workchain_has_define_method(self):
        """Test that FrequencySweepWorkChain has define method."""
        assert hasattr(FrequencySweepWorkChain, 'define')

    def test_workchain_has_frequencies_input(self):
        """Test that FrequencySweepWorkChain has frequencies input."""
        spec = FrequencySweepWorkChain.spec()

        # Check inputs exist
        inputs = spec.inputs
        assert 'frequencies' in inputs

    def test_workchain_has_aggregate_results_method(self):
        """Test that FrequencySweepWorkChain has aggregate_results method."""
        assert hasattr(FrequencySweepWorkChain, 'aggregate_results')


class TestSweepWorkChainIntegration:
    """Integration tests for sweep WorkChains (requires AiiDA profile)."""

    @pytest.mark.skip(reason="Requires AiiDA profile and setup")
    def test_temperature_sweep_can_be_submitted(self):
        """Test that TemperatureSweepWorkChain can be submitted."""
        pass

    @pytest.mark.skip(reason="Requires AiiDA profile and setup")
    def test_bond_dimension_sweep_can_be_submitted(self):
        """Test that BondDimensionSweepWorkChain can be submitted."""
        pass

    @pytest.mark.skip(reason="Requires AiiDA profile and setup")
    def test_frequency_sweep_can_be_submitted(self):
        """Test that FrequencySweepWorkChain can be submitted."""
        pass


class TestParameterSweepWorkChainMethods:
    """Test individual methods of ParameterSweepWorkChain."""

    def test_setup_method_signature(self):
        """Test that setup has correct signature."""
        import inspect
        sig = inspect.signature(ParameterSweepWorkChain.setup)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_launch_sweep_method_signature(self):
        """Test that launch_sweep has correct signature."""
        import inspect
        sig = inspect.signature(ParameterSweepWorkChain.launch_sweep)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_collect_results_method_signature(self):
        """Test that collect_results has correct signature."""
        import inspect
        sig = inspect.signature(ParameterSweepWorkChain.collect_results)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_aggregate_results_method_signature(self):
        """Test that aggregate_results has correct signature."""
        import inspect
        sig = inspect.signature(ParameterSweepWorkChain.aggregate_results)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_wrap_parameter_value_method(self):
        """Test _wrap_parameter_value method."""
        workchain = Mock(spec=ParameterSweepWorkChain)

        # Test different types
        result = ParameterSweepWorkChain._wrap_parameter_value(workchain, 1.5)
        assert isinstance(result, orm.Float)

        result = ParameterSweepWorkChain._wrap_parameter_value(workchain, 10)
        assert isinstance(result, orm.Int)

        result = ParameterSweepWorkChain._wrap_parameter_value(workchain, "test")
        assert isinstance(result, orm.Str)


class TestSweepWorkChainLogic:
    """Test logic of sweep WorkChains."""

    def test_parameter_values_extracted_correctly(self):
        """Test that parameter values are extracted from inputs."""
        workchain = Mock(spec=ParameterSweepWorkChain)
        workchain.inputs = Mock()
        workchain.inputs.parameter_name = Mock()
        workchain.inputs.parameter_name.value = "temperature"
        workchain.inputs.parameter_values = Mock()
        workchain.inputs.parameter_values.get_list = Mock(return_value=[1.0, 2.0, 3.0])
        workchain.ctx = Mock()
        workchain.report = Mock()

        ParameterSweepWorkChain.setup(workchain)

        # Verify parameter values were extracted
        assert workchain.ctx.param_values == [1.0, 2.0, 3.0]

    def test_aggregation_creates_arrays(self):
        """Test that aggregation creates numpy arrays."""
        workchain = Mock(spec=ParameterSweepWorkChain)
        workchain.ctx = Mock()
        workchain.ctx.param_name = "temperature"
        workchain.ctx.param_values = [1.0, 2.0, 3.0]
        workchain.ctx.results = [
            {'param_value': 1.0, 'output_params': {'energy': -1.0}},
            {'param_value': 2.0, 'output_params': {'energy': -2.0}},
            {'param_value': 3.0, 'output_params': {'energy': -3.0}},
        ]
        workchain.report = Mock()
        workchain.out = Mock()

        ParameterSweepWorkChain.aggregate_results(workchain)

        # Verify outputs were set
        assert workchain.out.called
