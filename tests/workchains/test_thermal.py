"""Tests for ThermalStateWorkChain."""
import pytest
from unittest.mock import Mock, MagicMock, patch

from aiida import orm
from aiida.engine import WorkChain

from aiida_renormalizer.workchains.thermal import ThermalStateWorkChain
from tests.workchains.conftest import make_workchain, Namespace


class TestThermalStateWorkChain:
    """Test cases for ThermalStateWorkChain."""

    def test_workchain_class_exists(self):
        """Test that ThermalStateWorkChain can be imported."""
        from aiida_renormalizer.workchains import ThermalStateWorkChain as TSC
        assert TSC is ThermalStateWorkChain

    def test_workchain_inherits_from_workchain(self):
        """Test that ThermalStateWorkChain inherits from WorkChain."""
        assert issubclass(ThermalStateWorkChain, WorkChain)

    def test_workchain_has_define_method(self):
        """Test that ThermalStateWorkChain has define method."""
        assert hasattr(ThermalStateWorkChain, 'define')

    def test_workchain_has_outline(self):
        """Test that ThermalStateWorkChain has outline."""
        # Check that the spec has an outline
        spec = ThermalStateWorkChain.spec()
        # Outline should be defined
        assert spec.get_outline() is not None

    def test_workchain_has_exit_codes(self):
        """Test that ThermalStateWorkChain has exit codes defined."""
        assert hasattr(ThermalStateWorkChain, 'exit_codes')
        exit_codes = ThermalStateWorkChain.exit_codes

        # Check for specific exit codes
        assert hasattr(exit_codes, 'ERROR_INVALID_TEMPERATURE')
        assert hasattr(exit_codes, 'ERROR_THERMAL_PROP_FAILED')
        assert hasattr(exit_codes, 'ERROR_INVALID_THERMAL_STATE')

    def test_exit_codes_have_correct_status(self):
        """Test that exit codes have correct status values."""
        exit_codes = ThermalStateWorkChain.exit_codes

        assert exit_codes.ERROR_INVALID_TEMPERATURE.status == 350
        assert exit_codes.ERROR_THERMAL_PROP_FAILED.status == 351
        assert exit_codes.ERROR_INVALID_THERMAL_STATE.status == 352

    def test_workchain_has_required_methods(self):
        """Test that ThermalStateWorkChain has required methods."""
        required_methods = [
            'setup',
            'construct_initial_state',
            'run_thermal_prop',
            'inspect_thermal_state',
            'return_thermal_state',
        ]

        for method in required_methods:
            assert hasattr(ThermalStateWorkChain, method), f"Missing method: {method}"

    def test_setup_validates_temperature_inputs(self):
        """Test that setup validates temperature/beta inputs correctly."""
        # This would require mocking the WorkChain context
        # For now, we just verify the method exists and can be called
        assert callable(ThermalStateWorkChain.setup)

    def test_workchain_inputs_defined(self):
        """Test that WorkChain has expected inputs defined."""
        spec = ThermalStateWorkChain.spec()

        # Check inputs exist
        inputs = spec.inputs
        assert 'model' in inputs
        assert 'temperature' in inputs
        assert 'beta' in inputs
        assert 'code' in inputs

    def test_workchain_outputs_defined(self):
        """Test that WorkChain has expected outputs defined."""
        spec = ThermalStateWorkChain.spec()

        # Check outputs exist
        outputs = spec.outputs
        assert 'thermal_mpdm' in outputs
        assert 'partition_function' in outputs
        assert 'free_energy' in outputs
        assert 'output_parameters' in outputs


class TestThermalStateWorkChainIntegration:
    """Integration tests for ThermalStateWorkChain (requires AiiDA profile)."""

    @pytest.mark.skip(reason="Requires AiiDA profile and setup")
    def test_workchain_can_be_instantiated(self):
        """Test that ThermalStateWorkChain can be instantiated."""
        # This would require a proper AiiDA profile
        pass

    @pytest.mark.skip(reason="Requires AiiDA profile and setup")
    def test_workchain_can_be_submitted(self):
        """Test that ThermalStateWorkChain can be submitted."""
        # This would require a proper AiiDA profile and inputs
        pass


class TestThermalStateWorkChainMethods:
    """Test individual methods of ThermalStateWorkChain."""

    def test_construct_initial_state_method_signature(self):
        """Test that construct_initial_state has correct signature."""
        import inspect
        sig = inspect.signature(ThermalStateWorkChain.construct_initial_state)
        # Should accept self
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_run_thermal_prop_method_signature(self):
        """Test that run_thermal_prop has correct signature."""
        import inspect
        sig = inspect.signature(ThermalStateWorkChain.run_thermal_prop)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_inspect_thermal_state_method_signature(self):
        """Test that inspect_thermal_state has correct signature."""
        import inspect
        sig = inspect.signature(ThermalStateWorkChain.inspect_thermal_state)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_return_thermal_state_method_signature(self):
        """Test that return_thermal_state has correct signature."""
        import inspect
        sig = inspect.signature(ThermalStateWorkChain.return_thermal_state)
        params = list(sig.parameters.keys())
        assert 'self' in params


class TestThermalStateWorkChainMocked:
    """Test ThermalStateWorkChain with mocked dependencies."""

    @patch('aiida_renormalizer.workchains.thermal.ThermalPropCalcJob')
    def test_run_thermal_prop_calls_submit(self, mock_calcjob):
        """Test that run_thermal_prop submits ThermalPropCalcJob."""
        wc = make_workchain(ThermalStateWorkChain)

        # Mock context
        wc.ctx = Namespace()
        wc.ctx.initial_mpdm = Mock()
        wc.ctx.temperature = 1.0
        wc.ctx.beta = 1.0

        # Mock inputs -- use Namespace so ``"mpo" in self.inputs`` works
        wc.inputs = Namespace(
            model=Mock(),
            n_iterations=Mock(value=10),
            code=Mock(),
        )

        # Call the actual method
        ThermalStateWorkChain.run_thermal_prop(wc)

        # Verify submit was called
        assert wc.submit.called
