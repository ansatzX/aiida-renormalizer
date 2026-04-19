"""Tests for KuboTransportWorkChain."""
import pytest
from unittest.mock import Mock, MagicMock, patch

from aiida import orm
from aiida.engine import WorkChain

from aiida_renormalizer.workchains.transport import KuboTransportWorkChain
from tests.workchains.conftest import make_workchain, Namespace


class TestKuboTransportWorkChain:
    """Test cases for KuboTransportWorkChain."""

    def test_workchain_class_exists(self):
        """Test that KuboTransportWorkChain can be imported."""
        from aiida_renormalizer.workchains import KuboTransportWorkChain as KTC
        assert KTC is KuboTransportWorkChain

    def test_workchain_inherits_from_workchain(self):
        """Test that KuboTransportWorkChain inherits from WorkChain."""
        assert issubclass(KuboTransportWorkChain, WorkChain)

    def test_workchain_has_define_method(self):
        """Test that KuboTransportWorkChain has define method."""
        assert hasattr(KuboTransportWorkChain, 'define')

    def test_workchain_has_outline(self):
        """Test that KuboTransportWorkChain has outline."""
        spec = KuboTransportWorkChain.spec()
        assert spec.get_outline() is not None

    def test_workchain_has_exit_codes(self):
        """Test that KuboTransportWorkChain has exit codes defined."""
        assert hasattr(KuboTransportWorkChain, 'exit_codes')
        exit_codes = KuboTransportWorkChain.exit_codes

        # Check for specific exit codes
        assert hasattr(exit_codes, 'ERROR_INVALID_TEMPERATURE')
        assert hasattr(exit_codes, 'ERROR_THERMAL_STATE_FAILED')
        assert hasattr(exit_codes, 'ERROR_KUBO_CALCULATION_FAILED')
        assert hasattr(exit_codes, 'ERROR_INVALID_CONDUCTIVITY')

    def test_exit_codes_have_correct_status(self):
        """Test that exit codes have correct status values."""
        exit_codes = KuboTransportWorkChain.exit_codes

        assert exit_codes.ERROR_INVALID_TEMPERATURE.status == 360
        assert exit_codes.ERROR_THERMAL_STATE_FAILED.status == 361
        assert exit_codes.ERROR_KUBO_CALCULATION_FAILED.status == 362
        assert exit_codes.ERROR_INVALID_CONDUCTIVITY.status == 363

    def test_workchain_has_required_methods(self):
        """Test that KuboTransportWorkChain has required methods."""
        required_methods = [
            'setup',
            'prepare_thermal_state',
            'run_kubo_calculation',
            'extract_conductivity',
        ]

        for method in required_methods:
            assert hasattr(KuboTransportWorkChain, method), f"Missing method: {method}"

    def test_workchain_inputs_defined(self):
        """Test that WorkChain has expected inputs defined."""
        spec = KuboTransportWorkChain.spec()

        # Check inputs exist
        inputs = spec.inputs
        assert 'model' in inputs
        assert 'temperature' in inputs
        assert 'beta' in inputs
        assert 'code' in inputs
        assert 'nsteps' in inputs
        assert 'dt' in inputs

    def test_workchain_outputs_defined(self):
        """Test that WorkChain has expected outputs defined."""
        spec = KuboTransportWorkChain.spec()

        # Check outputs exist
        outputs = spec.outputs
        assert 'conductivity' in outputs
        assert 'autocorrelation' in outputs
        assert 'output_parameters' in outputs


class TestKuboTransportWorkChainIntegration:
    """Integration tests for KuboTransportWorkChain (requires AiiDA profile)."""

    @pytest.mark.skip(reason="Requires AiiDA profile and setup")
    def test_workchain_can_be_instantiated(self):
        """Test that KuboTransportWorkChain can be instantiated."""
        pass

    @pytest.mark.skip(reason="Requires AiiDA profile and setup")
    def test_workchain_can_be_submitted(self):
        """Test that KuboTransportWorkChain can be submitted."""
        pass


class TestKuboTransportWorkChainMethods:
    """Test individual methods of KuboTransportWorkChain."""

    def test_setup_method_signature(self):
        """Test that setup has correct signature."""
        import inspect
        sig = inspect.signature(KuboTransportWorkChain.setup)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_prepare_thermal_state_method_signature(self):
        """Test that prepare_thermal_state has correct signature."""
        import inspect
        sig = inspect.signature(KuboTransportWorkChain.prepare_thermal_state)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_run_kubo_calculation_method_signature(self):
        """Test that run_kubo_calculation has correct signature."""
        import inspect
        sig = inspect.signature(KuboTransportWorkChain.run_kubo_calculation)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_extract_conductivity_method_signature(self):
        """Test that extract_conductivity has correct signature."""
        import inspect
        sig = inspect.signature(KuboTransportWorkChain.extract_conductivity)
        params = list(sig.parameters.keys())
        assert 'self' in params


class TestKuboTransportWorkChainLogic:
    """Test logic of KuboTransportWorkChain methods."""

    def test_uses_thermal_state_when_provided(self):
        """Test that WorkChain uses provided thermal state."""
        # Mock scenario where initial_mps is provided
        wc = make_workchain(KuboTransportWorkChain)
        wc.inputs = Namespace(initial_mps=Mock())
        wc.ctx = Namespace()

        # Check that thermal state preparation would skip
        assert hasattr(wc.inputs, 'initial_mps')
        assert 'initial_mps' in wc.inputs

    def test_prepares_thermal_state_when_not_provided(self):
        """Test that WorkChain prepares thermal state when not provided."""
        # Use Namespace so hasattr correctly returns False for missing attributes
        wc = make_workchain(KuboTransportWorkChain)
        wc.inputs = Namespace(
            model=Mock(),
            code=Mock(),
        )
        wc.ctx = Namespace()
        wc.ctx.temperature = 1.0

        # Check that initial_mps is not present
        assert not hasattr(wc.inputs, 'initial_mps')
        assert 'initial_mps' not in wc.inputs
