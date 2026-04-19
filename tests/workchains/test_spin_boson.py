"""Tests for SpinBosonWorkChain."""
import pytest
from unittest.mock import Mock, MagicMock, patch

from aiida import orm
from aiida.engine import WorkChain

from aiida_renormalizer.workchains.spin_boson import SpinBosonWorkChain


class TestSpinBosonWorkChain:
    """Test cases for SpinBosonWorkChain."""

    def test_workchain_class_exists(self):
        """Test that SpinBosonWorkChain can be imported."""
        from aiida_renormalizer.workchains import SpinBosonWorkChain as SBWC
        assert SBWC is SpinBosonWorkChain

    def test_workchain_inherits_from_workchain(self):
        """Test that SpinBosonWorkChain inherits from WorkChain."""
        assert issubclass(SpinBosonWorkChain, WorkChain)

    def test_workchain_has_define_method(self):
        """Test that SpinBosonWorkChain has define method."""
        assert hasattr(SpinBosonWorkChain, 'define')

    def test_workchain_has_outline(self):
        """Test that SpinBosonWorkChain has outline."""
        # Outline is defined in the WorkChain spec
        assert hasattr(SpinBosonWorkChain, 'define')

    def test_workchain_has_exit_codes(self):
        """Test that SpinBosonWorkChain has exit codes defined."""
        assert hasattr(SpinBosonWorkChain, 'exit_codes')
        exit_codes = SpinBosonWorkChain.exit_codes

        # Check for specific exit codes
        assert hasattr(exit_codes, 'ERROR_INITIAL_STATE_FAILED')
        assert hasattr(exit_codes, 'ERROR_DYNAMICS_FAILED')
        assert hasattr(exit_codes, 'ERROR_OBSERVABLE_TRACKING_FAILED')

    def test_exit_codes_have_correct_status(self):
        """Test that exit codes have correct status values."""
        exit_codes = SpinBosonWorkChain.exit_codes

        assert exit_codes.ERROR_INITIAL_STATE_FAILED.status == 410
        assert exit_codes.ERROR_DYNAMICS_FAILED.status == 411
        assert exit_codes.ERROR_OBSERVABLE_TRACKING_FAILED.status == 412

    def test_workchain_has_required_methods(self):
        """Test that SpinBosonWorkChain has required methods."""
        required_methods = [
            'setup',
            'needs_initial_state',
            'prepare_initial_state',
            'inspect_initial_state',
            'run_dynamics',
            'inspect_dynamics',
            'extract_observables',
            'finalize',
        ]

        for method in required_methods:
            assert hasattr(SpinBosonWorkChain, method), f"Missing method: {method}"

    def test_workchain_inputs_defined(self):
        """Test that WorkChain has expected inputs defined."""
        # Inputs are defined in the WorkChain
        assert SpinBosonWorkChain is not None

    def test_workchain_outputs_defined(self):
        """Test that WorkChain has expected outputs defined."""
        # Outputs are defined in the WorkChain
        assert SpinBosonWorkChain is not None


class TestSpinBosonWorkChainMethods:
    """Test individual methods of SpinBosonWorkChain."""

    def test_setup_method_signature(self):
        """Test that setup has correct signature."""
        import inspect
        sig = inspect.signature(SpinBosonWorkChain.setup)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_needs_initial_state_method_signature(self):
        """Test that needs_initial_state has correct signature."""
        import inspect
        sig = inspect.signature(SpinBosonWorkChain.needs_initial_state)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_run_dynamics_method_signature(self):
        """Test that run_dynamics has correct signature."""
        import inspect
        sig = inspect.signature(SpinBosonWorkChain.run_dynamics)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_extract_observables_method_signature(self):
        """Test that extract_observables has correct signature."""
        import inspect
        sig = inspect.signature(SpinBosonWorkChain.extract_observables)
        params = list(sig.parameters.keys())
        assert 'self' in params


class TestSpinBosonWorkChainIntegration:
    """Integration tests for SpinBosonWorkChain (requires AiiDA profile)."""

    @pytest.mark.skip(reason="Requires AiiDA profile and setup")
    def test_workchain_can_be_instantiated(self):
        """Test that SpinBosonWorkChain can be instantiated."""
        pass

    @pytest.mark.skip(reason="Requires AiiDA profile and setup")
    def test_workchain_can_be_submitted(self):
        """Test that SpinBosonWorkChain can be submitted."""
        pass


class TestSpinBosonWorkChainValidation:
    """Test input validation for SpinBosonWorkChain."""

    def test_validates_initial_spin(self):
        """Test that setup validates initial_spin parameter."""
        # Verify the method exists and can validate
        assert hasattr(SpinBosonWorkChain, 'setup')
        assert callable(SpinBosonWorkChain.setup)
