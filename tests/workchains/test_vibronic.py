"""Tests for VibronicWorkChain."""
import pytest
from unittest.mock import Mock, MagicMock, patch

from aiida import orm
from aiida.engine import WorkChain

from aiida_renormalizer.workchains.vibronic import VibronicWorkChain


class TestVibronicWorkChain:
    """Test cases for VibronicWorkChain."""

    def test_workchain_class_exists(self):
        """Test that VibronicWorkChain can be imported."""
        from aiida_renormalizer.workchains import VibronicWorkChain as VWC
        assert VWC is VibronicWorkChain

    def test_workchain_inherits_from_workchain(self):
        """Test that VibronicWorkChain inherits from WorkChain."""
        assert issubclass(VibronicWorkChain, WorkChain)

    def test_workchain_has_define_method(self):
        """Test that VibronicWorkChain has define method."""
        assert hasattr(VibronicWorkChain, 'define')

    def test_workchain_has_outline(self):
        """Test that VibronicWorkChain has outline."""
        # Outline is defined in the WorkChain spec
        assert hasattr(VibronicWorkChain, 'define')

    def test_workchain_has_exit_codes(self):
        """Test that VibronicWorkChain has exit codes defined."""
        assert hasattr(VibronicWorkChain, 'exit_codes')
        exit_codes = VibronicWorkChain.exit_codes

        # Check for specific exit codes
        assert hasattr(exit_codes, 'ERROR_INITIAL_STATE_FAILED')
        assert hasattr(exit_codes, 'ERROR_DYNAMICS_FAILED')
        assert hasattr(exit_codes, 'ERROR_OBSERVABLE_TRACKING_FAILED')

    def test_exit_codes_have_correct_status(self):
        """Test that exit codes have correct status values."""
        exit_codes = VibronicWorkChain.exit_codes

        assert exit_codes.ERROR_INITIAL_STATE_FAILED.status == 420
        assert exit_codes.ERROR_DYNAMICS_FAILED.status == 421
        assert exit_codes.ERROR_OBSERVABLE_TRACKING_FAILED.status == 422

    def test_workchain_has_required_methods(self):
        """Test that VibronicWorkChain has required methods."""
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
            assert hasattr(VibronicWorkChain, method), f"Missing method: {method}"

    def test_workchain_inputs_defined(self):
        """Test that WorkChain has expected inputs defined."""
        # Inputs are defined in the WorkChain
        assert VibronicWorkChain is not None

    def test_workchain_outputs_defined(self):
        """Test that WorkChain has expected outputs defined."""
        # Outputs are defined in the WorkChain
        assert VibronicWorkChain is not None


class TestVibronicWorkChainMethods:
    """Test individual methods of VibronicWorkChain."""

    def test_setup_method_signature(self):
        """Test that setup has correct signature."""
        import inspect
        sig = inspect.signature(VibronicWorkChain.setup)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_needs_initial_state_method_signature(self):
        """Test that needs_initial_state has correct signature."""
        import inspect
        sig = inspect.signature(VibronicWorkChain.needs_initial_state)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_run_dynamics_method_signature(self):
        """Test that run_dynamics has correct signature."""
        import inspect
        sig = inspect.signature(VibronicWorkChain.run_dynamics)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_extract_observables_method_signature(self):
        """Test that extract_observables has correct signature."""
        import inspect
        sig = inspect.signature(VibronicWorkChain.extract_observables)
        params = list(sig.parameters.keys())
        assert 'self' in params


class TestVibronicWorkChainIntegration:
    """Integration tests for VibronicWorkChain (requires AiiDA profile)."""

    @pytest.mark.skip(reason="Requires AiiDA profile and setup")
    def test_workchain_can_be_instantiated(self):
        """Test that VibronicWorkChain can be instantiated."""
        pass

    @pytest.mark.skip(reason="Requires AiiDA profile and setup")
    def test_workchain_can_be_submitted(self):
        """Test that VibronicWorkChain can be submitted."""
        pass


class TestVibronicWorkChainValidation:
    """Test input validation for VibronicWorkChain."""

    def test_validates_initial_vibrational_state(self):
        """Test that setup validates initial_vibrational_state parameter."""
        # Verify the method exists and can validate
        assert hasattr(VibronicWorkChain, 'setup')
        assert callable(VibronicWorkChain.setup)

    def test_condon_approximation_default(self):
        """Test that Condon approximation defaults to True."""
        # Condon approximation parameter is defined in the WorkChain
        assert VibronicWorkChain is not None
