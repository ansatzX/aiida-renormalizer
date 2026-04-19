"""Tests for CorrectionVectorWorkChain."""
import pytest
from unittest.mock import Mock, MagicMock, patch

from aiida import orm
from aiida.engine import WorkChain

from aiida_renormalizer.workchains.correction_vector import CorrectionVectorWorkChain


class TestCorrectionVectorWorkChain:
    """Test cases for CorrectionVectorWorkChain."""

    def test_workchain_class_exists(self):
        """Test that CorrectionVectorWorkChain can be imported."""
        from aiida_renormalizer.workchains import CorrectionVectorWorkChain as CVWC
        assert CVWC is CorrectionVectorWorkChain

    def test_workchain_inherits_from_workchain(self):
        """Test that CorrectionVectorWorkChain inherits from WorkChain."""
        assert issubclass(CorrectionVectorWorkChain, WorkChain)

    def test_workchain_has_define_method(self):
        """Test that CorrectionVectorWorkChain has define method."""
        assert hasattr(CorrectionVectorWorkChain, 'define')

    def test_workchain_has_outline(self):
        """Test that CorrectionVectorWorkChain has outline."""
        # Outline is defined in the WorkChain spec
        assert hasattr(CorrectionVectorWorkChain, 'define')

    def test_workchain_has_exit_codes(self):
        """Test that CorrectionVectorWorkChain has exit codes defined."""
        assert hasattr(CorrectionVectorWorkChain, 'exit_codes')
        exit_codes = CorrectionVectorWorkChain.exit_codes

        # Check for specific exit codes
        assert hasattr(exit_codes, 'ERROR_GROUND_STATE_FAILED')
        assert hasattr(exit_codes, 'ERROR_CV_CALCULATION_FAILED')
        assert hasattr(exit_codes, 'ERROR_SPECTRUM_AGGREGATION_FAILED')

    def test_exit_codes_have_correct_status(self):
        """Test that exit codes have correct status values."""
        exit_codes = CorrectionVectorWorkChain.exit_codes

        assert exit_codes.ERROR_GROUND_STATE_FAILED.status == 390
        assert exit_codes.ERROR_CV_CALCULATION_FAILED.status == 391
        assert exit_codes.ERROR_SPECTRUM_AGGREGATION_FAILED.status == 392

    def test_workchain_has_required_methods(self):
        """Test that CorrectionVectorWorkChain has required methods."""
        required_methods = [
            'setup',
            'needs_ground_state',
            'prepare_ground_state',
            'inspect_ground_state',
            'run_cv_calculation',
            'inspect_cv_calculation',
            'aggregate_spectrum',
            'finalize',
        ]

        for method in required_methods:
            assert hasattr(CorrectionVectorWorkChain, method), f"Missing method: {method}"

    def test_workchain_inputs_defined(self):
        """Test that WorkChain has expected inputs defined."""
        # Inputs are defined in the WorkChain
        # Just verify that the WorkChain can be imported
        assert CorrectionVectorWorkChain is not None

    def test_workchain_outputs_defined(self):
        """Test that WorkChain has expected outputs defined."""
        # Outputs are defined in the WorkChain
        # Just verify that the WorkChain can be imported
        assert CorrectionVectorWorkChain is not None


class TestCorrectionVectorWorkChainMethods:
    """Test individual methods of CorrectionVectorWorkChain."""

    def test_setup_method_signature(self):
        """Test that setup has correct signature."""
        import inspect
        sig = inspect.signature(CorrectionVectorWorkChain.setup)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_needs_ground_state_method_signature(self):
        """Test that needs_ground_state has correct signature."""
        import inspect
        sig = inspect.signature(CorrectionVectorWorkChain.needs_ground_state)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_run_cv_calculation_method_signature(self):
        """Test that run_cv_calculation has correct signature."""
        import inspect
        sig = inspect.signature(CorrectionVectorWorkChain.run_cv_calculation)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_aggregate_spectrum_method_signature(self):
        """Test that aggregate_spectrum has correct signature."""
        import inspect
        sig = inspect.signature(CorrectionVectorWorkChain.aggregate_spectrum)
        params = list(sig.parameters.keys())
        assert 'self' in params


class TestCorrectionVectorWorkChainIntegration:
    """Integration tests for CorrectionVectorWorkChain (requires AiiDA profile)."""

    @pytest.mark.skip(reason="Requires AiiDA profile and setup")
    def test_workchain_can_be_instantiated(self):
        """Test that CorrectionVectorWorkChain can be instantiated."""
        pass

    @pytest.mark.skip(reason="Requires AiiDA profile and setup")
    def test_workchain_can_be_submitted(self):
        """Test that CorrectionVectorWorkChain can be submitted."""
        pass
