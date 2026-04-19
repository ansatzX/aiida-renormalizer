"""Tests for CustomPipelineWorkChain."""
import pytest
from unittest.mock import Mock, MagicMock, patch

from aiida import orm
from aiida.engine import WorkChain

from aiida_renormalizer.workchains.custom import CustomPipelineWorkChain
from tests.workchains.conftest import make_workchain, Namespace


class TestCustomPipelineWorkChain:
    """Test cases for CustomPipelineWorkChain."""

    def test_workchain_class_exists(self):
        """Test that CustomPipelineWorkChain can be imported."""
        from aiida_renormalizer.workchains import CustomPipelineWorkChain as CPC
        assert CPC is CustomPipelineWorkChain

    def test_workchain_inherits_from_workchain(self):
        """Test that CustomPipelineWorkChain inherits from WorkChain."""
        assert issubclass(CustomPipelineWorkChain, WorkChain)

    def test_workchain_has_define_method(self):
        """Test that CustomPipelineWorkChain has define method."""
        assert hasattr(CustomPipelineWorkChain, 'define')

    def test_workchain_has_outline(self):
        """Test that CustomPipelineWorkChain has outline."""
        spec = CustomPipelineWorkChain.spec()
        assert spec.get_outline() is not None

    def test_workchain_has_exit_codes(self):
        """Test that CustomPipelineWorkChain has exit codes defined."""
        assert hasattr(CustomPipelineWorkChain, 'exit_codes')
        exit_codes = CustomPipelineWorkChain.exit_codes

        # Check for specific exit codes
        assert hasattr(exit_codes, 'ERROR_INVALID_PIPELINE')
        assert hasattr(exit_codes, 'ERROR_UNKNOWN_STEP_TYPE')
        assert hasattr(exit_codes, 'ERROR_STEP_FAILED')

    def test_exit_codes_have_correct_status(self):
        """Test that exit codes have correct status values."""
        exit_codes = CustomPipelineWorkChain.exit_codes

        assert exit_codes.ERROR_INVALID_PIPELINE.status == 370
        assert exit_codes.ERROR_UNKNOWN_STEP_TYPE.status == 371
        assert exit_codes.ERROR_STEP_FAILED.status == 372

    def test_workchain_has_required_methods(self):
        """Test that CustomPipelineWorkChain has required methods."""
        required_methods = [
            'setup',
            'has_more_steps',
            'dispatch_step',
            'collect_result',
            'finalize',
        ]

        for method in required_methods:
            assert hasattr(CustomPipelineWorkChain, method), f"Missing method: {method}"

    def test_workchain_has_dispatch_table(self):
        """Test that CustomPipelineWorkChain has dispatch table."""
        assert hasattr(CustomPipelineWorkChain, 'STEP_DISPATCH_TABLE')
        dispatch_table = CustomPipelineWorkChain.STEP_DISPATCH_TABLE

        # Check for expected step types
        assert 'apply_operator' in dispatch_table
        assert 'compress' in dispatch_table
        assert 'tdvp' in dispatch_table
        assert 'dmrg' in dispatch_table
        assert 'script' in dispatch_table

    def test_dispatch_table_has_correct_format(self):
        """Test that dispatch table entries have correct format."""
        dispatch_table = CustomPipelineWorkChain.STEP_DISPATCH_TABLE

        for step_type, entry in dispatch_table.items():
            assert isinstance(entry, tuple), f"{step_type} entry is not a tuple"
            assert len(entry) == 2, f"{step_type} entry does not have 2 elements"
            module_name, class_name = entry
            assert isinstance(module_name, str), f"{step_type} module name is not a string"
            assert isinstance(class_name, str), f"{step_type} class name is not a string"

    def test_workchain_inputs_defined(self):
        """Test that WorkChain has expected inputs defined."""
        spec = CustomPipelineWorkChain.spec()

        # Check inputs exist
        inputs = spec.inputs
        assert 'pipeline' in inputs
        assert 'model' in inputs
        assert 'code' in inputs

    def test_workchain_outputs_defined(self):
        """Test that WorkChain has expected outputs defined."""
        spec = CustomPipelineWorkChain.spec()

        # Check outputs exist
        outputs = spec.outputs
        assert 'output_parameters' in outputs


class TestCustomPipelineWorkChainIntegration:
    """Integration tests for CustomPipelineWorkChain (requires AiiDA profile)."""

    @pytest.mark.skip(reason="Requires AiiDA profile and setup")
    def test_workchain_can_be_instantiated(self):
        """Test that CustomPipelineWorkChain can be instantiated."""
        pass

    @pytest.mark.skip(reason="Requires AiiDA profile and setup")
    def test_workchain_can_be_submitted(self):
        """Test that CustomPipelineWorkChain can be submitted."""
        pass


class TestCustomPipelineWorkChainMethods:
    """Test individual methods of CustomPipelineWorkChain."""

    def test_setup_method_signature(self):
        """Test that setup has correct signature."""
        import inspect
        sig = inspect.signature(CustomPipelineWorkChain.setup)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_has_more_steps_method_signature(self):
        """Test that has_more_steps has correct signature."""
        import inspect
        sig = inspect.signature(CustomPipelineWorkChain.has_more_steps)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_dispatch_step_method_signature(self):
        """Test that dispatch_step has correct signature."""
        import inspect
        sig = inspect.signature(CustomPipelineWorkChain.dispatch_step)
        params = list(sig.parameters.keys())
        assert 'self' in params

    def test_collect_result_method_signature(self):
        """Test that collect_result has correct signature."""
        import inspect
        sig = inspect.signature(CustomPipelineWorkChain.collect_result)
        params = list(sig.parameters.keys())
        assert 'self' in params


class TestCustomPipelineWorkChainLogic:
    """Test logic of CustomPipelineWorkChain methods."""

    def test_has_more_steps_returns_true_initially(self):
        """Test that has_more_steps returns True initially."""
        wc = make_workchain(CustomPipelineWorkChain)
        wc.ctx = Namespace()
        wc.ctx.step_index = 0
        wc.ctx.pipeline = [Mock(), Mock(), Mock()]

        result = CustomPipelineWorkChain.has_more_steps(wc)
        assert result is True

    def test_has_more_steps_returns_false_when_complete(self):
        """Test that has_more_steps returns False when complete."""
        wc = make_workchain(CustomPipelineWorkChain)
        wc.ctx = Namespace()
        wc.ctx.step_index = 3
        wc.ctx.pipeline = [Mock(), Mock(), Mock()]

        result = CustomPipelineWorkChain.has_more_steps(wc)
        assert result is False

    def test_pipeline_specification_format(self):
        """Test that pipeline specification has correct format."""
        # Example pipeline specification
        pipeline = [
            {'step': 'apply_operator', 'inputs': {'operator': 'H'}},
            {'step': 'compress'},
            {'step': 'expectation', 'inputs': {'observable': 'S_z'}},
        ]

        # Validate format
        for step_spec in pipeline:
            assert 'step' in step_spec
            assert isinstance(step_spec['step'], str)
            if 'inputs' in step_spec:
                assert isinstance(step_spec['inputs'], dict)

    def test_dispatch_step_validates_step_type(self):
        """Test that dispatch_step validates step type."""
        wc = make_workchain(CustomPipelineWorkChain)

        wc.ctx = Namespace()
        wc.ctx.step_index = 0
        wc.ctx.pipeline = [{'step': 'unknown_step_type'}]

        # Provide exit_codes from the class
        wc.exit_codes = CustomPipelineWorkChain.exit_codes

        # Should return error for unknown step type
        result = CustomPipelineWorkChain.dispatch_step(wc)
        assert result == CustomPipelineWorkChain.exit_codes.ERROR_UNKNOWN_STEP_TYPE
