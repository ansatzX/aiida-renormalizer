"""Unit tests for RenoRestartWorkChain."""
import pytest
from unittest.mock import Mock, MagicMock

from aiida import orm
from aiida.engine import ProcessHandlerReport

from aiida_renormalizer.workchains.restart import RenoRestartWorkChain
from aiida_renormalizer.calculations.composite.dmrg import DMRGCalcJob
from tests.workchains.conftest import make_workchain, Namespace


def _make_restart_workchain():
    """Create a testable RenoRestartWorkChain subclass instance.

    The ``@process_handler`` decorator accesses
    ``instance.node.base.extras.get(instance._considered_handlers_extra, [])``
    and ``instance.node.base.extras.set(...)``.  We mock ``_node`` with a
    dict-backed extras helper so the decorator works end-to-end.
    """

    class TestRestartWorkChain(RenoRestartWorkChain):
        _process_class = DMRGCalcJob

    wc = make_workchain(TestRestartWorkChain)

    # Build a mock node whose base.extras behaves like a real dict store.
    extras_store: dict = {}

    mock_extras = Mock()
    mock_extras.get = lambda key, default=None: extras_store.get(key, default)
    mock_extras.set = lambda key, value: extras_store.__setitem__(key, value)

    mock_base = Mock()
    mock_base.extras = mock_extras

    mock_node = Mock()
    mock_node.base = mock_base

    # Assign as private attr (bypasses property descriptor via object.__setattr__)
    object.__setattr__(wc, "_node", mock_node)

    # Also wire the class-level attribute expected by BaseRestartWorkChain
    wc._considered_handlers_extra = "considered_handlers"

    # Initialize the extras store with one open sub-list (like inspect_process does)
    extras_store["considered_handlers"] = [[]]

    return wc, TestRestartWorkChain


def test_restart_workchain_setup():
    """Test RenoRestartWorkChain setup method."""
    # Create a concrete test class
    class TestRestartWorkChain(RenoRestartWorkChain):
        _process_class = DMRGCalcJob

    # Note: Cannot instantiate directly in test, just verify the class structure
    assert hasattr(TestRestartWorkChain, "setup")
    assert hasattr(TestRestartWorkChain, "results")
    assert hasattr(TestRestartWorkChain, "handle_not_converged")
    assert hasattr(TestRestartWorkChain, "handle_physical_validation")


def test_handle_not_converged_with_config():
    """Test handle_not_converged with bond dimension adjustment."""
    wc, cls = _make_restart_workchain()

    # Set up ctx.inputs as a real dict (the handler uses dict 'in' operator)
    wc.ctx.inputs = {
        "config": orm.Dict({"M_max": 100})
    }
    wc.exit_codes = cls.exit_codes

    # Create mock node with ERROR_NOT_CONVERGED exit status
    node = Mock()
    node.exit_status = 300  # ERROR_NOT_CONVERGED

    # Call handler directly (unbound, passing wc as self)
    result = cls.handle_not_converged(wc, node)

    # Check that bond dimension was increased
    assert result is not None
    assert isinstance(result, ProcessHandlerReport)
    assert wc.ctx.inputs["config"]["M_max"] == 150  # 100 * 1.5


def test_handle_not_converged_without_config():
    """Test handle_not_converged without bond dimension."""
    wc, cls = _make_restart_workchain()

    # Mock context without config
    wc.ctx.inputs = {}
    wc.exit_codes = cls.exit_codes

    # Create mock node
    node = Mock()
    node.exit_status = 300

    # Call handler
    result = cls.handle_not_converged(wc, node)

    # Should return error
    assert result is not None
    assert result.exit_code == cls.exit_codes.ERROR_MAXIMUM_ITERATIONS_EXCEEDED


def test_handle_physical_validation_with_dt():
    """Test handle_physical_validation with time step adjustment."""
    wc, cls = _make_restart_workchain()

    # Mock context with dt
    wc.ctx.inputs = {
        "dt": orm.Float(0.1)
    }
    wc.exit_codes = cls.exit_codes

    # Create mock node
    node = Mock()
    node.exit_status = 310  # ERROR_PHYSICAL_VALIDATION

    # Call handler
    result = cls.handle_physical_validation(wc, node)

    # Check that time step was decreased
    assert result is not None
    assert isinstance(result, ProcessHandlerReport)
    assert wc.ctx.inputs["dt"].value == 0.05  # 0.1 / 2


def test_handle_physical_validation_with_config_dt():
    """Test handle_physical_validation with config dt."""
    wc, cls = _make_restart_workchain()

    # Mock context with config
    wc.ctx.inputs = {
        "config": orm.Dict({"dt": 0.1})
    }
    wc.exit_codes = cls.exit_codes

    # Create mock node
    node = Mock()
    node.exit_status = 310

    # Call handler
    result = cls.handle_physical_validation(wc, node)

    # Check that config time step was decreased
    assert result is not None
    assert wc.ctx.inputs["config"]["dt"] == 0.05


def test_handle_physical_validation_no_dt():
    """Test handle_physical_validation without time step."""
    wc, cls = _make_restart_workchain()

    # Mock context without dt
    wc.ctx.inputs = {}
    wc.exit_codes = cls.exit_codes

    # Create mock node
    node = Mock()
    node.exit_status = 310

    # Call handler
    result = cls.handle_physical_validation(wc, node)

    # Should return error
    assert result is not None
    assert result.exit_code == cls.exit_codes.ERROR_PROCESS_FAILURE


def test_results_method():
    """Test results method output collection."""
    wc, cls = _make_restart_workchain()

    # Mock context with successful calculation
    wc.ctx.iteration = 2

    # Mock the last child with outputs (results uses node.outputs which
    # is iterated as ``for label in node.outputs``)
    last_child = Mock()
    last_child.outputs = {
        "output_mps": Mock(),
        "output_parameters": Mock(),
    }
    wc.ctx.children = [Mock(), last_child]

    # Call results
    cls.results(wc)

    # Check that outputs were forwarded
    assert wc.out.call_count == 2
