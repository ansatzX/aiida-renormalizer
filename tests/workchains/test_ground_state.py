"""Unit tests for GroundStateWorkChain."""
import pytest
from unittest.mock import Mock, MagicMock, patch

from aiida import orm

from aiida_renormalizer.workchains.ground_state import GroundStateWorkChain
from tests.workchains.conftest import make_workchain, Namespace


def test_ground_state_strategy_checks():
    """Test strategy selection methods."""
    wc = make_workchain(GroundStateWorkChain)

    # Test DMRG strategy
    wc.inputs = Namespace(strategy=orm.Str("dmrg"))
    assert wc.use_dmrg() is True
    assert wc.use_imag_time() is False

    # Test ImagTime strategy
    wc.inputs = Namespace(strategy=orm.Str("imag_time"))
    assert wc.use_dmrg() is False
    assert wc.use_imag_time() is True


def test_ground_state_run_dmrg():
    """Test DMRG strategy execution."""
    wc = make_workchain(GroundStateWorkChain)

    wc.inputs = Namespace(
        model=Mock(),
        code=Mock(),
        strategy=orm.Str("dmrg"),
    )
    wc.ctx = Namespace()

    # Run DMRG
    result = wc.run_dmrg()

    # Check that submit was called with DMRGCalcJob
    assert wc.submit.called
    assert result is not None


def test_ground_state_run_imag_time():
    """Test imaginary time strategy execution."""
    wc = make_workchain(GroundStateWorkChain)

    wc.inputs = Namespace(
        model=Mock(),
        code=Mock(),
        strategy=orm.Str("imag_time"),
    )
    wc.ctx = Namespace()

    # Run ImagTime
    result = wc.run_imag_time()

    # Check that submit was called with ImagTimeCalcJob
    assert wc.submit.called
    assert result is not None


def test_ground_state_inspect_dmrg_success():
    """Test inspect_dmrg with successful calculation."""
    wc = make_workchain(GroundStateWorkChain)

    # Mock successful calculation
    params = Mock()
    params.get_dict.return_value = {"energy": -1.5}

    mock_mps = Mock()
    calc = Mock()
    calc.is_finished_ok = True
    calc.outputs = Namespace(
        output_mps=mock_mps,
        output_parameters=params,
    )

    wc.ctx = Namespace(ground_state_calc=calc)

    # Inspect
    result = wc.inspect_dmrg()

    # Check results
    assert result is None  # No error
    assert wc.ctx.ground_state == mock_mps
    assert wc.ctx.energy == -1.5


def test_ground_state_inspect_dmrg_failure():
    """Test inspect_dmrg with failed calculation."""
    wc = make_workchain(GroundStateWorkChain)

    # Mock failed calculation
    calc = Mock()
    calc.is_finished_ok = False
    calc.exit_status = 300

    wc.ctx = Namespace(ground_state_calc=calc)

    # Inspect
    result = wc.inspect_dmrg()

    # Should return error
    assert result == wc.exit_codes.ERROR_CALCULATION_FAILED


def test_ground_state_inspect_imag_time_success():
    """Test inspect_imag_time with successful calculation."""
    wc = make_workchain(GroundStateWorkChain)

    # Mock successful calculation
    params = Mock()
    params.get_dict.return_value = {"energy": -1.3}

    mock_mps = Mock()
    calc = Mock()
    calc.is_finished_ok = True
    calc.outputs = Namespace(
        output_mps=mock_mps,
        output_parameters=params,
    )

    wc.ctx = Namespace(ground_state_calc=calc)

    # Inspect
    result = wc.inspect_imag_time()

    # Check results
    assert result is None
    assert wc.ctx.ground_state == mock_mps
    assert wc.ctx.energy == -1.3


def test_ground_state_unsupported_strategy():
    """Test unsupported strategy error."""
    wc = make_workchain(GroundStateWorkChain)

    wc.inputs = Namespace(strategy=orm.Str("invalid_strategy"))

    # Call
    result = wc.unsupported_strategy()

    # Should return error
    assert result == wc.exit_codes.ERROR_STRATEGY_NOT_SUPPORTED


def test_ground_state_finalize():
    """Test finalize method."""
    wc = make_workchain(GroundStateWorkChain)

    wc.inputs = Namespace(strategy=orm.Str("dmrg"))
    wc.ctx = Namespace(
        ground_state=Mock(),
        energy=-1.5,
        calc_params={"iterations": 10},
    )

    # Finalize
    wc.finalize()

    # Check outputs
    assert wc.out.call_count >= 3  # ground_state, energy, output_parameters


def test_ground_state_beta_default():
    """Test that beta defaults to high value for ground state."""
    wc = make_workchain(GroundStateWorkChain)

    wc.inputs = Namespace(
        model=Mock(),
        code=Mock(),
        strategy=orm.Str("imag_time"),
    )
    wc.ctx = Namespace()

    # Track what was submitted via self.submit
    submitted_inputs = {}

    def capture_submit(cls, **kwargs):
        submitted_inputs.update(kwargs)
        return Mock()

    wc.submit = capture_submit

    wc.run_imag_time()

    # Check that beta was set
    assert "beta" in submitted_inputs
    assert submitted_inputs["beta"].value == 100.0  # High beta for ground state
