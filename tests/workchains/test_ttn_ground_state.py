"""Unit tests for TtnGroundStateWorkChain."""
import pytest
from unittest.mock import Mock, MagicMock, patch

from aiida import orm

from aiida_renormalizer.workchains.ttn_ground_state import TtnGroundStateWorkChain
from tests.workchains.conftest import make_workchain, Namespace


def test_ttn_ground_state_run_optimization():
    """Test TTN optimization execution."""
    wc = make_workchain(TtnGroundStateWorkChain)

    # Use Namespace so ``"key" in self.inputs`` works correctly
    wc.inputs = Namespace(
        basis_tree=Mock(),
        ttno=Mock(),
        code=Mock(),
    )

    # Run optimization -- self.submit is already mocked by make_workchain
    result = TtnGroundStateWorkChain.run_optimization(wc)

    # Check that submit was called with OptimizeTtnsCalcJob
    assert wc.submit.called
    assert result is not None


def test_ttn_ground_state_inspect_success():
    """Test inspect_optimization with successful calculation."""
    wc = make_workchain(TtnGroundStateWorkChain)

    # Mock successful calculation
    calc = Mock()
    calc.is_finished_ok = True

    # Mock outputs using Namespace so ``"key" in calc.outputs`` works
    calc.outputs = Namespace(
        output_ttns=Mock(),
        output_parameters=Mock(
            get_dict=Mock(return_value={"energy": -1.5, "iterations": 10})
        ),
    )

    wc.ctx = Namespace()
    wc.ctx.ground_state_calc = calc

    # Inspect
    result = TtnGroundStateWorkChain.inspect_optimization(wc)

    # Check results
    assert result is None  # No error
    assert wc.ctx.ground_state == calc.outputs.output_ttns
    assert wc.ctx.energy == -1.5


def test_ttn_ground_state_inspect_not_converged():
    """Test inspect_optimization with non-converged calculation."""
    wc = make_workchain(TtnGroundStateWorkChain)

    # Mock non-converged calculation
    calc = Mock()
    calc.is_finished_ok = False
    calc.exit_status = 300  # ERROR_NOT_CONVERGED

    wc.ctx = Namespace()
    wc.ctx.ground_state_calc = calc
    wc.exit_codes = TtnGroundStateWorkChain.exit_codes

    # Inspect
    result = TtnGroundStateWorkChain.inspect_optimization(wc)

    # Should return error
    assert result == TtnGroundStateWorkChain.exit_codes.ERROR_NOT_CONVERGED


def test_ttn_ground_state_inspect_failure():
    """Test inspect_optimization with failed calculation."""
    wc = make_workchain(TtnGroundStateWorkChain)

    # Mock failed calculation
    calc = Mock()
    calc.is_finished_ok = False
    calc.exit_status = 301

    wc.ctx = Namespace()
    wc.ctx.ground_state_calc = calc
    wc.exit_codes = TtnGroundStateWorkChain.exit_codes

    # Inspect
    result = TtnGroundStateWorkChain.inspect_optimization(wc)

    # Should return error
    assert result == TtnGroundStateWorkChain.exit_codes.ERROR_CALCULATION_FAILED


def test_ttn_ground_state_inspect_energy_trajectory():
    """Test energy extraction from trajectory."""
    wc = make_workchain(TtnGroundStateWorkChain)

    # Mock successful calculation with energy trajectory
    calc = Mock()
    calc.is_finished_ok = True
    calc.outputs = Namespace(
        output_ttns=Mock(),
        output_parameters=Mock(
            get_dict=Mock(return_value={"energies": [-1.0, -1.4, -1.5]})
        ),
    )

    wc.ctx = Namespace()
    wc.ctx.ground_state_calc = calc

    # Inspect
    result = TtnGroundStateWorkChain.inspect_optimization(wc)

    # Check that last energy is used
    assert result is None
    assert wc.ctx.energy == -1.5


def test_ttn_ground_state_finalize():
    """Test finalize method."""
    wc = make_workchain(TtnGroundStateWorkChain)

    wc.ctx = Namespace()
    wc.ctx.ground_state = Mock()
    wc.ctx.energy = -1.5
    wc.ctx.calc_params = {"iterations": 10}

    # Finalize
    TtnGroundStateWorkChain.finalize(wc)

    # Check outputs
    assert wc.out.call_count >= 3  # ground_state, energy, output_parameters


def test_ttn_ground_state_with_initial_ttns():
    """Test that initial_ttns is passed when provided."""
    wc = make_workchain(TtnGroundStateWorkChain)

    wc.inputs = Namespace(
        basis_tree=Mock(),
        ttno=Mock(),
        initial_ttns=Mock(),
        code=Mock(),
    )

    # Run optimization
    TtnGroundStateWorkChain.run_optimization(wc)

    # Check that submit was called and initial_ttns was included
    assert wc.submit.called
    # The second positional arg is **inputs dict -- check the kwargs
    call_kwargs = wc.submit.call_args
    # submit(OptimizeTtnsCalcJob, **inputs) -- kwargs should have initial_ttns
    assert "initial_ttns" in call_kwargs.kwargs
