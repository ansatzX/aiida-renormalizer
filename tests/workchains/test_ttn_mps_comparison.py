"""Unit tests for TtnMpsComparisonWorkChain."""
import pytest
from unittest.mock import Mock, MagicMock, patch

from aiida import orm

from aiida_renormalizer.workchains.ttn_mps_comparison import TtnMpsComparisonWorkChain
from tests.workchains.conftest import make_workchain, Namespace


def test_ttn_mps_comparison_setup():
    """Test setup method."""
    wc = make_workchain(TtnMpsComparisonWorkChain)

    wc.inputs = Namespace(calculation_type=orm.Str("ground_state"))

    # Setup
    result = TtnMpsComparisonWorkChain.setup(wc)

    # Check initialization
    assert result is None


def test_ttn_mps_comparison_setup_unsupported():
    """Test setup with unsupported calculation type."""
    wc = make_workchain(TtnMpsComparisonWorkChain)

    wc.inputs = Namespace(calculation_type=orm.Str("invalid"))
    wc.exit_codes = TtnMpsComparisonWorkChain.exit_codes

    # Setup
    result = TtnMpsComparisonWorkChain.setup(wc)

    # Should return error
    assert result == TtnMpsComparisonWorkChain.exit_codes.ERROR_UNSUPPORTED_CALCULATION


def test_ttn_mps_comparison_run_mps():
    """Test run_mps method."""
    wc = make_workchain(TtnMpsComparisonWorkChain)

    # Use Namespace so ``"key" in self.inputs`` works
    wc.inputs = Namespace(
        model=Mock(),
        mpo=Mock(),
        code=Mock(),
        calculation_type=orm.Str("ground_state"),
    )

    # Run MPS -- self.submit is already mocked by make_workchain
    result = TtnMpsComparisonWorkChain.run_mps(wc)

    assert wc.submit.called
    assert result is not None


def test_ttn_mps_comparison_inspect_mps_success():
    """Test inspect_mps with successful calculation."""
    wc = make_workchain(TtnMpsComparisonWorkChain)

    # Mock successful calculation
    calc = Mock()
    calc.is_finished_ok = True

    energy_mock = Mock()
    energy_mock.value = -1.5
    params = Mock()
    params.get_dict.return_value = {"iterations": 10, "M_max": 50}

    calc.outputs = Namespace(
        ground_state=Mock(),
        energy=energy_mock,
        output_parameters=params,
    )

    wc.inputs = Namespace(calculation_type=orm.Str("ground_state"))

    wc.ctx = Namespace()
    wc.ctx.mps_calc = calc

    # Inspect
    result = TtnMpsComparisonWorkChain.inspect_mps(wc)

    # Check results
    assert result is None  # No error
    assert wc.ctx.mps_result == calc.outputs.ground_state
    assert wc.ctx.mps_energy == -1.5


def test_ttn_mps_comparison_inspect_mps_failure():
    """Test inspect_mps with failed calculation."""
    wc = make_workchain(TtnMpsComparisonWorkChain)

    calc = Mock()
    calc.is_finished_ok = False
    calc.exit_status = 320

    wc.inputs = Namespace(calculation_type=orm.Str("ground_state"))

    wc.ctx = Namespace()
    wc.ctx.mps_calc = calc
    wc.exit_codes = TtnMpsComparisonWorkChain.exit_codes

    result = TtnMpsComparisonWorkChain.inspect_mps(wc)

    assert result == TtnMpsComparisonWorkChain.exit_codes.ERROR_MPS_FAILED


def test_ttn_mps_comparison_run_ttn():
    """Test run_ttn method."""
    wc = make_workchain(TtnMpsComparisonWorkChain)

    # Use Namespace so ``"key" in self.inputs`` works
    wc.inputs = Namespace(
        basis_tree=Mock(),
        ttno=Mock(),
        code=Mock(),
        calculation_type=orm.Str("ground_state"),
    )

    # Run TTN -- self.submit is already mocked by make_workchain
    result = TtnMpsComparisonWorkChain.run_ttn(wc)

    assert wc.submit.called
    assert result is not None


def test_ttn_mps_comparison_inspect_ttn_success():
    """Test inspect_ttn with successful calculation."""
    wc = make_workchain(TtnMpsComparisonWorkChain)

    # Mock successful calculation
    calc = Mock()
    calc.is_finished_ok = True

    energy_mock = Mock()
    energy_mock.value = -1.45
    params = Mock()
    params.get_dict.return_value = {"iterations": 8, "bond_dims": [10, 15, 20]}

    calc.outputs = Namespace(
        ground_state=Mock(),
        energy=energy_mock,
        output_parameters=params,
    )

    wc.inputs = Namespace(calculation_type=orm.Str("ground_state"))

    wc.ctx = Namespace()
    wc.ctx.ttn_calc = calc

    # Inspect
    result = TtnMpsComparisonWorkChain.inspect_ttn(wc)

    # Check results
    assert result is None
    assert wc.ctx.ttn_result == calc.outputs.ground_state
    assert wc.ctx.ttn_energy == -1.45


def test_ttn_mps_comparison_inspect_ttn_failure():
    """Test inspect_ttn with failed calculation."""
    wc = make_workchain(TtnMpsComparisonWorkChain)

    calc = Mock()
    calc.is_finished_ok = False
    calc.exit_status = 330

    wc.inputs = Namespace(calculation_type=orm.Str("ground_state"))

    wc.ctx = Namespace()
    wc.ctx.ttn_calc = calc
    wc.exit_codes = TtnMpsComparisonWorkChain.exit_codes

    result = TtnMpsComparisonWorkChain.inspect_ttn(wc)

    assert result == TtnMpsComparisonWorkChain.exit_codes.ERROR_TTN_FAILED


def test_ttn_mps_comparison_compare_results():
    """Test compare_results method."""
    wc = make_workchain(TtnMpsComparisonWorkChain)

    wc.ctx = Namespace()
    wc.ctx.mps_energy = -1.5
    wc.ctx.ttn_energy = -1.45
    wc.ctx.mps_params = {"M_max": 50, "iterations": 10}
    wc.ctx.ttn_params = {"bond_dims": [10, 15, 20], "iterations": 8}

    # Compare
    TtnMpsComparisonWorkChain.compare_results(wc)

    # Check comparison data
    assert wc.ctx.comparison["mps_energy"] == -1.5
    assert wc.ctx.comparison["ttn_energy"] == -1.45
    assert abs(wc.ctx.comparison["energy_difference"] - 0.05) < 1e-10
    assert wc.ctx.comparison["mps_iterations"] == 10
    assert wc.ctx.comparison["ttn_iterations"] == 8


def test_ttn_mps_comparison_finalize():
    """Test finalize method."""
    wc = make_workchain(TtnMpsComparisonWorkChain)

    wc.inputs = Namespace(calculation_type=orm.Str("ground_state"))

    wc.ctx = Namespace()
    wc.ctx.mps_result = Mock()
    wc.ctx.ttn_result = Mock()
    wc.ctx.comparison = {
        "mps_energy": -1.5,
        "ttn_energy": -1.45,
        "energy_difference": 0.05,
    }

    # Finalize
    TtnMpsComparisonWorkChain.finalize(wc)

    # Check outputs
    assert wc.out.call_count >= 4  # mps_result, ttn_result, comparison_data, output_parameters


def test_ttn_mps_comparison_with_initial_states():
    """Test that initial states are passed when provided."""
    wc = make_workchain(TtnMpsComparisonWorkChain)

    wc.inputs = Namespace(
        model=Mock(),
        mpo=Mock(),
        initial_mps=Mock(),
        basis_tree=Mock(),
        ttno=Mock(),
        initial_ttns=Mock(),
        code=Mock(),
        calculation_type=orm.Str("ground_state"),
    )

    # Run MPS submission
    TtnMpsComparisonWorkChain.run_mps(wc)

    # Check that initial_mps was passed in submit kwargs
    assert wc.submit.called
    call_kwargs = wc.submit.call_args
    assert "initial_mps" in call_kwargs.kwargs


def test_ttn_mps_comparison_energy_difference_calculation():
    """Test that energy difference is calculated correctly."""
    wc = make_workchain(TtnMpsComparisonWorkChain)

    wc.ctx = Namespace()
    wc.ctx.mps_energy = -2.0
    wc.ctx.ttn_energy = -1.9
    wc.ctx.mps_params = {}
    wc.ctx.ttn_params = {}

    TtnMpsComparisonWorkChain.compare_results(wc)

    # Check energy difference
    assert abs(wc.ctx.comparison["energy_difference"] - 0.1) < 1e-10
