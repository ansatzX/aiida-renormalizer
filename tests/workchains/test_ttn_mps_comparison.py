"""Unit tests for TTNMPSComparisonWorkChain."""
import pytest
from unittest.mock import Mock

from aiida import orm

from aiida_renormalizer.workchains.ttn_mps_comparison import TTNMPSComparisonWorkChain
from tests.workchains.conftest import make_workchain, Namespace


def test_ttn_mps_comparison_setup():
    """Test setup method."""
    wc = make_workchain(TTNMPSComparisonWorkChain)

    wc.inputs = Namespace(calculation_type=orm.Str("ground_state"))

    # Setup
    result = TTNMPSComparisonWorkChain.setup(wc)

    # Check initialization
    assert result is None


def test_ttn_mps_comparison_setup_unsupported():
    """Test setup with unsupported calculation type."""
    wc = make_workchain(TTNMPSComparisonWorkChain)

    wc.inputs = Namespace(calculation_type=orm.Str("invalid"))
    wc.exit_codes = TTNMPSComparisonWorkChain.exit_codes

    # Setup
    result = TTNMPSComparisonWorkChain.setup(wc)

    # Should return error
    assert result == TTNMPSComparisonWorkChain.exit_codes.ERROR_UNSUPPORTED_CALCULATION


def test_ttn_mps_comparison_run_mps():
    """Test run_mps method."""
    wc = make_workchain(TTNMPSComparisonWorkChain)

    # Use Namespace so ``"key" in self.inputs`` works
    wc.inputs = Namespace(
        model=Mock(),
        mpo=Mock(),
        code=Mock(),
        calculation_type=orm.Str("ground_state"),
    )

    # Run MPS -- self.submit is already mocked by make_workchain
    TTNMPSComparisonWorkChain.run_mps(wc)
    assert wc.submit.called
    submit_kwargs = wc.submit.call_args.kwargs
    assert submit_kwargs["model"] is wc.inputs.model
    assert submit_kwargs["mpo"] is wc.inputs.mpo
    assert submit_kwargs["code"] is wc.inputs.code


def test_ttn_mps_comparison_inspect_mps_success():
    """Test inspect_mps with successful calculation."""
    wc = make_workchain(TTNMPSComparisonWorkChain)

    # Mock successful calculation
    calc = Mock()
    calc.is_finished_ok = True

    energy_mock = Mock()
    energy_mock.value = -1.5
    params = Mock()
    params.get_dict.return_value = {"iterations": 10, "M_max": 50}

    calc.outputs = Namespace(
        output_mps=Mock(),
        energy=energy_mock,
        output_parameters=params,
    )

    wc.inputs = Namespace(calculation_type=orm.Str("ground_state"))

    wc.ctx = Namespace()
    wc.ctx.mps_calc = calc

    # Inspect
    result = TTNMPSComparisonWorkChain.inspect_mps(wc)

    # Check results
    assert result is None  # No error
    assert wc.ctx.mps_result == calc.outputs.output_mps
    assert wc.ctx.mps_energy == -1.5


def test_ttn_mps_comparison_inspect_mps_failure():
    """Test inspect_mps with failed calculation."""
    wc = make_workchain(TTNMPSComparisonWorkChain)

    calc = Mock()
    calc.is_finished_ok = False
    calc.exit_status = 320

    wc.inputs = Namespace(calculation_type=orm.Str("ground_state"))

    wc.ctx = Namespace()
    wc.ctx.mps_calc = calc
    wc.exit_codes = TTNMPSComparisonWorkChain.exit_codes

    result = TTNMPSComparisonWorkChain.inspect_mps(wc)

    assert result == TTNMPSComparisonWorkChain.exit_codes.ERROR_MPS_FAILED


def test_ttn_mps_comparison_run_ttn():
    """Test run_ttn method."""
    wc = make_workchain(TTNMPSComparisonWorkChain)

    # Use Namespace so ``"key" in self.inputs`` works
    wc.inputs = Namespace(
        basis_tree=Mock(),
        ttno=Mock(),
        code=Mock(),
        calculation_type=orm.Str("ground_state"),
    )

    # Run TTN -- self.submit is already mocked by make_workchain
    TTNMPSComparisonWorkChain.run_ttn(wc)
    assert wc.submit.called
    submit_kwargs = wc.submit.call_args.kwargs
    assert submit_kwargs["basis_tree"] is wc.inputs.basis_tree
    assert submit_kwargs["ttno"] is wc.inputs.ttno
    assert submit_kwargs["code"] is wc.inputs.code


def test_ttn_mps_comparison_inspect_ttn_success():
    """Test inspect_ttn with successful calculation."""
    wc = make_workchain(TTNMPSComparisonWorkChain)

    # Mock successful calculation
    calc = Mock()
    calc.is_finished_ok = True

    energy_mock = Mock()
    energy_mock.value = -1.45
    params = Mock()
    params.get_dict.return_value = {"iterations": 8, "bond_dims": [10, 15, 20]}

    calc.outputs = Namespace(
        output_ttns=Mock(),
        energy=energy_mock,
        output_parameters=params,
    )

    wc.inputs = Namespace(calculation_type=orm.Str("ground_state"))

    wc.ctx = Namespace()
    wc.ctx.ttn_calc = calc

    # Inspect
    result = TTNMPSComparisonWorkChain.inspect_ttn(wc)

    # Check results
    assert result is None
    assert wc.ctx.ttn_result == calc.outputs.output_ttns
    assert wc.ctx.ttn_energy == -1.45


def test_ttn_mps_comparison_inspect_ttn_failure():
    """Test inspect_ttn with failed calculation."""
    wc = make_workchain(TTNMPSComparisonWorkChain)

    calc = Mock()
    calc.is_finished_ok = False
    calc.exit_status = 330

    wc.inputs = Namespace(calculation_type=orm.Str("ground_state"))

    wc.ctx = Namespace()
    wc.ctx.ttn_calc = calc
    wc.exit_codes = TTNMPSComparisonWorkChain.exit_codes

    result = TTNMPSComparisonWorkChain.inspect_ttn(wc)

    assert result == TTNMPSComparisonWorkChain.exit_codes.ERROR_TTN_FAILED


def test_ttn_mps_comparison_compare_results():
    """Test compare_results method."""
    wc = make_workchain(TTNMPSComparisonWorkChain)

    wc.ctx = Namespace()
    wc.ctx.mps_energy = -1.5
    wc.ctx.ttn_energy = -1.45
    wc.ctx.mps_params = {"M_max": 50, "iterations": 10}
    wc.ctx.ttn_params = {"bond_dims": [10, 15, 20], "iterations": 8}

    # Compare
    TTNMPSComparisonWorkChain.compare_results(wc)

    # Check comparison data
    assert wc.ctx.comparison["mps_energy"] == -1.5
    assert wc.ctx.comparison["ttn_energy"] == -1.45
    assert abs(wc.ctx.comparison["energy_difference"] - 0.05) < 1e-10
    assert wc.ctx.comparison["mps_iterations"] == 10
    assert wc.ctx.comparison["ttn_iterations"] == 8


def test_ttn_mps_comparison_finalize():
    """Test finalize method."""
    wc = make_workchain(TTNMPSComparisonWorkChain)

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
    TTNMPSComparisonWorkChain.finalize(wc)

    # Check outputs
    assert wc.out.call_count >= 4  # mps_result, ttn_result, comparison_data, output_parameters


def test_ttn_mps_comparison_with_initial_states():
    """Test that initial states are passed when provided."""
    wc = make_workchain(TTNMPSComparisonWorkChain)

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
    TTNMPSComparisonWorkChain.run_mps(wc)

    # Check that initial_mps was passed in submit kwargs
    assert wc.submit.called
    call_kwargs = wc.submit.call_args
    assert "initial_mps" in call_kwargs.kwargs


def test_ttn_mps_comparison_energy_difference_calculation():
    """Test that energy difference is calculated correctly."""
    wc = make_workchain(TTNMPSComparisonWorkChain)

    wc.ctx = Namespace()
    wc.ctx.mps_energy = -2.0
    wc.ctx.ttn_energy = -1.9
    wc.ctx.mps_params = {}
    wc.ctx.ttn_params = {}

    TTNMPSComparisonWorkChain.compare_results(wc)

    # Check energy difference
    assert abs(wc.ctx.comparison["energy_difference"] - 0.1) < 1e-10
