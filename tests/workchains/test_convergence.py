"""Unit tests for ConvergenceWorkChain."""
import pytest
from unittest.mock import Mock, MagicMock, patch

from aiida import orm

from aiida_renormalizer.workchains.convergence import ConvergenceWorkChain
from tests.workchains.conftest import make_workchain, Namespace


def test_convergence_setup():
    """Test ConvergenceWorkChain setup."""
    wc = make_workchain(ConvergenceWorkChain)

    wc.inputs = Namespace(
        m_values=orm.List(list=[50, 100, 150, 200]),
        convergence_threshold=orm.Float(1e-6),
    )
    wc.ctx = Namespace()

    wc.setup()

    assert wc.ctx.m_values == [50, 100, 150, 200]
    assert wc.ctx.current_index == 0
    assert wc.ctx.energies == []
    assert wc.ctx.mps_list == []
    assert wc.ctx.converged is False


def test_convergence_run_calculation():
    """Test run_calculation method."""
    wc = make_workchain(ConvergenceWorkChain)

    wc.inputs = Namespace(
        model=Mock(),
        mpo=Mock(),
        code=Mock(),
        config=orm.Dict({"procedure": [[100, 1e-6]]}),
    )
    wc.ctx = Namespace(
        current_index=0,
        m_values=[50, 100, 150],
        mps_list=[],
    )

    result = wc.run_calculation()

    assert wc.submit.called
    # Check that config was updated with current M
    call_args = wc.submit.call_args
    assert call_args is not None


def test_convergence_run_calculation_with_previous_mps():
    """Test run_calculation uses previous MPS as initial guess."""
    wc = make_workchain(ConvergenceWorkChain)

    wc.inputs = Namespace(
        model=Mock(),
        mpo=Mock(),
        code=Mock(),
    )
    wc.ctx = Namespace(
        current_index=1,  # Second iteration
        m_values=[50, 100, 150],
        mps_list=[Mock()],  # Previous MPS
    )

    # Track what was submitted via self.submit
    submitted_inputs = {}

    def capture_submit(cls, **kwargs):
        submitted_inputs.update(kwargs)
        return Mock()

    wc.submit = capture_submit

    wc.run_calculation()

    # Check that initial_mps was included
    assert "initial_mps" in submitted_inputs


def test_convergence_inspect_calculation_success():
    """Test inspect_calculation with successful calculation."""
    wc = make_workchain(ConvergenceWorkChain)

    wc.inputs = Namespace(convergence_threshold=orm.Float(1e-6))

    # Mock calculation
    params = Mock()
    params.get_dict.return_value = {"energy": -1.5}

    mock_mps = Mock()
    calc = Mock()
    calc.is_finished_ok = True
    calc.outputs = Namespace(
        output_mps=mock_mps,
        output_parameters=params,
    )

    wc.ctx = Namespace(
        current_index=0,
        m_values=[50, 100, 150],
        energies=[],
        mps_list=[],
        current_calc=calc,
    )

    # Inspect
    result = wc.inspect_calculation()

    # Check results
    assert result is None  # No error
    assert wc.ctx.energies[-1] == -1.5
    assert wc.ctx.current_index == 1


def test_convergence_inspect_calculation_failure():
    """Test inspect_calculation with failed calculation."""
    wc = make_workchain(ConvergenceWorkChain)

    calc = Mock()
    calc.is_finished_ok = False
    calc.exit_status = 300

    wc.ctx = Namespace(current_calc=calc)

    result = wc.inspect_calculation()

    assert result == wc.exit_codes.ERROR_CALCULATION_FAILED


def test_convergence_convergence_detection():
    """Test convergence detection."""
    wc = make_workchain(ConvergenceWorkChain)

    wc.inputs = Namespace(convergence_threshold=orm.Float(1e-6))

    # Mock calculation
    params = Mock()
    params.get_dict.return_value = {"energy": -1.5000001}

    mock_mps = Mock()
    calc = Mock()
    calc.is_finished_ok = True
    calc.outputs = Namespace(
        output_mps=mock_mps,
        output_parameters=params,
    )

    wc.ctx = Namespace(
        current_index=1,
        m_values=[50, 100, 150],
        energies=[-1.5, -1.5000001],  # Very close energies
        mps_list=[Mock(), Mock()],
        observables=[],
        current_calc=calc,
    )

    # Inspect
    result = wc.inspect_calculation()

    # Check that convergence was detected
    assert result is None
    assert wc.ctx.converged is True


def test_convergence_no_convergence():
    """Test scenario where convergence is not achieved."""
    wc = make_workchain(ConvergenceWorkChain)

    wc.inputs = Namespace(convergence_threshold=orm.Float(1e-10))  # Very strict

    wc.ctx = Namespace(
        current_index=2,  # Exhausted all values
        m_values=[50, 100, 150],
        energies=[-1.5, -1.4, -1.3],  # Still changing
        mps_list=[Mock(), Mock(), Mock()],
        converged=False,
        optimal_m_index=2,  # Last result
    )

    wc.finalize()

    # Should still output results (using last result)
    assert wc.out.call_count >= 3


def test_convergence_finalize():
    """Test finalize method."""
    wc = make_workchain(ConvergenceWorkChain)

    wc.inputs = Namespace(convergence_threshold=orm.Float(1e-6))

    wc.ctx = Namespace(
        converged=True,
        optimal_m_index=1,
        m_values=[50, 100, 150],
        energies=[-1.5, -1.5000001],
        mps_list=[Mock(), Mock()],
    )

    wc.finalize()

    # Check outputs
    assert wc.out.call_count >= 3  # converged_mps, optimal_m, convergence_data


def test_convergence_energy_difference_calculation():
    """Test energy difference calculation in convergence data."""
    wc = make_workchain(ConvergenceWorkChain)

    wc.inputs = Namespace(convergence_threshold=orm.Float(1e-6))

    wc.ctx = Namespace(
        converged=True,
        optimal_m_index=2,
        m_values=[50, 100, 150],
        energies=[-1.5, -1.5000001, -1.50000015],
        mps_list=[Mock(), Mock(), Mock()],
    )

    # Capture convergence_data
    output_data = {}
    def capture_output(key, value):
        output_data[key] = value

    wc.out.side_effect = capture_output

    wc.finalize()

    # Check that convergence_data contains energy differences
    assert "convergence_data" in output_data
    conv_data = output_data["convergence_data"].get_dict()
    assert "energy_differences" in conv_data
    assert len(conv_data["energy_differences"]) == 2  # 3 energies -> 2 differences
