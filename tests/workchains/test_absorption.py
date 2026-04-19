"""Unit tests for AbsorptionWorkChain."""
import pytest
from unittest.mock import Mock, MagicMock, patch

from aiida import orm

from aiida_renormalizer.workchains.absorption import AbsorptionWorkChain
from tests.workchains.conftest import make_workchain, Namespace


def test_absorption_needs_ground_state():
    """Test needs_ground_state method."""
    wc = make_workchain(AbsorptionWorkChain)

    # No ground state provided
    wc.inputs = Namespace()
    assert wc.needs_ground_state() is True

    # Ground state provided
    wc.inputs = Namespace(ground_state=Mock())
    assert wc.needs_ground_state() is False


def test_absorption_temperature_checks():
    """Test is_zero_temperature method."""
    wc = make_workchain(AbsorptionWorkChain)

    # Zero temperature (no temperature input)
    wc.inputs = Namespace()
    assert wc.is_zero_temperature() is True

    # Finite temperature
    wc.inputs = Namespace(temperature=orm.Float(300.0))
    assert wc.is_zero_temperature() is False


def test_absorption_prepare_ground_state_dmrg():
    """Test ground state preparation with DMRG."""
    wc = make_workchain(AbsorptionWorkChain)

    wc.inputs = Namespace(
        model=Mock(),
        code=Mock(),
        gs_method=orm.Str("dmrg"),
    )
    wc.ctx = Namespace()

    result = wc.prepare_ground_state()

    assert wc.submit.called
    assert result is not None


def test_absorption_prepare_ground_state_imag_time():
    """Test ground state preparation with ImagTime."""
    wc = make_workchain(AbsorptionWorkChain)

    wc.inputs = Namespace(
        model=Mock(),
        code=Mock(),
        gs_method=orm.Str("imag_time"),
    )
    wc.ctx = Namespace()

    result = wc.prepare_ground_state()

    assert wc.submit.called
    assert result is not None


def test_absorption_inspect_ground_state_success():
    """Test inspect_ground_state with successful calculation."""
    wc = make_workchain(AbsorptionWorkChain)

    mock_mps = Mock()
    calc = Mock()
    calc.is_finished_ok = True
    calc.outputs = Namespace(output_mps=mock_mps)

    wc.ctx = Namespace(ground_state_calc=calc)

    result = wc.inspect_ground_state()

    assert result is None
    assert wc.ctx.ground_state == mock_mps


def test_absorption_inspect_ground_state_failure():
    """Test inspect_ground_state with failed calculation."""
    wc = make_workchain(AbsorptionWorkChain)

    calc = Mock()
    calc.is_finished_ok = False
    calc.exit_status = 300

    wc.ctx = Namespace(ground_state_calc=calc)

    result = wc.inspect_ground_state()

    assert result == wc.exit_codes.ERROR_GROUND_STATE_FAILED


def test_absorption_run_zero_t():
    """Test zero-temperature spectrum calculation."""
    wc = make_workchain(AbsorptionWorkChain)

    wc.inputs = Namespace(
        model=Mock(),
        spectratype=orm.Str("abs"),
        propagation=orm.Str("two_way"),
        code=Mock(),
    )
    wc.ctx = Namespace(ground_state=Mock())

    result = wc.run_zero_t_spectrum()

    assert wc.submit.called


def test_absorption_run_finite_t():
    """Test finite-temperature spectrum calculation."""
    wc = make_workchain(AbsorptionWorkChain)

    wc.inputs = Namespace(
        model=Mock(),
        temperature=orm.Float(300.0),
        spectratype=orm.Str("abs"),
        code=Mock(),
    )
    wc.ctx = Namespace(ground_state=Mock())

    result = wc.run_finite_t_spectrum()

    assert wc.submit.called


def test_absorption_inspect_spectrum_success():
    """Test inspect_spectrum with successful calculation."""
    wc = make_workchain(AbsorptionWorkChain)

    params = Mock()
    params.get_dict.return_value = {
        "autocorrelation": [1.0, 0.5, 0.25],
        "time": [0.0, 1.0, 2.0],
    }

    calc = Mock()
    calc.is_finished_ok = True
    calc.outputs = Namespace(output_parameters=params)

    wc.ctx = Namespace(spectrum_calc=calc)

    result = wc.inspect_spectrum()

    assert result is None
    assert wc.ctx.spectrum_params is not None


def test_absorption_inspect_spectrum_failure():
    """Test inspect_spectrum with failed calculation."""
    wc = make_workchain(AbsorptionWorkChain)

    calc = Mock()
    calc.is_finished_ok = False
    calc.exit_status = 300

    wc.ctx = Namespace(spectrum_calc=calc)

    result = wc.inspect_spectrum()

    assert result == wc.exit_codes.ERROR_SPECTRUM_FAILED


def test_absorption_finalize():
    """Test finalize method."""
    wc = make_workchain(AbsorptionWorkChain)

    wc.inputs = Namespace(spectratype=orm.Str("abs"))
    wc.ctx = Namespace(
        ground_state=Mock(),
        spectrum_params={
            "autocorrelation": [1.0, 0.5, 0.25],
            "time": [0.0, 1.0, 2.0],
        },
    )

    # Mock is_zero_temperature
    wc.is_zero_temperature = Mock(return_value=True)

    wc.finalize()

    # Check outputs
    assert wc.out.call_count >= 2  # ground_state, output_parameters


def test_absorption_spectratype_validation():
    """Test spectratype validation."""
    wc = make_workchain(AbsorptionWorkChain)

    wc.inputs = Namespace(spectratype=orm.Str("invalid"))

    # Should raise ValueError
    with pytest.raises(ValueError):
        wc.setup()
