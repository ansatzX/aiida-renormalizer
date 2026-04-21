"""Behavioral tests for ThermalStateWorkChain."""
from __future__ import annotations

from unittest.mock import Mock

from aiida import orm

from aiida_renormalizer.workchains.thermal import ThermalStateWorkChain
from tests.workchains.conftest import Namespace, make_workchain


def test_setup_rejects_missing_temperature_and_beta():
    wc = make_workchain(ThermalStateWorkChain)
    wc.inputs = Namespace()
    wc.ctx = Namespace()

    result = wc.setup()

    assert result == wc.exit_codes.ERROR_INVALID_TEMPERATURE


def test_setup_rejects_temperature_and_beta_together():
    wc = make_workchain(ThermalStateWorkChain)
    wc.inputs = Namespace(temperature=orm.Float(1.0), beta=orm.Float(1.0))
    wc.ctx = Namespace()

    result = wc.setup()

    assert result == wc.exit_codes.ERROR_INVALID_TEMPERATURE


def test_setup_computes_beta_from_temperature():
    wc = make_workchain(ThermalStateWorkChain)
    wc.inputs = Namespace(temperature=orm.Float(2.0))
    wc.ctx = Namespace()

    result = wc.setup()

    assert result is None
    assert wc.ctx.temperature == 2.0
    assert wc.ctx.beta == 0.5


def test_inspect_thermal_state_rejects_non_positive_partition_function():
    wc = make_workchain(ThermalStateWorkChain)
    params = Mock()
    params.get_dict.return_value = {"partition_function": 0.0}
    calc = Mock()
    calc.is_finished_ok = True
    calc.outputs = Namespace(output_mps=Mock(), output_parameters=params)
    wc.ctx = Namespace(thermal_calc=calc, temperature=1.0)

    result = wc.inspect_thermal_state()

    assert result == wc.exit_codes.ERROR_INVALID_THERMAL_STATE


def test_inspect_thermal_state_extracts_thermodynamics():
    wc = make_workchain(ThermalStateWorkChain)
    params = Mock()
    params.get_dict.return_value = {"partition_function": 2.0, "extra": 1}
    output_mps = Mock()
    calc = Mock()
    calc.is_finished_ok = True
    calc.outputs = Namespace(output_mps=output_mps, output_parameters=params)
    wc.ctx = Namespace(thermal_calc=calc, temperature=1.0)

    result = wc.inspect_thermal_state()

    assert result is None
    assert wc.ctx.thermal_mpdm is output_mps
    assert wc.ctx.partition_function == 2.0
    assert wc.ctx.thermal_params["extra"] == 1


def test_run_initial_state_calcjob_submits():
    wc = make_workchain(ThermalStateWorkChain)
    wc.inputs = Namespace(
        model=Mock(),
        code=Mock(),
        space=orm.Str("GS"),
    )

    ThermalStateWorkChain.run_initial_state_calcjob(wc)

    assert wc.submit.called
    submit_kwargs = wc.submit.call_args.kwargs
    assert submit_kwargs["model"] is wc.inputs.model
    assert submit_kwargs["code"] is wc.inputs.code
    assert submit_kwargs["space"] == wc.inputs.space
