"""Unit tests for SbmModelFromModesWorkChain."""
from __future__ import annotations

from unittest.mock import Mock

from aiida import orm

from aiida_renormalizer.workchains.sbm_model_from_modes import SbmModelFromModesWorkChain
from tests.workchains.conftest import Namespace, make_workchain


def test_run_symbolic_spec_calcjob_submits():
    wc = make_workchain(SbmModelFromModesWorkChain)
    wc.inputs = Namespace(
        code=Mock(),
        epsilon=orm.Float(0.0),
        symbol_map=orm.Dict(dict={}),
        delta_eff=orm.Float(0.3),
    )
    wc.ctx = Namespace(
        bath_model_params={
            "omega_k": [0.5, 1.0],
            "c_j2": [0.1, 0.2],
            "delta_eff": 0.25,
        }
    )

    SbmModelFromModesWorkChain.run_symbolic_spec_calcjob(wc)
    assert wc.submit.called
    submit_kwargs = wc.submit.call_args.kwargs
    assert submit_kwargs["code"] is wc.inputs.code
    assert submit_kwargs["epsilon"] == wc.inputs.epsilon
    assert submit_kwargs["symbol_map"] == wc.inputs.symbol_map


def test_run_model_build_calcjob_submits():
    wc = make_workchain(SbmModelFromModesWorkChain)
    wc.inputs = Namespace(code=Mock())
    wc.ctx = Namespace(
        symbolic_inputs={
            "basis": [{"kind": "half_spin", "dof": "spin", "sigmaqn": [0, 0]}],
            "hamiltonian": [{"symbol": "sigma_x", "dofs": "spin", "factor": 1.0}],
        }
    )
    SbmModelFromModesWorkChain.run_model_build_calcjob(wc)
    assert wc.submit.called
    submit_kwargs = wc.submit.call_args.kwargs
    assert submit_kwargs["code"] is wc.inputs.code
    assert submit_kwargs["symbolic_inputs"].get_dict()["basis"][0]["dof"] == "spin"

def test_inspect_model_build_calcjob_stores_model():
    wc = make_workchain(SbmModelFromModesWorkChain)
    model_node = Mock()
    wc.ctx = Namespace(
        model_build_calc=Namespace(
            is_finished_ok=True,
            outputs=Namespace(output_model=model_node),
        )
    )
    result = SbmModelFromModesWorkChain.inspect_model_build_calcjob(wc)
    assert result is None
    assert wc.ctx.model_data is model_node
