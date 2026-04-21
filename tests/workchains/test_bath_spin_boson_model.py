"""Unit tests for BathSpinBosonModelWorkChain."""
from __future__ import annotations

from unittest.mock import Mock

from aiida import orm

from aiida_renormalizer.workchains.bath_spin_boson_model import BathSpinBosonModelWorkChain
from tests.workchains.conftest import Namespace, make_workchain


def test_run_bath_model_calcjob_submits():
    wc = make_workchain(BathSpinBosonModelWorkChain)
    wc.inputs = Namespace(
        code=Mock(),
        construction=orm.Str("param2mollist"),
        epsilon=orm.Float(0.0),
        spectral_density_type=orm.Str("ohmic_exp"),
        beta=orm.Float(0.7),
        alpha=orm.Float(0.1),
        raw_delta=orm.Float(1.0),
        omega_c=orm.Float(2.0),
        renormalization_p=orm.Float(1.2),
        n_phonons=orm.Int(8),
    )

    BathSpinBosonModelWorkChain.run_bath_model_calcjob(wc)
    assert wc.submit.called
    submit_kwargs = wc.submit.call_args.kwargs
    assert submit_kwargs["code"] is wc.inputs.code
    assert submit_kwargs["construction"] == wc.inputs.construction


def test_run_symbolic_spec_calcjob_submits(aiida_profile):
    wc = make_workchain(BathSpinBosonModelWorkChain)
    wc.inputs = Namespace(
        code=Mock(),
        epsilon=orm.Float(0.0),
        tree_type=orm.Str("binary"),
        m_max=orm.Int(16),
        symbol_map=orm.Dict(dict={}),
        min_nbas=orm.Float(4.0),
        nbas_prefactor=orm.Float(16.0),
        vib_prefix=orm.Str("v_"),
    )
    wc.ctx = Namespace(
        bath_params={
            "omega_k": [0.5, 1.0],
            "c_j2": [0.1, 0.2],
            "delta_eff": 0.3,
        }
    )

    BathSpinBosonModelWorkChain.run_symbolic_spec_calcjob(wc)
    assert wc.submit.called
    submit_kwargs = wc.submit.call_args.kwargs
    assert submit_kwargs["code"] is wc.inputs.code
    assert submit_kwargs["delta_eff"].value == 0.3


def test_inspect_symbolic_spec_calcjob_extracts_payload():
    wc = make_workchain(BathSpinBosonModelWorkChain)
    wc.ctx = Namespace(
        symbolic_calc=Namespace(
            is_finished_ok=True,
            outputs=Namespace(
                output_parameters=Mock(
                    get_dict=Mock(
                        return_value={
                            "symbolic_inputs": {
                                "basis": [{"kind": "half_spin", "dof": "spin", "sigmaqn": [0, 0]}],
                                "hamiltonian": [{"symbol": "sigma_x", "dofs": "spin", "factor": 1.0}],
                            },
                            "metadata": {"n_modes": 1},
                        }
                    )
                )
            ),
        )
    )

    result = BathSpinBosonModelWorkChain.inspect_symbolic_spec_calcjob(wc)
    assert result is None
    assert "symbolic_inputs" in wc.ctx
    assert "symbolic_metadata" in wc.ctx


def test_run_model_build_calcjob_submits():
    wc = make_workchain(BathSpinBosonModelWorkChain)
    wc.inputs = Namespace(code=Mock())
    wc.ctx = Namespace(
        symbolic_inputs={
            "basis": [{"kind": "half_spin", "dof": "spin", "sigmaqn": [0, 0]}],
            "hamiltonian": [{"symbol": "sigma_x", "dofs": "spin", "factor": 1.0}],
        }
    )

    BathSpinBosonModelWorkChain.run_model_build_calcjob(wc)
    assert wc.submit.called
    submit_kwargs = wc.submit.call_args.kwargs
    assert submit_kwargs["code"] is wc.inputs.code
    assert submit_kwargs["symbolic_inputs"].get_dict()["basis"][0]["dof"] == "spin"
