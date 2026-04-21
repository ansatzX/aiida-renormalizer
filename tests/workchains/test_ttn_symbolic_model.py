"""Unit tests for TTNSymbolicModelWorkChain."""
from __future__ import annotations

from unittest.mock import Mock

from aiida_renormalizer.workchains.ttn_symbolic_model import TTNSymbolicModelWorkChain
from tests.workchains.conftest import Namespace, make_workchain


def test_run_calcjob_submits_symbolic_model_calcjob():
    wc = make_workchain(TTNSymbolicModelWorkChain)
    wc.inputs = Namespace(
        code=Mock(),
        alpha=Mock(),
        s_exponent=Mock(),
        omega_c=Mock(),
        n_modes=Mock(),
        raw_delta=Mock(),
        renormalization_p=Mock(),
        tree_type=Mock(),
        m_max=Mock(),
        symbol_map=Mock(),
        process=Mock(),
    )

    TTNSymbolicModelWorkChain.run_calcjob(wc)
    assert wc.submit.called
    submit_kwargs = wc.submit.call_args.kwargs
    assert submit_kwargs["code"] is wc.inputs.code
    assert submit_kwargs["alpha"] is wc.inputs.alpha
    assert submit_kwargs["process"] is wc.inputs.process


def test_inspect_calcjob_extracts_outputs():
    wc = make_workchain(TTNSymbolicModelWorkChain)
    wc.ctx = Namespace(
        calc=Mock(
            is_finished_ok=True,
            outputs=Namespace(
                output_parameters=Mock(
                    get_dict=Mock(
                        return_value={
                            "symbolic_inputs": {"basis": [], "hamiltonian": []},
                            "metadata": {"n_modes": 8},
                        }
                    )
                )
            ),
        )
    )

    result = TTNSymbolicModelWorkChain.inspect_calcjob(wc)
    assert result is None
    assert "symbolic_inputs" in wc.ctx
    assert "metadata" in wc.ctx
