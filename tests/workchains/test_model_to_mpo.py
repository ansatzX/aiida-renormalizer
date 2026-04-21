"""Unit tests for ModelToMPOWorkChain."""
from __future__ import annotations

from unittest.mock import Mock

from aiida_renormalizer.workchains.model_to_mpo import ModelToMPOWorkChain
from tests.workchains.conftest import Namespace, make_workchain


def test_run_build_submits_build_mpo_without_local_op_rebuild():
    wc = make_workchain(ModelToMPOWorkChain)
    model = Mock()
    wc.inputs = Namespace(model=model, code=Mock())

    ModelToMPOWorkChain.run_build(wc)

    assert wc.submit.called
    submit_kwargs = wc.submit.call_args.kwargs
    assert submit_kwargs["model"] is model
    assert "op" not in submit_kwargs
