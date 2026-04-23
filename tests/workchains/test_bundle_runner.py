"""Unit tests for BundleRunnerWorkChain."""
from __future__ import annotations

from unittest.mock import Mock

from aiida import orm

from aiida_renormalizer.workchains.bundle_runner import BundleRunnerWorkChain
from tests.workchains.conftest import Namespace, make_workchain


def test_setup_rejects_missing_or_duplicated_manifest_inputs():
    wc = make_workchain(BundleRunnerWorkChain)
    wc.inputs = Namespace(code=Mock())
    wc.ctx = Namespace()
    assert BundleRunnerWorkChain.setup(wc) == wc.exit_codes.ERROR_INVALID_INPUT

    wc2 = make_workchain(BundleRunnerWorkChain)
    wc2.inputs = Namespace(
        code=Mock(),
        manifest=orm.Dict(dict={"stages": [{"name": "s1", "script": "x=1"}]}),
        stages=orm.List(list=[{"name": "s1", "script": "x=1"}]),
    )
    wc2.ctx = Namespace()
    assert BundleRunnerWorkChain.setup(wc2) == wc2.exit_codes.ERROR_INVALID_INPUT


def test_run_bundle_submits_calcjob_with_resume():
    wc = make_workchain(BundleRunnerWorkChain)
    wc.inputs = Namespace(
        code=Mock(),
        manifest=orm.Dict(dict={"stages": [{"name": "s1", "script": "x = 1"}]}),
        fail_fast=orm.Bool(True),
        max_retries=orm.Int(1),
        resume_from_stage=orm.Int(3),
    )
    wc.ctx = Namespace()
    assert BundleRunnerWorkChain.setup(wc) is None

    BundleRunnerWorkChain.run_bundle(wc)
    assert wc.submit.called
    submit_kwargs = wc.submit.call_args.kwargs
    assert submit_kwargs["manifest"] == wc.ctx.manifest
    assert submit_kwargs["resume_from_stage"].value == 3


def test_inspect_bundle_updates_resume_stage_on_failure():
    wc = make_workchain(BundleRunnerWorkChain)
    wc.inputs = Namespace(max_retries=orm.Int(2))
    params = Mock()
    params.get_dict.return_value = {"converged": False, "failed_stage": 5}
    wc.ctx = Namespace(
        attempt=1,
        finished=False,
        bundle_calc=Namespace(is_finished_ok=False, outputs=Namespace(output_parameters=params)),
        resume_from_stage=1,
        last_output_parameters=None,
    )

    result = BundleRunnerWorkChain.inspect_bundle(wc)
    assert result is None
    assert wc.ctx.resume_from_stage == 5
    assert wc.ctx.last_output_parameters == params


def test_finalize_sets_default_output_when_no_calc_output():
    wc = make_workchain(BundleRunnerWorkChain)
    wc.ctx = Namespace(last_output_parameters=None)

    BundleRunnerWorkChain.finalize(wc)

    assert wc.out.called
    args, _ = wc.out.call_args
    assert args[0] == "output_parameters"
    assert args[1].get_dict()["converged"] is False


def test_finalize_forwards_last_output_parameters():
    wc = make_workchain(BundleRunnerWorkChain)
    params = orm.Dict(dict={"converged": True, "failed_stage": None})
    wc.ctx = Namespace(last_output_parameters=params)

    BundleRunnerWorkChain.finalize(wc)

    assert wc.out.called
    args, _ = wc.out.call_args
    assert args[0] == "output_parameters"
    assert args[1] == params
