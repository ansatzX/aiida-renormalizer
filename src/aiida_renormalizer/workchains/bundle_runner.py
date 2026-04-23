"""WorkChain orchestration for BundleRunnerCalcJob with resume retries."""
from __future__ import annotations

from aiida import orm
from aiida.engine import ToContext, WorkChain, while_

from aiida_renormalizer.calculations.manifest_ops import generate_bundle_manifest
from aiida_renormalizer.calculations.basic.bundle_runner import BundleRunnerCalcJob


class BundleRunnerWorkChain(WorkChain):
    """Submit a single BundleRunnerCalcJob and optionally resume on stage failure."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("code", valid_type=orm.AbstractCode)
        spec.input("manifest", valid_type=orm.Dict, required=False)
        spec.input("stages", valid_type=orm.List, required=False)
        spec.input("fail_fast", valid_type=orm.Bool, default=lambda: orm.Bool(True))
        spec.input("max_retries", valid_type=orm.Int, default=lambda: orm.Int(0))
        spec.input("resume_from_stage", valid_type=orm.Int, default=lambda: orm.Int(1))

        spec.output("output_parameters", valid_type=orm.Dict)

        spec.exit_code(640, "ERROR_INVALID_INPUT", message="Provide either manifest or stages")
        spec.exit_code(641, "ERROR_BUNDLE_FAILED", message="BundleRunner failed after retries")

        spec.outline(
            cls.setup,
            while_(cls.should_run)(
                cls.run_bundle,
                cls.inspect_bundle,
            ),
            cls.finalize,
        )

    def setup(self):
        has_manifest = "manifest" in self.inputs
        has_stages = "stages" in self.inputs
        if has_manifest == has_stages:
            return self.exit_codes.ERROR_INVALID_INPUT

        self.ctx.attempt = 0
        self.ctx.finished = False
        self.ctx.resume_from_stage = self.inputs.resume_from_stage.value
        self.ctx.last_output_parameters = None

        if has_manifest:
            self.ctx.manifest = self.inputs.manifest
        else:
            self.ctx.manifest = generate_bundle_manifest(self.inputs.stages)

    def should_run(self):
        max_attempts = self.inputs.max_retries.value + 1
        return (not self.ctx.finished) and (self.ctx.attempt < max_attempts)

    def run_bundle(self):
        self.ctx.attempt += 1
        return ToContext(
            bundle_calc=self.submit(
                BundleRunnerCalcJob,
                code=self.inputs.code,
                manifest=self.ctx.manifest,
                resume_from_stage=orm.Int(self.ctx.resume_from_stage),
                fail_fast=self.inputs.fail_fast,
            )
        )

    def inspect_bundle(self):
        calc = self.ctx.bundle_calc

        if "output_parameters" in calc.outputs:
            self.ctx.last_output_parameters = calc.outputs.output_parameters
            params = calc.outputs.output_parameters.get_dict()
        else:
            params = {}

        if calc.is_finished_ok and params.get("converged") is True:
            self.ctx.finished = True
            return None

        failed_stage = params.get("failed_stage")
        if isinstance(failed_stage, int) and failed_stage >= 1:
            self.ctx.resume_from_stage = failed_stage
            self.report(
                f"Bundle attempt {self.ctx.attempt} failed at stage {failed_stage}; "
                f"resume_from_stage set to {failed_stage}"
            )
        else:
            self.ctx.resume_from_stage = 1

        max_attempts = self.inputs.max_retries.value + 1
        if self.ctx.attempt >= max_attempts:
            return self.exit_codes.ERROR_BUNDLE_FAILED

        return None

    def finalize(self):
        if self.ctx.last_output_parameters is None:
            self.out("output_parameters", orm.Dict(dict={"converged": False}))
            return

        self.out("output_parameters", self.ctx.last_output_parameters)
