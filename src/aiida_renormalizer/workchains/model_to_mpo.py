"""WorkChain: ModelData -> MPO via Hamiltonian OpSum."""
from __future__ import annotations

from aiida import orm
from aiida.engine import ToContext, WorkChain

from aiida_renormalizer.calculations.basic.build_mpo import BuildMPOCalcJob
from aiida_renormalizer.data import ModelData, MPOData, TensorNetworkLayoutData


class ModelToMPOWorkChain(WorkChain):
    """Build MPO from the model Hamiltonian declared in ModelData."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("model", valid_type=ModelData)
        spec.input("code", valid_type=orm.AbstractCode)
        spec.input("tn_layout", valid_type=TensorNetworkLayoutData, required=False)

        spec.output("mpo", valid_type=MPOData)
        spec.output("output_tn_layout", valid_type=TensorNetworkLayoutData, required=False)
        spec.output("output_parameters", valid_type=orm.Dict)

        spec.exit_code(570, "ERROR_BUILD_MPO_FAILED", message="MPO build failed")

        spec.outline(cls.run_build, cls.inspect_build, cls.finalize)

    def run_build(self):
        inputs = {
            "model": self.inputs.model,
            "code": self.inputs.code,
        }
        if "tn_layout" in self.inputs:
            inputs["tn_layout"] = self.inputs.tn_layout
        return ToContext(calc=self.submit(BuildMPOCalcJob, **inputs))

    def inspect_build(self):
        calc = self.ctx.calc
        if not calc.is_finished_ok:
            self.report(f"BuildMpo failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_BUILD_MPO_FAILED

    def finalize(self):
        calc = self.ctx.calc
        mpo = calc.outputs.output_mpo
        if not mpo.is_stored:
            mpo.store()
        self.out("mpo", mpo)
        if "output_tn_layout" in calc.outputs:
            self.out("output_tn_layout", calc.outputs.output_tn_layout)
        elif "tn_layout" in self.inputs:
            self.out("output_tn_layout", self.inputs.tn_layout)

        params = orm.Dict(dict={"calc_pk": calc.pk, "process_label": "BuildMPOCalcJob"})
        if not params.is_stored:
            params.store()
        self.out("output_parameters", params)
