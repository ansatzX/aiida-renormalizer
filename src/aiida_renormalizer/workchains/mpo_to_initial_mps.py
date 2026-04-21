"""WorkChain: model+mpo -> initial MPS."""
from __future__ import annotations

from aiida import orm
from aiida.engine import ToContext, WorkChain

from aiida_renormalizer.calculations.composite.dmrg import DMRGCalcJob
from aiida_renormalizer.data import ModelData, MPOData, MPSData, TensorNetworkLayoutData


class MPOToInitialMPSWorkChain(WorkChain):
    """Build an initial MPS state using DMRG ground-state optimization."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("model", valid_type=ModelData)
        spec.input("mpo", valid_type=MPOData)
        spec.input("code", valid_type=orm.AbstractCode)
        spec.input("tn_layout", valid_type=TensorNetworkLayoutData, required=False)

        spec.output("initial_mps", valid_type=MPSData)
        spec.output("output_tn_layout", valid_type=TensorNetworkLayoutData, required=False)
        spec.output("output_parameters", valid_type=orm.Dict)

        spec.exit_code(580, "ERROR_INITIAL_MPS_FAILED", message="Initial MPS generation failed")

        spec.outline(cls.run_dmrg, cls.inspect_dmrg, cls.finalize)

    def run_dmrg(self):
        inputs = {
            "model": self.inputs.model,
            "mpo": self.inputs.mpo,
            "code": self.inputs.code,
        }
        if "tn_layout" in self.inputs:
            inputs["tn_layout"] = self.inputs.tn_layout
        return ToContext(calc=self.submit(DMRGCalcJob, **inputs))

    def inspect_dmrg(self):
        calc = self.ctx.calc
        if not calc.is_finished_ok:
            self.report(f"DMRG failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_INITIAL_MPS_FAILED

    def finalize(self):
        calc = self.ctx.calc
        self.out("initial_mps", calc.outputs.output_mps)
        if "output_tn_layout" in calc.outputs:
            self.out("output_tn_layout", calc.outputs.output_tn_layout)
        elif "tn_layout" in self.inputs:
            self.out("output_tn_layout", self.inputs.tn_layout)
        self.out("output_parameters", calc.outputs.output_parameters)
