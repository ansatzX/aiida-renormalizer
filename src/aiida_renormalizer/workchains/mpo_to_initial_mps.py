"""WorkChain: model+mpo -> initial MPS."""
from __future__ import annotations

from aiida import orm
from aiida.engine import ToContext, WorkChain

from aiida_renormalizer.calculations.composite.dmrg import DMRGCalcJob
from aiida_renormalizer.data import ModelData, MPOData, MPSData


class MPOToInitialMPSWorkChain(WorkChain):
    """Build an initial MPS state using DMRG ground-state optimization."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("model", valid_type=ModelData)
        spec.input("mpo", valid_type=MPOData)
        spec.input("code", valid_type=orm.AbstractCode)

        spec.output("initial_mps", valid_type=MPSData)
        spec.output("output_parameters", valid_type=orm.Dict)

        spec.exit_code(580, "ERROR_INITIAL_MPS_FAILED", message="Initial MPS generation failed")

        spec.outline(cls.run_dmrg, cls.inspect_dmrg, cls.finalize)

    def run_dmrg(self):
        inputs = {
            "model": self.inputs.model,
            "mpo": self.inputs.mpo,
            "code": self.inputs.code,
        }
        return ToContext(calc=self.submit(DMRGCalcJob, **inputs))

    def inspect_dmrg(self):
        calc = self.ctx.calc
        if not calc.is_finished_ok:
            self.report(f"DMRG failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_INITIAL_MPS_FAILED

    def finalize(self):
        calc = self.ctx.calc
        self.out("initial_mps", calc.outputs.output_mps)
        self.out("output_parameters", calc.outputs.output_parameters)
