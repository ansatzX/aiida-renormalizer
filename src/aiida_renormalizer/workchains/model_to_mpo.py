"""WorkChain: ModelData -> MPO via Hamiltonian OpSum."""
from __future__ import annotations

from aiida import orm
from aiida.engine import ToContext, WorkChain

from aiida_renormalizer.calculations.basic.build_mpo import BuildMPOCalcJob
from aiida_renormalizer.data import ModelData, MPOData


class ModelToMPOWorkChain(WorkChain):
    """Build MPO from the model Hamiltonian declared in ModelData."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("model", valid_type=ModelData)
        spec.input("code", valid_type=orm.AbstractCode)

        spec.output("mpo", valid_type=MPOData)
        spec.output("output_parameters", valid_type=orm.Dict)

        spec.exit_code(570, "ERROR_BUILD_MPO_FAILED", message="MPO build failed")

        spec.outline(cls.run_build, cls.inspect_build, cls.finalize)

    def run_build(self):
        from renormalizer.model.op import OpSum

        from aiida_renormalizer.data.op import OpData

        model = self.inputs.model.load_model()
        ham = model.ham_terms
        if not isinstance(ham, OpSum):
            ham = OpSum(list(ham))
        op_data = OpData.from_opsum(ham)

        inputs = {
            "model": self.inputs.model,
            "code": self.inputs.code,
            "op": op_data,
        }
        return ToContext(calc=self.submit(BuildMPOCalcJob, **inputs))

    def inspect_build(self):
        calc = self.ctx.calc
        if not calc.is_finished_ok:
            self.report(f"BuildMpo failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_BUILD_MPO_FAILED

    def finalize(self):
        calc = self.ctx.calc
        self.out("mpo", calc.outputs.output_mpo)
        self.out(
            "output_parameters",
            orm.Dict(dict={"calc_pk": calc.pk, "process_label": "BuildMPOCalcJob"}),
        )
