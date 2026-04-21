"""WorkChain: initial MPS + MPO -> real-time dynamics."""
from __future__ import annotations

from aiida import orm
from aiida.engine import ToContext, WorkChain

from aiida_renormalizer.calculations.composite.tdvp import TDVPCalcJob
from aiida_renormalizer.data import ModelData, MPOData, MPSData, TensorNetworkLayoutData


class MPSDynamicsWorkChain(WorkChain):
    """Run TDVP dynamics from prepared MPS/MPO."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("model", valid_type=ModelData)
        spec.input("mpo", valid_type=MPOData)
        spec.input("initial_mps", valid_type=MPSData)
        spec.input("code", valid_type=orm.AbstractCode)
        spec.input("total_time", valid_type=orm.Float)
        spec.input("dt", valid_type=orm.Float)
        spec.input("trajectory_interval", valid_type=orm.Int, default=lambda: orm.Int(1))
        spec.input("tn_layout", valid_type=TensorNetworkLayoutData, required=False)

        spec.output("final_mps", valid_type=MPSData)
        spec.output("trajectory", valid_type=orm.ArrayData, required=False)
        spec.output("output_tn_layout", valid_type=TensorNetworkLayoutData, required=False)
        spec.output("output_parameters", valid_type=orm.Dict)

        spec.exit_code(590, "ERROR_DYNAMICS_FAILED", message="TDVP dynamics failed")

        spec.outline(cls.run_tdvp, cls.inspect_tdvp, cls.finalize)

    def run_tdvp(self):
        inputs = {
            "model": self.inputs.model,
            "mpo": self.inputs.mpo,
            "initial_mps": self.inputs.initial_mps,
            "code": self.inputs.code,
            "total_time": self.inputs.total_time,
            "dt": self.inputs.dt,
            "trajectory_interval": self.inputs.trajectory_interval,
        }
        if "tn_layout" in self.inputs:
            inputs["tn_layout"] = self.inputs.tn_layout
        return ToContext(calc=self.submit(TDVPCalcJob, **inputs))

    def inspect_tdvp(self):
        calc = self.ctx.calc
        if not calc.is_finished_ok:
            self.report(f"TDVP failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_DYNAMICS_FAILED

    def finalize(self):
        calc = self.ctx.calc
        self.out("final_mps", calc.outputs.output_mps)
        if "output_tn_layout" in calc.outputs:
            self.out("output_tn_layout", calc.outputs.output_tn_layout)
        elif "tn_layout" in self.inputs:
            self.out("output_tn_layout", self.inputs.tn_layout)
        if "trajectory" in calc.outputs:
            self.out("trajectory", calc.outputs.trajectory)
        self.out("output_parameters", calc.outputs.output_parameters)
