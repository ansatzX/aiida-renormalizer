"""WorkChain for symbolic TTNS dynamics evolution."""
from __future__ import annotations

from aiida import orm
from aiida.engine import ToContext, WorkChain

from aiida_renormalizer.calculations.ttn.ttns_symbolic_evolve import TTNSSymbolicEvolveCalcJob
from aiida_renormalizer.data import BasisTreeData, TTNSData, TensorNetworkLayoutData


class TTNSymbolicDynamicsWorkChain(WorkChain):
    """Evolve TTNS dynamics from symbolic TTN/TTNO declaration."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("code", valid_type=orm.AbstractCode, help="Code for CalcJob execution")
        spec.input("symbolic_inputs", valid_type=orm.Dict)
        spec.input("dt", valid_type=orm.Float)
        spec.input("nsteps", valid_type=orm.Int)
        spec.input("method", valid_type=orm.Str, default=lambda: orm.Str("tdvp_ps"))
        spec.input("tn_layout", valid_type=TensorNetworkLayoutData, required=False)

        spec.output("output_ttns", valid_type=TTNSData, required=False)
        spec.output("output_basis_tree", valid_type=BasisTreeData, required=False)
        spec.output("output_tn_layout", valid_type=TensorNetworkLayoutData, required=False)
        spec.output("output_parameters", valid_type=orm.Dict)

        spec.exit_code(540, "ERROR_DYNAMICS_FAILED", message="Symbolic TTNS dynamics failed")

        spec.outline(cls.run_dynamics, cls.inspect_dynamics, cls.finalize)

    def run_dynamics(self):
        inputs = {
            "code": self.inputs.code,
            "symbolic_inputs": self.inputs.symbolic_inputs,
            "dt": self.inputs.dt,
            "nsteps": self.inputs.nsteps,
            "method": self.inputs.method,
        }
        if "tn_layout" in self.inputs:
            inputs["tn_layout"] = self.inputs.tn_layout
        return ToContext(dynamics_calc=self.submit(TTNSSymbolicEvolveCalcJob, **inputs))

    def inspect_dynamics(self):
        calc = self.ctx.dynamics_calc
        if not calc.is_finished_ok:
            self.report(f"Dynamics failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_DYNAMICS_FAILED

    def finalize(self):
        calc = self.ctx.dynamics_calc
        if "output_ttns" in calc.outputs:
            self.out("output_ttns", calc.outputs.output_ttns)
        if "output_basis_tree" in calc.outputs:
            self.out("output_basis_tree", calc.outputs.output_basis_tree)
        if "output_tn_layout" in calc.outputs:
            self.out("output_tn_layout", calc.outputs.output_tn_layout)
        elif "tn_layout" in self.inputs:
            self.out("output_tn_layout", self.inputs.tn_layout)
        self.out("output_parameters", calc.outputs.output_parameters)
