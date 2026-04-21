"""WorkChain wrapper for symbolic TTN model payload generation."""
from __future__ import annotations

from aiida import orm
from aiida.engine import ToContext, WorkChain

from aiida_renormalizer.calculations.ttn.symbolic_model import TTNSymbolicModelCalcJob
from aiida_renormalizer.data import BasisTreeData


class TTNSymbolicModelWorkChain(WorkChain):
    """Build symbolic TTN/TTNO inputs from declarative SBM process spec."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("code", valid_type=orm.AbstractCode, help="Code for CalcJob execution")
        spec.input("alpha", valid_type=orm.Float, help="SBM coupling strength")
        spec.input("s_exponent", valid_type=orm.Float, default=lambda: orm.Float(1.0))
        spec.input("omega_c", valid_type=orm.Float, help="Bath cutoff frequency")
        spec.input("n_modes", valid_type=orm.Int, help="Number of bath modes")
        spec.input("raw_delta", valid_type=orm.Float, help="Bare tunneling parameter")
        spec.input(
            "renormalization_p",
            valid_type=orm.Float,
            help="Adiabatic renormalization parameter p",
        )
        spec.input("tree_type", valid_type=orm.Str, default=lambda: orm.Str("binary"))
        spec.input("m_max", valid_type=orm.Int, default=lambda: orm.Int(16))
        spec.input("symbol_map", valid_type=orm.Dict, default=lambda: orm.Dict(dict={}))
        spec.input(
            "process",
            valid_type=orm.List,
            default=lambda: orm.List(
                list=[
                    "build_sdf",
                    "adiabatic_renormalization",
                    "discretize_bath",
                    "build_symbolic_hamiltonian",
                    "ttns_tdvp",
                ]
            ),
        )

        spec.output("symbolic_inputs", valid_type=orm.Dict)
        spec.output("basis_tree", valid_type=BasisTreeData, required=False)
        spec.output("output_parameters", valid_type=orm.Dict)

        spec.exit_code(530, "ERROR_INVALID_OUTPUT", message="Invalid symbolic model outputs")
        spec.exit_code(531, "ERROR_CALCJOB_FAILED", message="Symbolic model calcjob failed")

        spec.outline(cls.run_calcjob, cls.inspect_calcjob, cls.finalize)

    def run_calcjob(self):
        return ToContext(
            calc=self.submit(
                TTNSymbolicModelCalcJob,
                code=self.inputs.code,
                alpha=self.inputs.alpha,
                s_exponent=self.inputs.s_exponent,
                omega_c=self.inputs.omega_c,
                n_modes=self.inputs.n_modes,
                raw_delta=self.inputs.raw_delta,
                renormalization_p=self.inputs.renormalization_p,
                tree_type=self.inputs.tree_type,
                m_max=self.inputs.m_max,
                symbol_map=self.inputs.symbol_map,
                process=self.inputs.process,
            )
        )

    def inspect_calcjob(self):
        calc = self.ctx.calc
        if not calc.is_finished_ok:
            self.report(f"TTNSymbolicModelCalcJob failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_CALCJOB_FAILED

        payload = calc.outputs.output_parameters.get_dict()
        symbolic_inputs = payload.get("symbolic_inputs")
        if not isinstance(symbolic_inputs, dict):
            self.report("Missing or invalid symbolic_inputs in calcjob output_parameters")
            return self.exit_codes.ERROR_INVALID_OUTPUT

        metadata = payload.get("metadata")
        if not isinstance(metadata, dict):
            self.report("Missing or invalid metadata in calcjob output_parameters")
            return self.exit_codes.ERROR_INVALID_OUTPUT

        self.ctx.symbolic_inputs = orm.Dict(dict=symbolic_inputs)
        self.ctx.metadata = orm.Dict(dict=metadata)

    def finalize(self):
        self.out("symbolic_inputs", self.ctx.symbolic_inputs)
        if "output_basis_tree" in self.ctx.calc.outputs:
            self.out("basis_tree", self.ctx.calc.outputs.output_basis_tree)
        self.out("output_parameters", self.ctx.metadata)
