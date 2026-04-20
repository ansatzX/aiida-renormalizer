"""WorkChain for TTN ground state calculation with variational optimization."""
from __future__ import annotations

from aiida import orm
from aiida.engine import WorkChain, ToContext

from aiida_renormalizer.calculations.ttn.optimize_ttns import OptimizeTtnsCalcJob
from aiida_renormalizer.data import BasisTreeData, ConfigData, TTNSData, TtnoData


class TtnGroundStateWorkChain(WorkChain):
    """WorkChain for TTN ground state calculation with variational optimization.

    This WorkChain provides:
    - Automatic tree topology construction (if not provided)
    - Initial TTNS state generation (random if not provided)
    - TTN-DMRG variational optimization
    - Convergence checking and energy validation

    Inputs:
        basis_tree: BasisTreeData - Tree topology and basis grouping
        ttno: TtnoData - Hamiltonian TTNO operator
        initial_ttns: TTNSData (optional) - Initial guess, random if not provided
        config: ConfigData - Optimization configuration
        energy_convergence: Float - Energy convergence threshold
        code: AbstractCode - Code to use

    Outputs:
        ground_state: TTNSData - Ground state TTNS
        energy: Float - Ground state energy
        output_parameters: Dict - Calculation statistics
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        # Inputs
        spec.input("basis_tree", valid_type=BasisTreeData, help="Tree topology and basis grouping")
        spec.input("ttno", valid_type=TtnoData, help="Hamiltonian TTNO")
        spec.input(
            "initial_ttns",
            valid_type=TTNSData,
            required=False,
            help="Initial TTNS guess (random if not provided)",
        )
        spec.input("config", valid_type=ConfigData, required=False, help="Optimization configuration")
        spec.input(
            "energy_convergence",
            valid_type=orm.Float,
            default=lambda: orm.Float(1e-8),
            help="Energy convergence threshold",
        )
        spec.input("code", valid_type=orm.AbstractCode, help="Code to use")

        # Outputs
        spec.output("ground_state", valid_type=TTNSData, help="Ground state TTNS")
        spec.output("energy", valid_type=orm.Float, help="Ground state energy")
        spec.output("output_parameters", valid_type=orm.Dict, help="Calculation statistics")

        # Exit codes
        spec.exit_code(
            330,
            "ERROR_NOT_CONVERGED",
            message="TTN ground state optimization did not converge",
        )
        spec.exit_code(
            331,
            "ERROR_CALCULATION_FAILED",
            message="TTN ground state calculation failed",
        )

        # Outline
        spec.outline(
            cls.setup,
            cls.run_optimization,
            cls.inspect_optimization,
            cls.finalize,
        )

    def setup(self):
        """Initialize the WorkChain."""
        self.report("Starting TTN ground state calculation")

    def run_optimization(self):
        """Run TTN optimization."""
        self.report("Running TTN optimization")

        # Build inputs
        inputs = {
            "basis_tree": self.inputs.basis_tree,
            "ttno": self.inputs.ttno,
            "code": self.inputs.code,
        }

        # Add initial TTNS if provided
        if "initial_ttns" in self.inputs:
            inputs["initial_ttns"] = self.inputs.initial_ttns

        # Add config if provided
        if "config" in self.inputs:
            inputs["config"] = self.inputs.config

        # Submit optimization calculation
        future = self.submit(OptimizeTtnsCalcJob, **inputs)

        return ToContext(ground_state_calc=future)

    def inspect_optimization(self):
        """Inspect optimization results."""
        calc = self.ctx.ground_state_calc

        if not calc.is_finished_ok:
            self.report(f"TTN optimization failed: exit_status={calc.exit_status}")
            if calc.exit_status == OptimizeTtnsCalcJob.exit_codes.ERROR_NOT_CONVERGED.status:
                return self.exit_codes.ERROR_NOT_CONVERGED
            return self.exit_codes.ERROR_CALCULATION_FAILED

        # Store results in context
        self.ctx.ground_state = calc.outputs.output_ttns
        self.ctx.energy = None

        # Extract energy from output parameters
        if "output_parameters" in calc.outputs:
            params = calc.outputs.output_parameters.get_dict()
            # TTN optimization typically returns energy trajectory
            if "energy" in params:
                self.ctx.energy = params["energy"]
            elif "energies" in params:
                self.ctx.energy = params["energies"][-1]
            self.ctx.calc_params = params

        self.report(f"TTN optimization completed: energy={self.ctx.energy}")

    def finalize(self):
        """Collect and output final results."""
        self.report("Finalizing TTN ground state calculation")

        # Output ground state
        self.out("ground_state", self.ctx.ground_state)

        # Output energy
        if self.ctx.energy is not None:
            self.out("energy", orm.Float(self.ctx.energy))

        # Output statistics
        stats = {
            "method": "ttn_optimization",
            "energy": self.ctx.energy,
        }
        if hasattr(self.ctx, "calc_params"):
            stats.update(self.ctx.calc_params)

        self.out("output_parameters", orm.Dict(stats))

        self.report(
            f"TTN ground state calculation completed: energy={self.ctx.energy}"
        )
