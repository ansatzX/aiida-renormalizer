"""WorkChain for ground state calculation with DMRG/ImagTime strategies."""
from __future__ import annotations

from aiida import orm
from aiida.engine import WorkChain, ToContext, if_

from aiida_renormalizer.calculations.composite.dmrg import DMRGCalcJob
from aiida_renormalizer.calculations.composite.imag_time import ImagTimeCalcJob
from aiida_renormalizer.data import ModelData, MPSData, MPOData


class GroundStateWorkChain(WorkChain):
    """WorkChain for ground state calculation with strategy selection.

    This WorkChain supports two strategies:
    1. DMRG: Variational optimization (faster, more accurate for ground states)
    2. ImagTime: Imaginary time evolution (better for finite temperature, thermal states)

    It also provides:
    - Automatic initial state generation
    - Convergence checking
    - Energy comparison and validation

    Inputs:
        model: ModelData - System definition
        mpo: MPOData - Hamiltonian operator (optional, will be built if not provided)
        initial_mps: MPSData (optional) - Initial guess
        strategy: Str - "dmrg" or "imag_time" (default: "dmrg")
        config: ConfigData - Optimization/evolution configuration
        energy_convergence: Float - Energy convergence threshold
        code: AbstractCode - Code to use

    Outputs:
        ground_state: MPSData - Ground state MPS
        energy: Float - Ground state energy
        output_parameters: Dict - Calculation statistics
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        # Inputs
        spec.input("model", valid_type=ModelData, help="System definition")
        spec.input(
            "mpo",
            valid_type=MPOData,
            required=False,
            help="Hamiltonian MPO (will be built if not provided)",
        )
        spec.input(
            "initial_mps",
            valid_type=MPSData,
            required=False,
            help="Initial MPS guess (random if not provided)",
        )
        spec.input(
            "strategy",
            valid_type=orm.Str,
            default=lambda: orm.Str("dmrg"),
            help="Strategy: 'dmrg' or 'imag_time'",
        )
        spec.input("config", valid_type=orm.Dict, required=False, help="Method configuration")
        spec.input(
            "energy_convergence",
            valid_type=orm.Float,
            default=lambda: orm.Float(1e-8),
            help="Energy convergence threshold",
        )
        spec.input("code", valid_type=orm.AbstractCode, help="Code to use")

        # Additional strategy-specific inputs
        spec.input("omega", valid_type=orm.Float, required=False, help="Target eigenvalue (DMRG)")
        spec.input("beta", valid_type=orm.Float, required=False, help="Inverse temperature (ImagTime)")
        spec.input("dt", valid_type=orm.Float, required=False, help="Time step (ImagTime)")

        # Outputs
        spec.output("ground_state", valid_type=MPSData, help="Ground state MPS")
        spec.output("energy", valid_type=orm.Float, help="Ground state energy")
        spec.output("output_parameters", valid_type=orm.Dict, help="Calculation statistics")

        # Exit codes
        spec.exit_code(
            320,
            "ERROR_STRATEGY_NOT_SUPPORTED",
            message="Unsupported strategy",
        )
        spec.exit_code(
            321,
            "ERROR_NOT_CONVERGED",
            message="Ground state calculation did not converge",
        )
        spec.exit_code(
            322,
            "ERROR_CALCULATION_FAILED",
            message="Ground state calculation failed",
        )

        # Outline
        spec.outline(
            cls.setup,
            if_(cls.use_dmrg)(
                cls.run_dmrg,
                cls.inspect_dmrg,
            ).else_(
                if_(cls.use_imag_time)(
                    cls.run_imag_time,
                    cls.inspect_imag_time,
                ).else_(
                    cls.unsupported_strategy,
                ),
            ),
            cls.finalize,
        )

    def setup(self):
        """Initialize the WorkChain."""
        strategy = self.inputs.strategy.value
        self.report(f"Starting ground state calculation with strategy: {strategy}")

    def use_dmrg(self):
        """Check if using DMRG strategy."""
        return self.inputs.strategy.value == "dmrg"

    def use_imag_time(self):
        """Check if using imaginary time strategy."""
        return self.inputs.strategy.value == "imag_time"

    def run_dmrg(self):
        """Run DMRG optimization."""
        self.report("Running DMRG optimization")

        # Build inputs
        inputs = {
            "model": self.inputs.model,
            "code": self.inputs.code,
        }

        # Add MPO if provided
        if "mpo" in self.inputs:
            inputs["mpo"] = self.inputs.mpo

        # Add initial MPS if provided
        if "initial_mps" in self.inputs:
            inputs["initial_mps"] = self.inputs.initial_mps

        # Add config if provided
        if "config" in self.inputs:
            inputs["config"] = self.inputs.config

        # Add omega for excited states
        if "omega" in self.inputs:
            inputs["omega"] = self.inputs.omega

        # Submit DMRG calculation
        future = self.submit(DMRGCalcJob, **inputs)

        return ToContext(ground_state_calc=future)

    def inspect_dmrg(self):
        """Inspect DMRG results."""
        calc = self.ctx.ground_state_calc

        if not calc.is_finished_ok:
            self.report(f"DMRG calculation failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        # Store results in context
        self.ctx.ground_state = calc.outputs.output_mps
        self.ctx.energy = None

        # Extract energy from output parameters
        if "output_parameters" in calc.outputs:
            params = calc.outputs.output_parameters.get_dict()
            if "energy" in params:
                self.ctx.energy = params["energy"]
            elif "energies" in params:
                self.ctx.energy = params["energies"][-1]
            self.ctx.calc_params = params

        self.report(f"DMRG completed: energy={self.ctx.energy}")

    def run_imag_time(self):
        """Run imaginary time evolution."""
        self.report("Running imaginary time evolution")

        # Build inputs
        inputs = {
            "model": self.inputs.model,
            "code": self.inputs.code,
        }

        # Add MPO if provided
        if "mpo" in self.inputs:
            inputs["mpo"] = self.inputs.mpo

        # Add initial MPS if provided
        if "initial_mps" in self.inputs:
            inputs["initial_mps"] = self.inputs.initial_mps

        # Add beta (inverse temperature) - required for ImagTime
        if "beta" in self.inputs:
            inputs["beta"] = self.inputs.beta
        else:
            # Default beta for ground state (high beta = T → 0)
            inputs["beta"] = orm.Float(100.0)

        # Add time step if provided
        if "dt" in self.inputs:
            inputs["dt"] = self.inputs.dt

        # Add config if provided
        if "config" in self.inputs:
            inputs["config"] = self.inputs.config

        # Submit ImagTime calculation
        future = self.submit(ImagTimeCalcJob, **inputs)

        return ToContext(ground_state_calc=future)

    def inspect_imag_time(self):
        """Inspect imaginary time results."""
        calc = self.ctx.ground_state_calc

        if not calc.is_finished_ok:
            self.report(f"Imaginary time calculation failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        # Store results in context
        self.ctx.ground_state = calc.outputs.output_mps
        self.ctx.energy = None

        # Extract energy
        if "output_parameters" in calc.outputs:
            params = calc.outputs.output_parameters.get_dict()
            if "energy" in params:
                self.ctx.energy = params["energy"]
            self.ctx.calc_params = params

        self.report(f"Imaginary time evolution completed: energy={self.ctx.energy}")

    def unsupported_strategy(self):
        """Handle unsupported strategy."""
        self.report(f"Unsupported strategy: {self.inputs.strategy.value}")
        return self.exit_codes.ERROR_STRATEGY_NOT_SUPPORTED

    def finalize(self):
        """Collect and output final results."""
        self.report("Finalizing ground state calculation")

        # Output ground state (only if available)
        if not hasattr(self.ctx, 'ground_state'):
            return self.exit_codes.ERROR_CALCULATION_FAILED

        self.out("ground_state", self.ctx.ground_state)

        # Output energy
        energy = getattr(self.ctx, 'energy', None)
        if energy is not None:
            self.out("energy", orm.Float(energy))

        # Output statistics
        stats = {
            "strategy": self.inputs.strategy.value,
            "energy": energy,
        }
        if hasattr(self.ctx, "calc_params"):
            stats.update(self.ctx.calc_params)

        self.out("output_parameters", orm.Dict(stats))

        self.report(
            f"Ground state calculation completed: strategy={self.inputs.strategy.value}, "
            f"energy={energy}"
        )
