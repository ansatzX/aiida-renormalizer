"""WorkChain for convergence testing with respect to bond dimension."""
from __future__ import annotations

from aiida import orm
from aiida.engine import WorkChain, ToContext, while_

from aiida_renormalizer.calculations.composite.dmrg import DMRGCalcJob
from aiida_renormalizer.data import ModelData, MpsData, MpoData


class ConvergenceWorkChain(WorkChain):
    """WorkChain for bond dimension convergence testing.

    This WorkChain performs systematic convergence testing by:
    1. Running calculations at increasing bond dimensions
    2. Tracking energy (and optionally other observables) convergence
    3. Determining when convergence is achieved
    4. Reporting convergence behavior

    Applications:
    - DMRG bond dimension convergence
    - MPS truncation error analysis
    - Quality assessment for production calculations

    Inputs:
        model: ModelData - System definition
        mpo: MpoData - Hamiltonian operator
        initial_mps: MpsData (optional) - Initial guess
        m_values: List - List of bond dimensions to test
        convergence_threshold: Float - Energy convergence threshold
        config: ConfigData - DMRG configuration
        code: AbstractCode - Code to use

    Outputs:
        converged_mps: MpsData - MPS at converged bond dimension
        convergence_data: Dict - Energy and observables vs bond dimension
        optimal_m: Int - Optimal (converged) bond dimension
        output_parameters: Dict - Convergence statistics
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        # Inputs
        spec.input("model", valid_type=ModelData, help="System definition")
        spec.input("mpo", valid_type=MpoData, help="Hamiltonian MPO")
        spec.input(
            "initial_mps",
            valid_type=MpsData,
            required=False,
            help="Initial MPS guess",
        )
        spec.input(
            "m_values",
            valid_type=orm.List,
            help="List of bond dimensions to test",
        )
        spec.input(
            "convergence_threshold",
            valid_type=orm.Float,
            default=lambda: orm.Float(1e-6),
            help="Energy convergence threshold",
        )
        spec.input("config", valid_type=orm.Dict, required=False, help="DMRG configuration")
        spec.input("code", valid_type=orm.AbstractCode, help="Code to use")
        spec.input(
            "observables",
            valid_type=orm.Dict,
            required=False,
            help="Additional observables to track",
        )

        # Outputs
        spec.output("converged_mps", valid_type=MpsData, help="MPS at converged bond dimension")
        spec.output("convergence_data", valid_type=orm.Dict, help="Convergence data")
        spec.output("optimal_m", valid_type=orm.Int, help="Optimal bond dimension")
        spec.output("output_parameters", valid_type=orm.Dict, help="Convergence statistics")

        # Exit codes
        spec.exit_code(
            340,
            "ERROR_NO_CONVERGENCE",
            message="Did not converge within provided bond dimensions",
        )
        spec.exit_code(
            341,
            "ERROR_CALCULATION_FAILED",
            message="DMRG calculation failed",
        )

        # Outline
        spec.outline(
            cls.setup,
            while_(cls.not_converged)(
                cls.run_calculation,
                cls.inspect_calculation,
            ),
            cls.finalize,
        )

    def setup(self):
        """Initialize the WorkChain."""
        self.report("Starting convergence testing")

        # Initialize context
        self.ctx.m_values = self.inputs.m_values.get_list()
        self.ctx.current_index = 0
        self.ctx.energies = []
        self.ctx.observables = []
        self.ctx.mps_list = []
        self.ctx.converged = False

        self.report(f"Testing bond dimensions: {self.ctx.m_values}")

    def not_converged(self):
        """Check if convergence testing should continue."""
        # Stop if converged
        if self.ctx.converged:
            return False

        # Stop if no more bond dimensions to test
        if self.ctx.current_index >= len(self.ctx.m_values):
            self.report("Exhausted all bond dimensions without convergence")
            return False

        return True

    def run_calculation(self):
        """Run DMRG at current bond dimension."""
        m_current = self.ctx.m_values[self.ctx.current_index]
        self.report(f"Running DMRG with M = {m_current}")

        # Build inputs
        inputs = {
            "model": self.inputs.model,
            "mpo": self.inputs.mpo,
            "code": self.inputs.code,
        }

        # Add initial MPS if available (use result from previous iteration)
        if self.ctx.current_index > 0 and self.ctx.mps_list:
            inputs["initial_mps"] = self.ctx.mps_list[-1]
        elif "initial_mps" in self.inputs:
            inputs["initial_mps"] = self.inputs.initial_mps

        # Update config with current bond dimension
        if "config" in self.inputs:
            config_dict = self.inputs.config.get_dict()
        else:
            config_dict = {}

        config_dict["M_max"] = m_current
        inputs["config"] = orm.Dict(config_dict)

        # Submit calculation
        future = self.submit(DMRGCalcJob, **inputs)

        return ToContext(current_calc=future)

    def inspect_calculation(self):
        """Inspect the calculation results."""
        calc = self.ctx.current_calc

        # Check if calculation succeeded
        if not calc.is_finished_ok:
            self.report(f"DMRG calculation failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_CALCULATION_FAILED

        # Extract results
        m_current = self.ctx.m_values[self.ctx.current_index]
        energy = None
        observables = {}

        if "output_parameters" in calc.outputs:
            params = calc.outputs.output_parameters.get_dict()
            # Extract energy
            if "energy" in params:
                energy = params["energy"]
            elif "energies" in params and params["energies"]:
                energy = params["energies"][-1]

        # Store results
        self.ctx.energies.append(energy)
        self.ctx.mps_list.append(calc.outputs.output_mps)

        self.report(f"M = {m_current}: energy = {energy}")

        # Check for convergence
        if len(self.ctx.energies) >= 2:
            energy_diff = abs(self.ctx.energies[-1] - self.ctx.energies[-2])
            threshold = self.inputs.convergence_threshold.value

            self.report(f"Energy difference: {energy_diff} (threshold: {threshold})")

            if energy_diff < threshold:
                self.report("Convergence achieved!")
                self.ctx.converged = True
                self.ctx.optimal_m_index = self.ctx.current_index

        # Move to next bond dimension
        self.ctx.current_index += 1

    def finalize(self):
        """Collect and output final results."""
        self.report("Finalizing convergence testing")

        # Check if converged
        if not self.ctx.converged:
            self.report("Warning: Did not achieve convergence")
            # Use the last result anyway
            if self.ctx.mps_list:
                self.ctx.optimal_m_index = len(self.ctx.mps_list) - 1
            else:
                return self.exit_codes.ERROR_NO_CONVERGENCE

        # Output converged MPS
        optimal_mps = self.ctx.mps_list[self.ctx.optimal_m_index]
        self.out("converged_mps", optimal_mps)

        # Output optimal bond dimension
        optimal_m = self.ctx.m_values[self.ctx.optimal_m_index]
        self.out("optimal_m", orm.Int(optimal_m))

        # Build convergence data
        convergence_data = {
            "m_values": self.ctx.m_values[:len(self.ctx.energies)],
            "energies": self.ctx.energies,
        }

        # Calculate energy differences
        if len(self.ctx.energies) >= 2:
            energy_diffs = [
                abs(self.ctx.energies[i] - self.ctx.energies[i-1])
                for i in range(1, len(self.ctx.energies))
            ]
            convergence_data["energy_differences"] = energy_diffs

        self.out("convergence_data", orm.Dict(convergence_data))

        # Output statistics
        stats = {
            "optimal_m": optimal_m,
            "final_energy": self.ctx.energies[self.ctx.optimal_m_index],
            "n_calculations": len(self.ctx.energies),
            "converged": self.ctx.converged,
            "convergence_threshold": self.inputs.convergence_threshold.value,
        }

        if len(self.ctx.energies) >= 2:
            stats["final_energy_diff"] = abs(
                self.ctx.energies[-1] - self.ctx.energies[-2]
            )

        self.out("output_parameters", orm.Dict(stats))

        if self.ctx.converged:
            self.report(
                f"Convergence testing completed: optimal_M={optimal_m}, "
                f"energy={self.ctx.energies[self.ctx.optimal_m_index]}"
            )
        else:
            self.report(
                f"Convergence testing completed without convergence: "
                f"best_M={optimal_m}, energy={self.ctx.energies[self.ctx.optimal_m_index]}"
            )
