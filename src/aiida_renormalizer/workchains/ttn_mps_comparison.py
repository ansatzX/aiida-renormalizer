"""WorkChain for comparing TTN vs MPS performance on the same calculation."""
from __future__ import annotations

from aiida import orm
from aiida.engine import WorkChain, ToContext

from aiida_renormalizer.workchains.ground_state import GroundStateWorkChain
from aiida_renormalizer.workchains.ttn_ground_state import TtnGroundStateWorkChain
from aiida_renormalizer.data import ModelData, MpsData, MpoData, BasisTreeData, TtnoData, TTNSData


class TtnMpsComparisonWorkChain(WorkChain):
    """WorkChain for benchmarking TTN vs MPS on the same problem.

    This WorkChain performs comparative analysis by:
    1. Running ground state calculation with both MPS (DMRG) and TTN
    2. Comparing accuracy (energy, observables)
    3. Comparing efficiency (bond dimensions, computational cost)
    4. Reporting comparative analysis

    Applications:
    - Benchmark TTN vs MPS on specific systems
    - Determine optimal tensor network structure for a problem
    - Study TTN advantages for specific topologies

    Inputs:
        model: ModelData - System definition
        mpo: MpoData - Hamiltonian MPO (for MPS)
        basis_tree: BasisTreeData - Tree topology (for TTN)
        ttno: TtnoData - Hamiltonian TTNO (for TTN)
        initial_mps: MpsData (optional) - Initial MPS guess
        initial_ttns: TTNSData (optional) - Initial TTNS guess
        calculation_type: Str - "ground_state" or "time_evolution"
        config_mps: Dict - MPS calculation configuration
        config_ttn: Dict - TTN calculation configuration
        code: AbstractCode - Code to use

    Outputs:
        mps_result: MpsData - MPS result
        ttn_result: TTNSData - TTN result
        comparison_data: Dict - Comparative analysis
        output_parameters: Dict - Comparison statistics
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        # Inputs
        spec.input("model", valid_type=ModelData, help="System definition")
        spec.input("mpo", valid_type=MpoData, help="Hamiltonian MPO (for MPS)")
        spec.input("basis_tree", valid_type=BasisTreeData, help="Tree topology (for TTN)")
        spec.input("ttno", valid_type=TtnoData, help="Hamiltonian TTNO (for TTN)")
        spec.input(
            "initial_mps",
            valid_type=MpsData,
            required=False,
            help="Initial MPS guess",
        )
        spec.input(
            "initial_ttns",
            valid_type=TTNSData,
            required=False,
            help="Initial TTNS guess",
        )
        spec.input(
            "calculation_type",
            valid_type=orm.Str,
            default=lambda: orm.Str("ground_state"),
            help="Type of calculation: 'ground_state' or 'time_evolution'",
        )
        spec.input("config_mps", valid_type=orm.Dict, required=False, help="MPS config")
        spec.input("config_ttn", valid_type=orm.Dict, required=False, help="TTN config")
        spec.input("code", valid_type=orm.AbstractCode, help="Code to use")

        # Additional inputs for time evolution
        spec.input("total_time", valid_type=orm.Float, required=False, help="Total evolution time")
        spec.input("dt", valid_type=orm.Float, required=False, help="Time step")

        # Outputs
        spec.output("mps_result", valid_type=MpsData, help="MPS result")
        spec.output("ttn_result", valid_type=TTNSData, help="TTN result")
        spec.output("comparison_data", valid_type=orm.Dict, help="Comparative analysis")
        spec.output("output_parameters", valid_type=orm.Dict, help="Comparison statistics")

        # Exit codes
        spec.exit_code(
            350,
            "ERROR_MPS_FAILED",
            message="MPS calculation failed",
        )
        spec.exit_code(
            351,
            "ERROR_TTN_FAILED",
            message="TTN calculation failed",
        )
        spec.exit_code(
            352,
            "ERROR_UNSUPPORTED_CALCULATION",
            message="Unsupported calculation type",
        )

        # Outline
        spec.outline(
            cls.setup,
            cls.run_mps,
            cls.inspect_mps,
            cls.run_ttn,
            cls.inspect_ttn,
            cls.compare_results,
            cls.finalize,
        )

    def setup(self):
        """Initialize the WorkChain."""
        calc_type = self.inputs.calculation_type.value
        self.report(f"Starting TTN vs MPS comparison: calculation_type={calc_type}")

        if calc_type not in ["ground_state", "time_evolution"]:
            return self.exit_codes.ERROR_UNSUPPORTED_CALCULATION

    def run_mps(self):
        """Run MPS calculation."""
        calc_type = self.inputs.calculation_type.value
        self.report(f"Running MPS {calc_type}")

        if calc_type == "ground_state":
            # Build inputs for GroundStateWorkChain
            inputs = {
                "model": self.inputs.model,
                "mpo": self.inputs.mpo,
                "strategy": orm.Str("dmrg"),
                "code": self.inputs.code,
            }

            if "initial_mps" in self.inputs:
                inputs["initial_mps"] = self.inputs.initial_mps

            if "config_mps" in self.inputs:
                inputs["config"] = self.inputs.config_mps

            # Submit MPS ground state calculation
            future = self.submit(GroundStateWorkChain, **inputs)
            return ToContext(mps_calc=future)

        else:
            # Time evolution - would use TimeEvolutionWorkChain
            self.report("MPS time evolution not yet implemented in comparison")
            return self.exit_codes.ERROR_UNSUPPORTED_CALCULATION

    def inspect_mps(self):
        """Inspect MPS results."""
        calc_type = self.inputs.calculation_type.value

        if calc_type == "ground_state":
            calc = self.ctx.mps_calc

            if not calc.is_finished_ok:
                self.report(f"MPS calculation failed: exit_status={calc.exit_status}")
                return self.exit_codes.ERROR_MPS_FAILED

            # Store results
            self.ctx.mps_result = calc.outputs.ground_state
            self.ctx.mps_energy = None

            if "energy" in calc.outputs:
                self.ctx.mps_energy = calc.outputs.energy.value

            if "output_parameters" in calc.outputs:
                params = calc.outputs.output_parameters.get_dict()
                self.ctx.mps_params = params

            self.report(f"MPS calculation completed: energy={self.ctx.mps_energy}")

    def run_ttn(self):
        """Run TTN calculation."""
        calc_type = self.inputs.calculation_type.value
        self.report(f"Running TTN {calc_type}")

        if calc_type == "ground_state":
            # Build inputs for TtnGroundStateWorkChain
            inputs = {
                "basis_tree": self.inputs.basis_tree,
                "ttno": self.inputs.ttno,
                "code": self.inputs.code,
            }

            if "initial_ttns" in self.inputs:
                inputs["initial_ttns"] = self.inputs.initial_ttns

            if "config_ttn" in self.inputs:
                inputs["config"] = self.inputs.config_ttn

            # Submit TTN ground state calculation
            future = self.submit(TtnGroundStateWorkChain, **inputs)
            return ToContext(ttn_calc=future)

        else:
            # Time evolution - would use TtnTimeEvolutionWorkChain
            self.report("TTN time evolution not yet implemented in comparison")
            return self.exit_codes.ERROR_UNSUPPORTED_CALCULATION

    def inspect_ttn(self):
        """Inspect TTN results."""
        calc_type = self.inputs.calculation_type.value

        if calc_type == "ground_state":
            calc = self.ctx.ttn_calc

            if not calc.is_finished_ok:
                self.report(f"TTN calculation failed: exit_status={calc.exit_status}")
                return self.exit_codes.ERROR_TTN_FAILED

            # Store results
            self.ctx.ttn_result = calc.outputs.ground_state
            self.ctx.ttn_energy = None

            if "energy" in calc.outputs:
                self.ctx.ttn_energy = calc.outputs.energy.value

            if "output_parameters" in calc.outputs:
                params = calc.outputs.output_parameters.get_dict()
                self.ctx.ttn_params = params

            self.report(f"TTN calculation completed: energy={self.ctx.ttn_energy}")

    def compare_results(self):
        """Compare MPS and TTN results."""
        self.report("Comparing MPS and TTN results")

        # Compare energies
        energy_diff = None
        if self.ctx.mps_energy is not None and self.ctx.ttn_energy is not None:
            energy_diff = abs(self.ctx.mps_energy - self.ctx.ttn_energy)
            self.report(f"Energy difference: {energy_diff}")

        # Compare bond dimensions
        mps_bond_dims = None
        ttn_bond_dims = None

        if hasattr(self.ctx, "mps_params"):
            if "bond_dims" in self.ctx.mps_params:
                mps_bond_dims = self.ctx.mps_params["bond_dims"]
            elif "M_max" in self.ctx.mps_params:
                mps_bond_dims = self.ctx.mps_params["M_max"]

        if hasattr(self.ctx, "ttn_params"):
            if "bond_dims" in self.ctx.ttn_params:
                ttn_bond_dims = self.ctx.ttn_params["bond_dims"]

        # Compare computational cost (simplified: just iteration count)
        mps_iterations = None
        ttn_iterations = None

        if hasattr(self.ctx, "mps_params"):
            if "n_iterations" in self.ctx.mps_params:
                mps_iterations = self.ctx.mps_params["n_iterations"]
            elif "iterations" in self.ctx.mps_params:
                mps_iterations = self.ctx.mps_params["iterations"]

        if hasattr(self.ctx, "ttn_params"):
            if "n_iterations" in self.ctx.ttn_params:
                ttn_iterations = self.ctx.ttn_params["n_iterations"]
            elif "iterations" in self.ctx.ttn_params:
                ttn_iterations = self.ctx.ttn_params["iterations"]

        # Store comparison
        self.ctx.comparison = {
            "mps_energy": self.ctx.mps_energy,
            "ttn_energy": self.ctx.ttn_energy,
            "energy_difference": energy_diff,
            "mps_bond_dims": mps_bond_dims,
            "ttn_bond_dims": ttn_bond_dims,
            "mps_iterations": mps_iterations,
            "ttn_iterations": ttn_iterations,
        }

    def finalize(self):
        """Collect and output final results."""
        self.report("Finalizing comparison")

        # Output MPS result
        self.out("mps_result", self.ctx.mps_result)

        # Output TTN result
        self.out("ttn_result", self.ctx.ttn_result)

        # Output comparison data
        self.out("comparison_data", orm.Dict(self.ctx.comparison))

        # Output statistics
        stats = {
            "calculation_type": self.inputs.calculation_type.value,
        }
        stats.update(self.ctx.comparison)

        self.out("output_parameters", orm.Dict(stats))

        self.report(
            f"TTN vs MPS comparison completed: energy_diff={self.ctx.comparison.get('energy_difference')}"
        )
