"""WorkChain for Kubo linear response transport."""
from __future__ import annotations

from aiida import orm
from aiida.engine import WorkChain, ToContext

from aiida_renormalizer.calculations.composite.thermal_prop import ThermalPropCalcJob
from aiida_renormalizer.calculations.spectra.kubo import KuboCalcJob
from aiida_renormalizer.data import ModelData, MPSData, MPOData, TensorNetworkLayoutData


class KuboTransportWorkChain(WorkChain):
    """WorkChain for Kubo linear response transport calculations.

    This WorkChain computes transport coefficients (mobility, conductivity)
    via the Green-Kubo formula:
        σ = (1/kT) ∫₀^∞ dt <j(t)j(0)>

    Steps:
    1. Prepare thermal density matrix at target temperature
    2. Compute current-current correlation function via real-time evolution
    3. Integrate correlation function to obtain conductivity
    4. Optionally compute mobility from conductivity

    Inputs:
        model: ModelData - System definition
        mpo: MPOData - Hamiltonian operator (optional)
        initial_mps: MPSData - Pre-computed thermal state (optional)
        temperature: Float - Temperature (alternative to beta)
        beta: Float - Inverse temperature (alternative to temperature)
        current_op: OpData - Current operator (will be constructed if not provided)
        distance_matrix: ArrayData - Distance matrix for current operator
        config: Dict - Evolution configuration
        nsteps: Int - Number of time steps for correlation
        dt: Float - Time step
        insteps: Int - Steps for imaginary time propagation (if no initial_mps)
        code: AbstractCode - Code to use

    Outputs:
        conductivity: ArrayData - Conductivity tensor
        mobility: Float - Mobility (optional, requires carrier density)
        autocorrelation: ArrayData - Current-current correlation function
        output_parameters: Dict - Transport statistics

    Exit Codes:
        360: ERROR_INVALID_TEMPERATURE
        361: ERROR_THERMAL_STATE_FAILED
        362: ERROR_KUBO_CALCULATION_FAILED
        363: ERROR_INVALID_CONDUCTIVITY
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        # Inputs - System definition
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
            help="Pre-computed thermal density matrix (MpDm)",
        )

        # Inputs - Temperature
        spec.input(
            "temperature",
            valid_type=orm.Float,
            required=False,
            help="Temperature (use this OR beta)",
        )
        spec.input(
            "beta",
            valid_type=orm.Float,
            required=False,
            help="Inverse temperature (use this OR temperature)",
        )

        # Inputs - Current operator
        spec.input(
            "distance_matrix",
            valid_type=orm.ArrayData,
            required=False,
            help="Distance matrix D_ij = P_i - P_j for current operator",
        )

        # Inputs - Propagation parameters
        spec.input("config", valid_type=orm.Dict, required=False, help="Evolution configuration")
        spec.input(
            "nsteps",
            valid_type=orm.Int,
            default=lambda: orm.Int(100),
            help="Number of time steps for correlation function",
        )
        spec.input(
            "dt",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.1),
            help="Time step for real-time evolution",
        )
        spec.input(
            "insteps",
            valid_type=orm.Int,
            default=lambda: orm.Int(1),
            help="Steps for imaginary time propagation (if no initial_mps)",
        )

        # Inputs - Code
        spec.input("code", valid_type=orm.AbstractCode, help="Code to use")
        spec.input("tn_layout", valid_type=TensorNetworkLayoutData, required=False, help="Shared tensor-network layout metadata")

        # Outputs
        spec.output("conductivity", valid_type=orm.ArrayData, help="Conductivity tensor")
        spec.output("mobility", valid_type=orm.Float, required=False, help="Mobility")
        spec.output("autocorrelation", valid_type=orm.ArrayData, help="Current-current correlation")
        spec.output("output_tn_layout", valid_type=TensorNetworkLayoutData, required=False, help="Shared tensor-network layout metadata")
        spec.output("output_parameters", valid_type=orm.Dict, help="Transport statistics")

        # Exit codes
        spec.exit_code(
            360,
            "ERROR_INVALID_TEMPERATURE",
            message="Invalid temperature specification",
        )
        spec.exit_code(
            361,
            "ERROR_THERMAL_STATE_FAILED",
            message="Thermal state preparation failed",
        )
        spec.exit_code(
            362,
            "ERROR_KUBO_CALCULATION_FAILED",
            message="Kubo calculation failed",
        )
        spec.exit_code(
            363,
            "ERROR_INVALID_CONDUCTIVITY",
            message="Invalid conductivity result",
        )

        # Outline
        spec.outline(
            cls.setup,
            cls.prepare_thermal_state,
            cls.run_kubo_calculation,
            cls.extract_conductivity,
        )

    def setup(self):
        """Initialize the WorkChain and validate inputs."""
        self.report("Starting Kubo transport calculation")

        # Validate temperature/beta inputs
        has_temp = "temperature" in self.inputs
        has_beta = "beta" in self.inputs

        if has_temp and has_beta:
            self.report("ERROR: Both temperature and beta provided")
            return self.exit_codes.ERROR_INVALID_TEMPERATURE
        elif not has_temp and not has_beta:
            self.report("ERROR: Neither temperature nor beta provided")
            return self.exit_codes.ERROR_INVALID_TEMPERATURE

        # Compute beta if temperature provided
        if has_temp:
            temperature = self.inputs.temperature.value
            if temperature <= 0:
                self.report(f"ERROR: Invalid temperature: {temperature}")
                return self.exit_codes.ERROR_INVALID_TEMPERATURE
            self.ctx.beta = 1.0 / temperature
            self.ctx.temperature = temperature
        else:
            beta = self.inputs.beta.value
            if beta <= 0:
                self.report(f"ERROR: Invalid beta: {beta}")
                return self.exit_codes.ERROR_INVALID_TEMPERATURE
            self.ctx.beta = beta
            self.ctx.temperature = 1.0 / beta
        if "tn_layout" in self.inputs:
            self.ctx.tn_layout = self.inputs.tn_layout

        self.report(f"Target: beta={self.ctx.beta}, temperature={self.ctx.temperature}")

    def prepare_thermal_state(self):
        """Prepare thermal density matrix if not provided."""
        # Check if thermal state is already provided
        if "initial_mps" in self.inputs:
            self.report("Using provided thermal state")
            self.ctx.thermal_mpdm = self.inputs.initial_mps
            return

        # Otherwise, prepare thermal state via ThermalPropCalcJob
        self.report("Preparing thermal state via ThermalPropCalcJob")

        inputs = {
            "model": self.inputs.model,
            "temperature": orm.Float(self.ctx.temperature),
            "code": self.inputs.code,
        }

        # Add MPO if provided
        if "mpo" in self.inputs:
            inputs["mpo"] = self.inputs.mpo
        if hasattr(self.ctx, "tn_layout"):
            inputs["tn_layout"] = self.ctx.tn_layout

        # Add config if provided
        if "config" in self.inputs:
            inputs["config"] = self.inputs.config

        # Submit thermal state preparation
        future = self.submit(ThermalPropCalcJob, **inputs)

        return ToContext(thermal_state_calc=future)

    def run_kubo_calculation(self):
        """Run KuboCalcJob for current-current correlation."""
        self.report("Running Kubo calculation for current-current correlation")

        # Get thermal state
        if "initial_mps" in self.inputs:
            thermal_mpdm = self.ctx.thermal_mpdm
        else:
            thermal_calc = self.ctx.thermal_state_calc
            if not thermal_calc.is_finished_ok:
                self.report(f"Thermal state calculation failed: exit_status={thermal_calc.exit_status}")
                return self.exit_codes.ERROR_THERMAL_STATE_FAILED
            thermal_mpdm = thermal_calc.outputs.output_mps
            if "output_tn_layout" in thermal_calc.outputs:
                self.ctx.tn_layout = thermal_calc.outputs.output_tn_layout

        # Build inputs
        inputs = {
            "model": self.inputs.model,
            "temperature": orm.Float(self.ctx.temperature),
            "insteps": self.inputs.insteps,
            "code": self.inputs.code,
        }

        # Add thermal state
        inputs["initial_mps"] = thermal_mpdm

        # Add MPO if provided
        if "mpo" in self.inputs:
            inputs["mpo"] = self.inputs.mpo

        # Add distance matrix if provided
        if "distance_matrix" in self.inputs:
            inputs["distance_matrix"] = self.inputs.distance_matrix

        # Add config if provided
        if "config" in self.inputs:
            inputs["config"] = self.inputs.config
        if hasattr(self.ctx, "tn_layout"):
            inputs["tn_layout"] = self.ctx.tn_layout

        # Submit Kubo calculation
        future = self.submit(KuboCalcJob, **inputs)

        return ToContext(kubo_calc=future)

    def extract_conductivity(self):
        """Extract conductivity and mobility from Kubo results."""
        calc = self.ctx.kubo_calc

        if not calc.is_finished_ok:
            self.report(f"Kubo calculation failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_KUBO_CALCULATION_FAILED

        # Output conductivity
        if "conductivity" in calc.outputs:
            self.out("conductivity", calc.outputs.conductivity)
        elif "output_parameters" in calc.outputs:
            # Extract from output parameters
            params = calc.outputs.output_parameters.get_dict()

            if "conductivity" in params:
                import numpy as np
                from aiida.orm import ArrayData

                conductivity_data = ArrayData()
                conductivity_data.set_array("conductivity", np.array(params["conductivity"]))
                self.out("conductivity", conductivity_data)

        # Output autocorrelation
        if "autocorrelation" in calc.outputs:
            self.out("autocorrelation", calc.outputs.autocorrelation)
        if "output_tn_layout" in calc.outputs:
            self.out("output_tn_layout", calc.outputs.output_tn_layout)
        elif hasattr(self.ctx, "tn_layout"):
            self.out("output_tn_layout", self.ctx.tn_layout)

        # Output mobility (optional)
        if "mobility" in calc.outputs:
            self.out("mobility", calc.outputs.mobility)

        # Output statistics
        stats = {
            "beta": self.ctx.beta,
            "temperature": self.ctx.temperature,
            "nsteps": self.inputs.nsteps.value,
            "dt": self.inputs.dt.value,
        }

        if "output_parameters" in calc.outputs:
            params = calc.outputs.output_parameters.get_dict()
            stats.update(params)

        self.out("output_parameters", orm.Dict(stats))

        self.report("Kubo transport calculation completed")
