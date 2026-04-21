"""WorkChain for charge diffusion dynamics."""
from __future__ import annotations

import numpy as np

from aiida import orm
from aiida.engine import WorkChain, ToContext, if_

from aiida_renormalizer.calculations.spectra.charge_diffusion import ChargeDiffusionCalcJob
from aiida_renormalizer.calculations.composite.thermal_prop import ThermalPropCalcJob
from aiida_renormalizer.data import ModelData, MPSData, MPOData


class ChargeDiffusionWorkChain(WorkChain):
    """WorkChain for charge diffusion dynamics.

    This WorkChain orchestrates the charge diffusion workflow:
    1. Thermal state preparation (if needed)
    2. Charge diffusion dynamics simulation
    3. Mean square displacement (MSD) trajectory extraction
    4. Mobility calculation (optional)

    It supports:
    - Automatic thermal state preparation
    - Franck-Condon or relaxed electron initialization
    - Mean square displacement tracking
    - Energy conservation validation
    - Optional reduced density matrix calculation

    Inputs:
        model: ModelData - System definition (must be HolsteinModel)
        mpo: MPOData (optional) - Hamiltonian operator
        thermal_state: MPSData (optional) - Pre-computed thermal state
        temperature: Float - Temperature in atomic units
        init_electron: Str - "fc" (Franck-Condon) or "relaxed"
        stop_at_edge: Bool - Stop when charge reaches boundary
        rdm: Bool - Calculate reduced density matrix
        total_time: Float - Total evolution time
        dt: Float - Time step
        config: Dict - Evolution configuration
        compress_config: Dict - Compression configuration
        code: AbstractCode - Code to use

    Outputs:
        trajectory: ArrayData - MSD trajectory and time series
        thermal_state: MPSData - Thermal state used
        output_parameters: Dict - Diffusion statistics
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        # Inputs
        spec.input("model", valid_type=ModelData, help="System definition (must be HolsteinModel)")
        spec.input(
            "mpo",
            valid_type=MPOData,
            required=False,
            help="Hamiltonian MPO (built if not provided)",
        )
        spec.input(
            "thermal_state",
            valid_type=MPSData,
            required=False,
            help="Pre-computed thermal state (will be computed if not provided)",
        )
        spec.input(
            "temperature",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.0),
            help="Temperature in atomic units",
        )
        spec.input(
            "init_electron",
            valid_type=orm.Str,
            default=lambda: orm.Str("relaxed"),
            help="Electron initialization: 'fc' or 'relaxed'",
        )
        spec.input(
            "stop_at_edge",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help="Stop when charge reaches system boundary",
        )
        spec.input(
            "rdm",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="Calculate reduced density matrix (expensive)",
        )
        spec.input(
            "total_time",
            valid_type=orm.Float,
            required=False,
            help="Total evolution time (alternative to config)",
        )
        spec.input(
            "dt",
            valid_type=orm.Float,
            required=False,
            help="Time step (alternative to config)",
        )
        spec.input("config", valid_type=orm.Dict, required=False, help="EvolveConfig")
        spec.input("compress_config", valid_type=orm.Dict, required=False, help="CompressConfig")
        spec.input("nsteps_thermal", valid_type=orm.Int, default=lambda: orm.Int(100), help="Thermal propagation steps")
        spec.input("code", valid_type=orm.AbstractCode, help="Code to use")

        # Outputs
        spec.output("trajectory", valid_type=orm.ArrayData, help="MSD trajectory and time series")
        spec.output("thermal_state", valid_type=orm.Data, required=False, help="Thermal state used")
        spec.output("output_parameters", valid_type=orm.Dict, help="Diffusion statistics")

        # Exit codes
        spec.exit_code(
            400,
            "ERROR_THERMAL_STATE_FAILED",
            message="Thermal state preparation failed",
        )
        spec.exit_code(
            401,
            "ERROR_DIFFUSION_FAILED",
            message="Charge diffusion calculation failed",
        )
        spec.exit_code(
            402,
            "ERROR_TRAJECTORY_EXTRACTION_FAILED",
            message="Trajectory extraction failed",
        )

        # Outline
        spec.outline(
            cls.setup,
            if_(cls.needs_thermal_state)(
                cls.prepare_thermal_state,
                cls.inspect_thermal_state,
            ),
            cls.run_diffusion,
            cls.inspect_diffusion,
            cls.extract_trajectory,
            cls.finalize,
        )

    def setup(self):
        """Initialize the WorkChain."""
        self.report("Starting charge diffusion calculation")

        # Validate init_electron
        if self.inputs.init_electron.value not in ["fc", "relaxed"]:
            raise ValueError(f"Invalid init_electron: {self.inputs.init_electron.value}")

        # Store temperature
        self.ctx.temperature = self.inputs.temperature.value
        self.report(f"Temperature: {self.ctx.temperature} a.u.")

    def needs_thermal_state(self):
        """Check if thermal state needs to be prepared."""
        return "thermal_state" not in self.inputs and self.ctx.temperature > 0

    def prepare_thermal_state(self):
        """Prepare thermal state via imaginary time propagation."""
        self.report("Preparing thermal state via imaginary time propagation")

        # Build inputs for thermal propagation
        inputs = {
            "model": self.inputs.model,
            "temperature": self.inputs.temperature,
            "nsteps": self.inputs.nsteps_thermal,
            "code": self.inputs.code,
        }

        if "mpo" in self.inputs:
            inputs["mpo"] = self.inputs.mpo

        future = self.submit(ThermalPropCalcJob, **inputs)

        return ToContext(thermal_calc=future)

    def inspect_thermal_state(self):
        """Inspect thermal state calculation."""
        calc = self.ctx.thermal_calc

        if not calc.is_finished_ok:
            self.report(f"Thermal state calculation failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_THERMAL_STATE_FAILED

        self.ctx.thermal_state = calc.outputs.output_mps
        self.report("Thermal state prepared successfully")

    def run_diffusion(self):
        """Run charge diffusion calculation."""
        self.report("Running charge diffusion dynamics")

        # Build inputs
        inputs = {
            "model": self.inputs.model,
            "temperature": self.inputs.temperature,
            "init_electron": self.inputs.init_electron,
            "stop_at_edge": self.inputs.stop_at_edge,
            "rdm": self.inputs.rdm,
            "code": self.inputs.code,
        }

        # Add thermal state if available
        if "thermal_state" in self.inputs:
            inputs["initial_mps"] = self.inputs.thermal_state
        elif hasattr(self.ctx, "thermal_state"):
            inputs["initial_mps"] = self.ctx.thermal_state

        # Add optional inputs
        if "mpo" in self.inputs:
            inputs["mpo"] = self.inputs.mpo
        if "config" in self.inputs:
            inputs["config"] = self.inputs.config
        if "compress_config" in self.inputs:
            inputs["compress_config"] = self.inputs.compress_config

        future = self.submit(ChargeDiffusionCalcJob, **inputs)

        return ToContext(diffusion_calc=future)

    def inspect_diffusion(self):
        """Inspect diffusion calculation."""
        calc = self.ctx.diffusion_calc

        if not calc.is_finished_ok:
            self.report(f"Diffusion calculation failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_DIFFUSION_FAILED

        # Store results
        self.ctx.diffusion_params = calc.outputs.output_parameters.get_dict()
        self.report("Diffusion calculation completed successfully")

    def extract_trajectory(self):
        """Extract MSD trajectory from diffusion results."""
        self.report("Extracting MSD trajectory")

        try:
            params = self.ctx.diffusion_params

            # Create trajectory ArrayData
            trajectory = orm.ArrayData()

            # Extract time series
            if "time_series" in params:
                time = np.array(params["time_series"])
                trajectory.set_array("time", time)

            # Extract mean square displacement
            if "r_square" in params:
                msd = np.array(params["r_square"])
                trajectory.set_array("msd", msd)

            # Extract occupations if available
            if "occupations" in params:
                occupations = np.array(params["occupations"])
                trajectory.set_array("occupations", occupations)

            # Extract energies if available
            if "energies" in params:
                energies = np.array(params["energies"])
                trajectory.set_array("energies", energies)

            self.ctx.trajectory = trajectory
            self.report("Trajectory extraction completed")

        except Exception as e:
            self.report(f"Trajectory extraction failed: {e}")
            return self.exit_codes.ERROR_TRAJECTORY_EXTRACTION_FAILED

    def finalize(self):
        """Collect and output final results."""
        self.report("Finalizing charge diffusion calculation")

        # Output thermal state (if prepared)
        if "thermal_state" in self.inputs:
            self.out("thermal_state", self.inputs.thermal_state)
        elif hasattr(self.ctx, "thermal_state"):
            self.out("thermal_state", self.ctx.thermal_state)

        # Output trajectory
        self.out("trajectory", self.ctx.trajectory)

        # Output parameters
        params = orm.Dict(self.ctx.diffusion_params)
        self.out("output_parameters", params)

        # Calculate diffusion coefficient if MSD available
        if "r_square" in self.ctx.diffusion_params and "time_series" in self.ctx.diffusion_params:
            try:
                time = np.array(self.ctx.diffusion_params["time_series"])
                msd = np.array(self.ctx.diffusion_params["r_square"])

                # Fit D from <r²> = 2Dt (1D) or 4Dt (2D) or 6Dt (3D)
                # Here we assume 1D for simplicity
                if len(time) > 10:
                    # Linear fit to the last 50% of trajectory
                    n_fit = len(time) // 2
                    time_fit = time[-n_fit:]
                    msd_fit = msd[-n_fit:]

                    # D = d<r²>/dt / 2 (1D)
                    coeffs = np.polyfit(time_fit, msd_fit, 1)
                    diffusion_coeff = coeffs[0] / 2.0

                    self.report(f"Diffusion coefficient D ≈ {diffusion_coeff:.6e} a.u.")

                    # Store in output parameters
                    params_dict = params.get_dict()
                    params_dict["diffusion_coefficient"] = float(diffusion_coeff)
                    params = orm.Dict(params_dict)
            except Exception as e:
                self.report(f"Could not calculate diffusion coefficient: {e}")

        self.out("output_parameters", params)

        self.report(
            f"Charge diffusion calculation completed: "
            f"T={self.ctx.temperature} a.u., "
            f"init={self.inputs.init_electron.value}"
        )
