"""WorkChain for vibronic coupling dynamics."""
from __future__ import annotations

import numpy as np

from aiida import orm
from aiida.engine import WorkChain, ToContext, if_

from aiida_renormalizer.calculations.composite.tdvp import TDVPCalcJob
from aiida_renormalizer.calculations.composite.dmrg import DMRGCalcJob
from aiida_renormalizer.data import ModelData, MPSData, MPOData


class VibronicWorkChain(WorkChain):
    """WorkChain for vibronic coupling dynamics.

    This WorkChain orchestrates vibronic dynamics simulations:
    1. Initial state preparation (electronic + vibrational)
    2. Vibronic dynamics propagation
    3. Observables tracking (population transfer, vibrational modes)
    4. Energy flow analysis

    Vibronic coupling describes the interaction between electronic
    states and nuclear vibrations, leading to:
    - Non-adiabatic transitions (internal conversion, intersystem crossing)
    - Vibrational energy redistribution
    - Condon vs non-Condon effects
    - Exciton-phonon coupling

    Inputs:
        model: ModelData - Vibronic model definition
        mpo: MPOData (optional) - Hamiltonian operator
        initial_state: MPSData (optional) - Initial state
        initial_electronic_state: Int - Initial electronic state index
        initial_vibrational_state: Str - "ground", "thermal", or "excited"
        temperature: Float - Temperature for thermal vibrational state
        total_time: Float - Total evolution time
        dt: Float - Time step
        condon_approximation: Bool - Use Condon approximation
        tracking_electronic: Bool - Track electronic populations
        tracking_vibrational: Bool - Track vibrational occupations
        config: Dict - TDVP configuration
        compress_config: Dict - Compression configuration
        code: AbstractCode - Code to use

    Outputs:
        trajectory: ArrayData - Time evolution of observables
        final_state: MPSData - Final evolved state
        output_parameters: Dict - Dynamics statistics
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        # Inputs
        spec.input("model", valid_type=ModelData, help="Vibronic model definition")
        spec.input(
            "mpo",
            valid_type=MPOData,
            required=False,
            help="Hamiltonian MPO (built if not provided)",
        )
        spec.input(
            "initial_state",
            valid_type=MPSData,
            required=False,
            help="Initial state (will be prepared if not provided)",
        )
        spec.input(
            "initial_electronic_state",
            valid_type=orm.Int,
            default=lambda: orm.Int(0),
            help="Initial electronic state index (0-indexed)",
        )
        spec.input(
            "initial_vibrational_state",
            valid_type=orm.Str,
            default=lambda: orm.Str("ground"),
            help="Initial vibrational state: 'ground', 'thermal', or 'excited'",
        )
        spec.input(
            "temperature",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.0),
            help="Temperature for thermal vibrational state",
        )
        spec.input(
            "total_time",
            valid_type=orm.Float,
            help="Total evolution time",
        )
        spec.input(
            "dt",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.01),
            help="Time step",
        )
        spec.input(
            "condon_approximation",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help="Use Condon approximation (transition dipole independent of nuclear coordinates)",
        )
        spec.input(
            "tracking_electronic",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help="Track electronic populations",
        )
        spec.input(
            "tracking_vibrational",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help="Track vibrational occupations",
        )
        spec.input("config", valid_type=orm.Dict, required=False, help="TDVP configuration")
        spec.input("compress_config", valid_type=orm.Dict, required=False, help="Compression configuration")
        spec.input("code", valid_type=orm.AbstractCode, help="Code to use")

        # Outputs
        spec.output("trajectory", valid_type=orm.ArrayData, help="Time evolution of observables")
        spec.output("final_state", valid_type=MPSData, help="Final evolved state")
        spec.output("output_parameters", valid_type=orm.Dict, help="Dynamics statistics")

        # Exit codes
        spec.exit_code(
            420,
            "ERROR_INITIAL_STATE_FAILED",
            message="Initial state preparation failed",
        )
        spec.exit_code(
            421,
            "ERROR_DYNAMICS_FAILED",
            message="Dynamics propagation failed",
        )
        spec.exit_code(
            422,
            "ERROR_OBSERVABLE_TRACKING_FAILED",
            message="Observable tracking failed",
        )

        # Outline
        spec.outline(
            cls.setup,
            if_(cls.needs_initial_state)(
                cls.prepare_initial_state,
                cls.inspect_initial_state,
            ),
            cls.run_dynamics,
            cls.inspect_dynamics,
            cls.extract_observables,
            cls.finalize,
        )

    def setup(self):
        """Initialize the WorkChain."""
        self.report("Starting vibronic dynamics calculation")

        # Validate vibrational state
        if self.inputs.initial_vibrational_state.value not in ["ground", "thermal", "excited"]:
            raise ValueError(f"Invalid initial_vibrational_state: {self.inputs.initial_vibrational_state.value}")

        # Calculate number of steps
        self.ctx.nsteps = int(self.inputs.total_time.value / self.inputs.dt.value)
        self.report(f"Evolution: {self.ctx.nsteps} steps, dt={self.inputs.dt.value}")

        # Store configuration
        self.ctx.condon = self.inputs.condon_approximation.value
        self.report(f"Condon approximation: {self.ctx.condon}")

    def needs_initial_state(self):
        """Check if initial state needs to be prepared."""
        return "initial_state" not in self.inputs

    def prepare_initial_state(self):
        """Prepare initial electronic + vibrational state."""
        vib_state = self.inputs.initial_vibrational_state.value
        elec_state = self.inputs.initial_electronic_state.value

        self.report(f"Preparing initial state: electronic={elec_state}, vibrational={vib_state}")

        # Build inputs for state preparation
        inputs = {
            "model": self.inputs.model,
            "code": self.inputs.code,
        }

        if "mpo" in self.inputs:
            inputs["mpo"] = self.inputs.mpo

        # For thermal vibrational state, use thermal propagation
        if vib_state == "thermal" and self.inputs.temperature.value > 0:
            from aiida_renormalizer.calculations.composite.thermal_prop import ThermalPropCalcJob

            inputs["temperature"] = self.inputs.temperature
            inputs["nsteps"] = orm.Int(100)

            future = self.submit(ThermalPropCalcJob, **inputs)

        # For ground state (both electronic and vibrational), use DMRG
        else:
            inputs["config"] = self.inputs.config if "config" in self.inputs else orm.Dict({})

            future = self.submit(DMRGCalcJob, **inputs)

        return ToContext(initial_state_calc=future)

    def inspect_initial_state(self):
        """Inspect initial state calculation."""
        calc = self.ctx.initial_state_calc

        if not calc.is_finished_ok:
            self.report(f"Initial state preparation failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_INITIAL_STATE_FAILED

        self.ctx.initial_state = calc.outputs.output_mps
        self.report("Initial state prepared successfully")

    def run_dynamics(self):
        """Run vibronic dynamics propagation."""
        self.report("Running vibronic dynamics")

        # Get initial state
        initial_state = self.inputs.initial_state if "initial_state" in self.inputs else self.ctx.initial_state

        # Build inputs for TDVP
        inputs = {
            "model": self.inputs.model,
            "initial_mps": initial_state,
            "dt": self.inputs.dt,
            "nsteps": orm.Int(self.ctx.nsteps),
            "code": self.inputs.code,
        }

        # Add optional inputs
        if "mpo" in self.inputs:
            inputs["mpo"] = self.inputs.mpo
        if "config" in self.inputs:
            inputs["config"] = self.inputs.config
        if "compress_config" in self.inputs:
            inputs["compress_config"] = self.inputs.compress_config

        future = self.submit(TDVPCalcJob, **inputs)

        return ToContext(dynamics_calc=future)

    def inspect_dynamics(self):
        """Inspect dynamics calculation."""
        calc = self.ctx.dynamics_calc

        if not calc.is_finished_ok:
            self.report(f"Dynamics calculation failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_DYNAMICS_FAILED

        # Store results
        self.ctx.final_state = calc.outputs.output_mps
        self.ctx.dynamics_params = calc.outputs.output_parameters.get_dict()
        self.report("Vibronic dynamics completed successfully")

    def extract_observables(self):
        """Extract observables from dynamics trajectory."""
        self.report("Extracting vibronic observables")

        try:
            params = self.ctx.dynamics_params

            # Create trajectory ArrayData
            trajectory = orm.ArrayData()

            # Extract time series
            if "time_series" in params:
                time = np.array(params["time_series"])
                trajectory.set_array("time", time)

            # Extract electronic populations
            if self.inputs.tracking_electronic.value and "electronic_populations" in params:
                elec_pop = np.array(params["electronic_populations"])
                trajectory.set_array("electronic_populations", elec_pop)

            # Extract vibrational occupations
            if self.inputs.tracking_vibrational.value and "vibrational_occupations" in params:
                vib_occ = np.array(params["vibrational_occupations"])
                trajectory.set_array("vibrational_occupations", vib_occ)

            # Extract energies
            if "energies" in params:
                energy = np.array(params["energies"])
                trajectory.set_array("energy", energy)

            # Extract energy components if available
            if "electronic_energy" in params:
                e_elec = np.array(params["electronic_energy"])
                trajectory.set_array("electronic_energy", e_elec)

            if "vibrational_energy" in params:
                e_vib = np.array(params["vibrational_energy"])
                trajectory.set_array("vibrational_energy", e_vib)

            if "coupling_energy" in params:
                e_coupling = np.array(params["coupling_energy"])
                trajectory.set_array("coupling_energy", e_coupling)

            self.ctx.trajectory = trajectory
            self.report("Observables extraction completed")

        except Exception as e:
            self.report(f"Observable extraction failed: {e}")
            return self.exit_codes.ERROR_OBSERVABLE_TRACKING_FAILED

    def finalize(self):
        """Collect and output final results."""
        self.report("Finalizing vibronic dynamics calculation")

        # Output trajectory
        self.out("trajectory", self.ctx.trajectory)

        # Output final state
        self.out("final_state", self.ctx.final_state)

        # Output parameters
        params = orm.Dict(self.ctx.dynamics_params)
        self.out("output_parameters", params)

        # Calculate population transfer
        if self.inputs.tracking_electronic.value and "electronic_populations" in self.ctx.dynamics_params:
            elec_pop = np.array(self.ctx.dynamics_params["electronic_populations"])

            if len(elec_pop) > 0:
                initial_pop = elec_pop[0]
                final_pop = elec_pop[-1]

                # Find maximum population transfer
                if len(elec_pop.shape) == 2:  # Multiple electronic states
                    # Calculate population transfer for each state
                    pop_transfer = np.abs(final_pop - initial_pop)
                    max_transfer = np.max(pop_transfer)

                    self.report(
                        f"Vibronic dynamics completed: "
                        f"max population transfer = {max_transfer:.3f}, "
                        f"Condon = {self.ctx.condon}"
                    )

                    # Add to output parameters
                    params_dict = params.get_dict()
                    params_dict["max_population_transfer"] = float(max_transfer)
                    params = orm.Dict(params_dict)
                else:
                    self.report(f"Vibronic dynamics completed: Condon = {self.ctx.condon}")
            else:
                self.report(f"Vibronic dynamics completed: Condon = {self.ctx.condon}")
        else:
            self.report(f"Vibronic dynamics completed: Condon = {self.ctx.condon}")

        self.out("output_parameters", params)

        # Energy conservation check
        if "energies" in self.ctx.dynamics_params:
            energy = np.array(self.ctx.dynamics_params["energies"])
            if len(energy) > 1:
                energy_drift = np.abs(energy[-1] - energy[0])
                relative_drift = energy_drift / np.abs(energy[0]) if energy[0] != 0 else energy_drift

                self.report(f"Energy conservation: relative drift = {relative_drift:.6e}")

                params_dict = params.get_dict()
                params_dict["energy_drift"] = float(energy_drift)
                params_dict["relative_energy_drift"] = float(relative_drift)
                params = orm.Dict(params_dict)
                self.out("output_parameters", params)
