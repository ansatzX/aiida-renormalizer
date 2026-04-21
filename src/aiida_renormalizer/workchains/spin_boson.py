"""WorkChain for spin-boson dynamics."""
from __future__ import annotations

import numpy as np

from aiida import orm
from aiida.engine import WorkChain, ToContext, if_, while_

from aiida_renormalizer.calculations.composite.tdvp import TDVPCalcJob
from aiida_renormalizer.calculations.composite.dmrg import DMRGCalcJob
from aiida_renormalizer.data import ModelData, MPSData, MPOData


class SpinBosonWorkChain(WorkChain):
    """WorkChain for spin-boson model dynamics.

    This WorkChain orchestrates spin-boson dynamics simulations:
    1. Initial state preparation (thermal or ground state)
    2. Non-equilibrium dynamics propagation
    3. Observables tracking (population, coherence, energy)
    4. Renormalization effects analysis

    The spin-boson model describes a two-level system (spin) coupled to
    a bosonic bath (phonons), exhibiting rich quantum dynamics including:
    - Coherent oscillations
    - Quantum decoherence
    - Dissipative dynamics
    - Localization transitions

    Inputs:
        model: ModelData - Spin-boson model definition
        mpo: MPOData (optional) - Hamiltonian operator
        initial_state: MPSData (optional) - Initial state
        initial_spin: Str - Initial spin state: "up", "down", or "superposition"
        temperature: Float - Temperature for thermal initial state
        total_time: Float - Total evolution time
        dt: Float - Time step
        renormalize: Bool - Apply polaron renormalization
        tracking_observables: List - Observables to track during dynamics
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
        spec.input("model", valid_type=ModelData, help="Spin-boson model definition")
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
            "initial_spin",
            valid_type=orm.Str,
            default=lambda: orm.Str("up"),
            help="Initial spin state: 'up', 'down', or 'superposition'",
        )
        spec.input(
            "temperature",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.0),
            help="Temperature for thermal initial state",
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
            "renormalize",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="Apply polaron renormalization",
        )
        spec.input(
            "tracking_observables",
            valid_type=orm.List,
            default=lambda: orm.List(["population", "coherence", "energy"]),
            help="Observables to track during dynamics",
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
            410,
            "ERROR_INITIAL_STATE_FAILED",
            message="Initial state preparation failed",
        )
        spec.exit_code(
            411,
            "ERROR_DYNAMICS_FAILED",
            message="Dynamics propagation failed",
        )
        spec.exit_code(
            412,
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
        self.report("Starting spin-boson dynamics calculation")

        # Validate initial spin
        if self.inputs.initial_spin.value not in ["up", "down", "superposition"]:
            raise ValueError(f"Invalid initial_spin: {self.inputs.initial_spin.value}")

        # Calculate number of steps
        self.ctx.nsteps = int(self.inputs.total_time.value / self.inputs.dt.value)
        self.report(f"Evolution: {self.ctx.nsteps} steps, dt={self.inputs.dt.value}")

    def needs_initial_state(self):
        """Check if initial state needs to be prepared."""
        return "initial_state" not in self.inputs

    def prepare_initial_state(self):
        """Prepare initial state based on spin configuration."""
        initial_spin = self.inputs.initial_spin.value
        self.report(f"Preparing initial state: {initial_spin}")

        # Build inputs for state preparation
        # This is a simplified version - actual implementation would construct
        # the appropriate initial MPS for the spin-boson model

        inputs = {
            "model": self.inputs.model,
            "code": self.inputs.code,
        }

        if "mpo" in self.inputs:
            inputs["mpo"] = self.inputs.mpo

        # For thermal initial state, use thermal propagation
        if self.inputs.temperature.value > 0:
            from aiida_renormalizer.calculations.composite.thermal_prop import ThermalPropCalcJob

            inputs["temperature"] = self.inputs.temperature
            inputs["nsteps"] = orm.Int(100)

            future = self.submit(ThermalPropCalcJob, **inputs)

        # For ground state, use DMRG
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
        """Run time evolution dynamics."""
        self.report("Running spin-boson dynamics")

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
        self.report("Dynamics calculation completed successfully")

    def extract_observables(self):
        """Extract observables from dynamics trajectory."""
        self.report("Extracting observables from trajectory")

        try:
            params = self.ctx.dynamics_params
            tracking = self.inputs.tracking_observables.get_list()

            # Create trajectory ArrayData
            trajectory = orm.ArrayData()

            # Extract time series
            if "time_series" in params:
                time = np.array(params["time_series"])
                trajectory.set_array("time", time)

            # Extract population (spin-up probability)
            if "population" in tracking and "populations" in params:
                population = np.array(params["populations"])
                trajectory.set_array("population", population)

            # Extract coherence (off-diagonal density matrix element)
            if "coherence" in tracking and "coherence" in params:
                coherence = np.array(params["coherence"])
                trajectory.set_array("coherence", coherence)

            # Extract energy
            if "energy" in tracking and "energies" in params:
                energy = np.array(params["energies"])
                trajectory.set_array("energy", energy)

            # Extract entropy (for entanglement analysis)
            if "entropy" in tracking and "entropy" in params:
                entropy = np.array(params["entropy"])
                trajectory.set_array("entropy", entropy)

            self.ctx.trajectory = trajectory
            self.report("Observables extraction completed")

        except Exception as e:
            self.report(f"Observable extraction failed: {e}")
            return self.exit_codes.ERROR_OBSERVABLE_TRACKING_FAILED

    def finalize(self):
        """Collect and output final results."""
        self.report("Finalizing spin-boson dynamics calculation")

        # Output trajectory
        self.out("trajectory", self.ctx.trajectory)

        # Output final state
        self.out("final_state", self.ctx.final_state)

        # Output parameters
        params = orm.Dict(self.ctx.dynamics_params)
        self.out("output_parameters", params)

        # Calculate key quantities
        if "population" in self.inputs.tracking_observables.get_list():
            if "populations" in self.ctx.dynamics_params:
                population = np.array(self.ctx.dynamics_params["populations"])
                initial_pop = population[0] if len(population) > 0 else 0.0
                final_pop = population[-1] if len(population) > 0 else 0.0

                self.report(
                    f"Spin-boson dynamics completed: "
                    f"initial_pop={initial_pop:.3f}, final_pop={final_pop:.3f}, "
                    f"Δ={final_pop - initial_pop:.3f}"
                )

                # Add to output parameters
                params_dict = params.get_dict()
                params_dict["initial_population"] = float(initial_pop)
                params_dict["final_population"] = float(final_pop)
                params = orm.Dict(params_dict)
        else:
            self.report("Spin-boson dynamics calculation completed")

        self.out("output_parameters", params)
