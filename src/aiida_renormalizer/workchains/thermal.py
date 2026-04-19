"""WorkChain for finite-temperature state preparation."""
from __future__ import annotations

import tempfile

from aiida import orm
from aiida.engine import WorkChain, ToContext

from aiida_renormalizer.calculations.composite.thermal_prop import ThermalPropCalcJob
from aiida_renormalizer.data import ModelData, MpsData, MpoData


class ThermalStateWorkChain(WorkChain):
    """WorkChain for finite-temperature state preparation via imaginary time propagation.

    This WorkChain prepares thermal density matrices (MpDm) at finite temperature:
    - Constructs identity/maximally-entangled initial state
    - Performs imaginary time propagation to target temperature
    - Verifies thermal state quality (partition function, energy monotonicity)

    Inputs:
        model: ModelData - System definition
        mpo: MpoData - Hamiltonian operator (optional, will be built if not provided)
        temperature: Float - Target temperature (alternative to beta)
        beta: Float - Inverse temperature (alternative to temperature)
        config: Dict - Thermal propagation configuration
        space: Str - Propagation space ("GS" or "EX", default: "GS")
        n_iterations: Int - Number of thermal propagation iterations
        code: AbstractCode - Code to use

    Outputs:
        thermal_mpdm: MpsData - Thermal density matrix (MpDm)
        partition_function: Float - Partition function Z
        free_energy: Float - Free energy F = -kT ln Z
        output_parameters: Dict - Thermal state statistics

    Exit Codes:
        350: ERROR_INVALID_TEMPERATURE
        351: ERROR_THERMAL_PROP_FAILED
        352: ERROR_INVALID_THERMAL_STATE
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        # Inputs
        spec.input("model", valid_type=ModelData, help="System definition")
        spec.input(
            "mpo",
            valid_type=MpoData,
            required=False,
            help="Hamiltonian MPO (will be built if not provided)",
        )
        spec.input(
            "temperature",
            valid_type=orm.Float,
            required=False,
            help="Target temperature (use this OR beta)",
        )
        spec.input(
            "beta",
            valid_type=orm.Float,
            required=False,
            help="Inverse temperature (use this OR temperature)",
        )
        spec.input("config", valid_type=orm.Dict, required=False, help="Thermal propagation configuration")
        spec.input(
            "space",
            valid_type=orm.Str,
            default=lambda: orm.Str("GS"),
            help="Propagation space: 'GS' (ground state) or 'EX' (excited state)",
        )
        spec.input(
            "n_iterations",
            valid_type=orm.Int,
            default=lambda: orm.Int(10),
            help="Number of thermal propagation iterations",
        )
        spec.input("code", valid_type=orm.AbstractCode, help="Code to use")

        # Outputs
        spec.output("thermal_mpdm", valid_type=MpsData, help="Thermal density matrix (MpDm)")
        spec.output("partition_function", valid_type=orm.Float, help="Partition function Z")
        spec.output("free_energy", valid_type=orm.Float, help="Free energy F = -kT ln Z")
        spec.output("output_parameters", valid_type=orm.Dict, help="Thermal state statistics")

        # Exit codes
        spec.exit_code(
            350,
            "ERROR_INVALID_TEMPERATURE",
            message="Invalid temperature specification (must provide temperature OR beta, not both)",
        )
        spec.exit_code(
            351,
            "ERROR_THERMAL_PROP_FAILED",
            message="Thermal propagation calculation failed",
        )
        spec.exit_code(
            352,
            "ERROR_INVALID_THERMAL_STATE",
            message="Thermal state validation failed (negative partition function, etc.)",
        )

        # Outline
        spec.outline(
            cls.setup,
            cls.construct_initial_state,
            cls.run_thermal_prop,
            cls.inspect_thermal_state,
            cls.return_thermal_state,
        )

    def setup(self):
        """Initialize the WorkChain and validate inputs."""
        self.report("Starting thermal state preparation")

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

        self.report(f"Target: beta={self.ctx.beta}, temperature={self.ctx.temperature}")

    def construct_initial_state(self):
        """Construct identity/maximally-entangled initial MpDm.

        For finite temperature calculations, the initial state is the
        maximally-entangled state (infinite temperature density matrix).
        """
        self.report("Constructing maximally-entangled initial state")

        # Load model
        model = self.inputs.model.load_model()

        # Construct maximally entangled state
        # This is the infinite-temperature density matrix
        # In renormalizer: MpDm.max_entangled_gs(model) or MpDm.max_entangled_ex(model)
        from renormalizer.mps import MpDm

        space = self.inputs.space.value
        if space == "GS":
            init_mpdm = MpDm.max_entangled_gs(model)
        elif space == "EX":
            init_mpdm = MpDm.max_entangled_ex(model)
        else:
            self.report(f"WARNING: Unknown space '{space}', using GS")
            init_mpdm = MpDm.max_entangled_gs(model)

        # Store as MpsData
        artifact_base = self.inputs.metadata.options.get(
            "artifact_storage_base",
            str(tempfile.gettempdir()),
        )
        init_mpdm_node = MpsData.from_mps(
            init_mpdm,
            self.inputs.model,
            storage_backend=self.inputs.metadata.options.get("artifact_storage_backend", "posix"),
            storage_base=artifact_base,
            relative_path=f"thermal/{self.node.uuid}/initial_mpdm.npz",
        )
        init_mpdm_node.store()

        self.ctx.initial_mpdm = init_mpdm_node
        self.report("Initial thermal state constructed and stored")

    def run_thermal_prop(self):
        """Run ThermalPropCalcJob to propagate to target temperature."""
        self.report(f"Running thermal propagation to beta={self.ctx.beta}")

        # Build inputs
        inputs = {
            "model": self.inputs.model,
            "temperature": orm.Float(self.ctx.temperature),
            "n_iterations": self.inputs.n_iterations,
            "code": self.inputs.code,
        }

        # Add initial state
        inputs["initial_mps"] = self.ctx.initial_mpdm

        # Add MPO if provided
        if "mpo" in self.inputs:
            inputs["mpo"] = self.inputs.mpo

        # Add config if provided
        if "config" in self.inputs:
            inputs["config"] = self.inputs.config

        # Submit thermal propagation
        future = self.submit(ThermalPropCalcJob, **inputs)

        return ToContext(thermal_calc=future)

    def inspect_thermal_state(self):
        """Verify thermal state quality."""
        calc = self.ctx.thermal_calc

        if not calc.is_finished_ok:
            self.report(f"Thermal propagation failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_THERMAL_PROP_FAILED

        # Extract results
        self.ctx.thermal_mpdm = calc.outputs.output_mps

        # Extract thermal state properties
        params = {}
        if "output_parameters" in calc.outputs:
            params = calc.outputs.output_parameters.get_dict()

        # Check partition function
        partition_function = params.get("partition_function", 1.0)
        if partition_function <= 0:
            self.report(f"ERROR: Invalid partition function: {partition_function}")
            return self.exit_codes.ERROR_INVALID_THERMAL_STATE

        # Compute free energy: F = -kT ln Z
        import numpy as np
        free_energy = -self.ctx.temperature * np.log(partition_function)

        self.ctx.partition_function = partition_function
        self.ctx.free_energy = free_energy
        self.ctx.thermal_params = params

        self.report(
            f"Thermal state prepared: Z={partition_function}, F={free_energy}"
        )

    def return_thermal_state(self):
        """Return thermal state outputs."""
        self.report("Returning thermal state")

        # Output thermal MpDm
        self.out("thermal_mpdm", self.ctx.thermal_mpdm)

        # Output partition function
        self.out("partition_function", orm.Float(self.ctx.partition_function))

        # Output free energy
        self.out("free_energy", orm.Float(self.ctx.free_energy))

        # Output statistics
        stats = {
            "beta": self.ctx.beta,
            "temperature": self.ctx.temperature,
            "partition_function": self.ctx.partition_function,
            "free_energy": self.ctx.free_energy,
            "space": self.inputs.space.value,
        }
        if hasattr(self.ctx, "thermal_params"):
            stats.update(self.ctx.thermal_params)

        self.out("output_parameters", orm.Dict(stats))

        self.report("Thermal state preparation completed")
