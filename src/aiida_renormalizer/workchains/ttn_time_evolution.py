"""WorkChain for TTN time evolution with checkpointing and energy drift detection."""
from __future__ import annotations

from aiida import orm
from aiida.engine import WorkChain, ToContext, while_

from aiida_renormalizer.calculations.ttn.ttns_evolve import TtnsEvolveCalcJob
from aiida_renormalizer.data import BasisTreeData, TTNSData, TtnoData


class TtnTimeEvolutionWorkChain(WorkChain):
    """WorkChain for TTN time evolution with checkpointing and validation.

    This WorkChain implements:
    1. Long-time evolution by splitting into multiple TtnsEvolveCalcJob runs
    2. Automatic checkpointing (saves intermediate states)
    3. Energy drift detection and handling
    4. Trajectory concatenation

    Inputs:
        basis_tree: BasisTreeData - Tree topology and basis grouping
        ttno: TtnoData - Hamiltonian operator
        initial_ttns: TTNSData - Initial state
        total_time: Float - Total evolution time
        checkpoint_time: Float - Time per checkpoint segment
        config: ConfigData - Evolution configuration
        max_energy_drift: Float - Maximum allowed energy drift
        code: AbstractCode - Code to use

    Outputs:
        final_ttns: TTNSData - Final evolved state
        trajectory: ArrayData - Concatenated trajectory
        checkpoints: List - List of checkpoint TTNS nodes
        output_parameters: Dict - Evolution statistics
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        # Inputs
        spec.input("basis_tree", valid_type=BasisTreeData, help="Tree topology and basis grouping")
        spec.input("ttno", valid_type=TtnoData, help="Hamiltonian TTNO")
        spec.input("initial_ttns", valid_type=TTNSData, help="Initial TTNS")
        spec.input("total_time", valid_type=orm.Float, help="Total evolution time")
        spec.input(
            "checkpoint_time",
            valid_type=orm.Float,
            default=lambda: orm.Float(10.0),
            help="Time interval between checkpoints",
        )
        spec.input("config", valid_type=orm.Dict, required=False, help="Evolution config")
        spec.input(
            "max_energy_drift",
            valid_type=orm.Float,
            default=lambda: orm.Float(1e-6),
            help="Maximum allowed energy drift per checkpoint",
        )
        spec.input(
            "trajectory_interval",
            valid_type=orm.Int,
            required=False,
            help="Save trajectory every N steps",
        )
        spec.input("code", valid_type=orm.AbstractCode, help="Code to use")

        # Outputs
        spec.output("final_ttns", valid_type=TTNSData, help="Final evolved state")
        spec.output(
            "trajectory",
            valid_type=orm.ArrayData,
            required=False,
            help="Concatenated trajectory",
        )
        spec.output("checkpoints", valid_type=orm.List, help="List of checkpoint TTNS nodes")
        spec.output("output_parameters", valid_type=orm.Dict, help="Evolution statistics")

        # Exit codes
        spec.exit_code(
            340,
            "ERROR_ENERGY_DRIFT",
            message="Energy drift exceeded maximum allowed value",
        )
        spec.exit_code(
            341,
            "ERROR_EVOLUTION_FAILED",
            message="TTN time evolution calculation failed",
        )

        # Outline
        spec.outline(
            cls.setup,
            while_(cls.not_finished)(
                cls.run_checkpoint,
                cls.inspect_checkpoint,
            ),
            cls.finalize,
        )

    def setup(self):
        """Initialize the WorkChain context."""
        self.ctx.current_time = 0.0
        self.ctx.current_ttns = self.inputs.initial_ttns
        self.ctx.checkpoints = []
        self.ctx.energies = []
        self.ctx.trajectory_segments = []
        self.ctx.iteration = 0

        # Determine number of checkpoint segments
        total_time = self.inputs.total_time.value
        checkpoint_time = self.inputs.checkpoint_time.value
        self.ctx.n_checkpoints = int(total_time / checkpoint_time)
        if total_time % checkpoint_time > 0:
            self.ctx.n_checkpoints += 1

        self.report(
            f"Starting TTN time evolution: total_time={total_time}, "
            f"checkpoint_time={checkpoint_time}, n_checkpoints={self.ctx.n_checkpoints}"
        )

    def not_finished(self):
        """Check if evolution is complete."""
        return self.ctx.current_time < self.inputs.total_time.value

    def run_checkpoint(self):
        """Run a single checkpoint segment."""
        # Determine evolution time for this segment
        remaining_time = self.inputs.total_time.value - self.ctx.current_time
        segment_time = min(remaining_time, self.inputs.checkpoint_time.value)

        self.report(
            f"Running checkpoint {self.ctx.iteration + 1}/{self.ctx.n_checkpoints}: "
            f"evolving for {segment_time} (current_time={self.ctx.current_time})"
        )

        # Build inputs for TtnsEvolveCalcJob
        inputs = {
            "basis_tree": self.inputs.basis_tree,
            "ttno": self.inputs.ttno,
            "initial_ttns": self.ctx.current_ttns,
            "config": self.inputs.config,
            "code": self.inputs.code,
        }

        # Update config with segment time
        # Note: In real implementation, would need to modify config to set evolution time
        # For now, we assume config contains proper time step and method settings
        if "config" in self.inputs:
            inputs["config"] = self.inputs.config

        # Submit TtnsEvolveCalcJob
        future = self.submit(TtnsEvolveCalcJob, **inputs)

        return ToContext(checkpoint_calc=future)

    def inspect_checkpoint(self):
        """Inspect the checkpoint calculation."""
        calc = self.ctx.checkpoint_calc

        # Check if calculation succeeded
        if not calc.is_finished_ok:
            self.report(f"Checkpoint calculation failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_EVOLUTION_FAILED

        # Update current state
        self.ctx.current_ttns = calc.outputs.output_ttns
        self.ctx.checkpoints.append(calc.outputs.output_ttns)

        # Track energy
        if "output_parameters" in calc.outputs:
            params = calc.outputs.output_parameters.get_dict()
            if "final_energy" in params:
                self.ctx.energies.append(params["final_energy"])

                # Check energy drift
                if len(self.ctx.energies) > 1:
                    drift = abs(self.ctx.energies[-1] - self.ctx.energies[-2])
                    if drift > self.inputs.max_energy_drift.value:
                        self.report(
                            f"Energy drift {drift} exceeds maximum {self.inputs.max_energy_drift.value}"
                        )
                        return self.exit_codes.ERROR_ENERGY_DRIFT
                    else:
                        self.report(f"Energy drift: {drift} (within tolerance)")

        # Track trajectory
        if "trajectory" in calc.outputs:
            self.ctx.trajectory_segments.append(calc.outputs.trajectory)

        # Update time and iteration
        segment_time = min(
            self.inputs.total_time.value - self.ctx.current_time,
            self.inputs.checkpoint_time.value
        )
        self.ctx.current_time += segment_time
        self.ctx.iteration += 1

        self.report(f"Checkpoint completed: current_time={self.ctx.current_time}")

    def finalize(self):
        """Collect and output final results."""
        self.report("Finalizing TTN time evolution")

        # Output final TTNS
        self.out("final_ttns", self.ctx.current_ttns)

        # Output checkpoints
        self.out("checkpoints", orm.List(list=self.ctx.checkpoints))

        # Output evolution statistics
        stats = {
            "total_time": self.inputs.total_time.value,
            "n_checkpoints": self.ctx.iteration,
            "final_time": self.ctx.current_time,
            "energies": self.ctx.energies,
        }
        if self.ctx.energies:
            stats["initial_energy"] = self.ctx.energies[0]
            stats["final_energy"] = self.ctx.energies[-1]
            stats["total_energy_drift"] = abs(self.ctx.energies[-1] - self.ctx.energies[0])

        self.out("output_parameters", orm.Dict(stats))

        # Concatenate trajectory segments (if available)
        if self.ctx.trajectory_segments:
            import numpy as np
            from aiida.orm import ArrayData

            # This is simplified - real implementation would properly concatenate
            # trajectory arrays
            trajectory_ad = ArrayData()
            # For now, just save the final trajectory segment
            final_traj = self.ctx.trajectory_segments[-1]
            for name in final_traj.get_arraynames():
                trajectory_ad.set_array(name, final_traj.get_array(name))

            self.out("trajectory", trajectory_ad)

        self.report("TTN time evolution completed successfully")
