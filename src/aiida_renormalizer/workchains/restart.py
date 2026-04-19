"""Base WorkChain with automatic restart capability."""
from __future__ import annotations

from aiida import orm
from aiida.engine import BaseRestartWorkChain, process_handler, ProcessHandlerReport, while_


class RenoRestartWorkChain(BaseRestartWorkChain):
    """Base WorkChain for Reno calculations with automatic restart.

    This class provides automatic restart capabilities for Reno CalcJobs,
    handling common failure modes like:
    - Convergence failures
    - Memory limits
    - Time limits
    - Physical validation failures

    Subclasses should:
    1. Set _process_class to the specific CalcJob
    2. Override setup() to initialize self.ctx.inputs
    3. Add custom process_handlers for specific failure modes
    4. Override results() if needed to customize output collection
    """

    # Subclasses should override this
    _process_class = None

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        # Common inputs for all Reno WorkChains
        spec.input(
            "max_iterations",
            valid_type=orm.Int,
            default=lambda: orm.Int(5),
            help="Maximum number of restart iterations",
        )

        # Outline follows the standard restart pattern
        spec.outline(
            cls.setup,
            while_(cls.should_run_process)(
                cls.run_process,
                cls.inspect_process,
            ),
            cls.results,
        )

        # Exit codes
        spec.exit_code(
            900,
            "ERROR_MAXIMUM_ITERATIONS_EXCEEDED",
            message="Maximum iterations exceeded without convergence",
        )
        spec.exit_code(
            901,
            "ERROR_PROCESS_FAILURE",
            message="Subprocess failed with unhandled exit code",
        )

    def setup(self):
        """Initialize the WorkChain context."""
        super().setup()
        # Subclasses should override this and set self.ctx.inputs
        self.ctx.inputs = {}

    @process_handler()
    def handle_not_converged(self, node):
        """Handle non-convergence by increasing bond dimension.

        This is a common strategy for DMRG and time evolution calculations.
        """
        # Check if this is a convergence error (exit code 300)
        if node.exit_status != 300:
            return None

        # Increase bond dimension if available
        if "config" in self.ctx.inputs:
            config_dict = self.ctx.inputs["config"].get_dict()
            if "M_max" in config_dict:
                old_m = config_dict["M_max"]
                new_m = int(old_m * 1.5)
                config_dict["M_max"] = new_m
                self.ctx.inputs["config"] = orm.Dict(config_dict)
                self.report(f"Increased bond dimension from {old_m} to {new_m}")
                return ProcessHandlerReport()

        # If we can't increase bond dimension, give up
        self.report("Cannot handle convergence failure - no bond dimension to increase")
        return ProcessHandlerReport(
            exit_code=self.exit_codes.ERROR_MAXIMUM_ITERATIONS_EXCEEDED
        )

    @process_handler()
    def handle_physical_validation(self, node):
        """Handle physical validation failures.

        Common causes:
        - Energy drift in time evolution
        - Norm violation
        - Numerical instability
        """
        # Check if this is a physical validation error (exit code 310)
        if node.exit_status != 310:
            return None

        # Decrease time step if applicable
        if "dt" in self.ctx.inputs:
            old_dt = self.ctx.inputs["dt"].value
            new_dt = old_dt / 2.0
            self.ctx.inputs["dt"] = orm.Float(new_dt)
            self.report(f"Decreased time step from {old_dt} to {new_dt}")
            return ProcessHandlerReport()

        # Try decreasing time step in config
        if "config" in self.ctx.inputs:
            config_dict = self.ctx.inputs["config"].get_dict()
            if "dt" in config_dict:
                old_dt = config_dict["dt"]
                new_dt = old_dt / 2.0
                config_dict["dt"] = new_dt
                self.ctx.inputs["config"] = orm.Dict(config_dict)
                self.report(f"Decreased config time step from {old_dt} to {new_dt}")
                return ProcessHandlerReport()

        self.report("Cannot handle physical validation failure")
        return ProcessHandlerReport(
            exit_code=self.exit_codes.ERROR_PROCESS_FAILURE
        )

    def results(self):
        """Collect results from the successful process."""
        node = self.ctx.children[self.ctx.iteration - 1]

        # Pass through all outputs from the subprocess
        for label in node.outputs:
            self.out(label, node.outputs[label])

        self.report(f"WorkChain completed successfully after {self.ctx.iteration} iterations")
