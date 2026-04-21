"""WorkChain for custom declarative pipeline orchestration."""
from __future__ import annotations

from aiida import orm
from aiida.engine import WorkChain, ToContext, while_

from aiida_renormalizer.data import ModelData, MPSData


class CustomPipelineWorkChain(WorkChain):
    """WorkChain for user-defined declarative pipeline orchestration.

    This WorkChain allows users to define a sequence of L1/L2/L3 operations
    that will be executed in order with full provenance tracking.

    Pipeline specification:
        pipeline: orm.List - List of step specifications

    Each step is a dictionary:
        {
            'step': 'step_type',  # e.g., 'compress', 'tdvp', 'dmrg'
            'inputs': {...},       # Step-specific inputs (optional)
        }

    Supported step types:
        - 'compress': CompressCalcJob
        - 'expectation': ExpectationCalcJob
        - 'tdvp': TDVPCalcJob
        - 'dmrg': DMRGCalcJob
        - 'imag_time': ImagTimeCalcJob
        - 'script': RenoScriptCalcJob

    Inputs:
        pipeline: List - List of step specifications
        model: ModelData - System definition (optional, can be per-step)
        initial_state: MPSData - Initial MPS state
        code: AbstractCode - Code to use (optional, can be per-step)

    Outputs:
        final_state: MPSData - Final MPS after pipeline execution
        output_parameters: Dict - Pipeline execution statistics

    Exit Codes:
        370: ERROR_INVALID_PIPELINE
        371: ERROR_UNKNOWN_STEP_TYPE
        372: ERROR_STEP_FAILED
    """

    # Dispatch table mapping step types to their CalcJob classes
    STEP_DISPATCH_TABLE = {
        'compress': ('aiida_renormalizer.calculations.basic', 'CompressCalcJob'),
        'expectation': ('aiida_renormalizer.calculations.basic', 'ExpectationCalcJob'),
        'build_mpo': ('aiida_renormalizer.calculations.basic', 'BuildMPOCalcJob'),
        'tdvp': ('aiida_renormalizer.calculations.composite', 'TDVPCalcJob'),
        'dmrg': ('aiida_renormalizer.calculations.composite', 'DMRGCalcJob'),
        'imag_time': ('aiida_renormalizer.calculations.composite', 'ImagTimeCalcJob'),
        'thermal_prop': ('aiida_renormalizer.calculations.composite', 'ThermalPropCalcJob'),
        'property': ('aiida_renormalizer.calculations.composite', 'PropertyCalcJob'),
        'spectra_zero_t': ('aiida_renormalizer.calculations.spectra', 'SpectraZeroTCalcJob'),
        'spectra_finite_t': ('aiida_renormalizer.calculations.spectra', 'SpectraFiniteTCalcJob'),
        'kubo': ('aiida_renormalizer.calculations.spectra', 'KuboCalcJob'),
        'script': ('aiida_renormalizer.calculations.scripted', 'RenoScriptCalcJob'),
        'bath_spin_boson_model': ('aiida_renormalizer.calculations.bath', 'BathSpinBosonModelCalcJob'),
    }

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        # Inputs
        spec.input(
            "pipeline",
            valid_type=orm.List,
            help="List of step specifications (each step is a dict with 'step' and 'inputs')",
        )
        spec.input(
            "model",
            valid_type=ModelData,
            required=False,
            help="System definition (can be overridden per-step)",
        )
        spec.input(
            "initial_state",
            valid_type=MPSData,
            required=False,
            help="Initial MPS state",
        )
        spec.input(
            "code",
            valid_type=orm.AbstractCode,
            required=False,
            help="Code to use (can be overridden per-step)",
        )

        # Outputs
        spec.output("final_state", valid_type=MPSData, required=False, help="Final MPS state")
        spec.output("output_parameters", valid_type=orm.Dict, help="Pipeline execution statistics")

        # Exit codes
        spec.exit_code(
            370,
            "ERROR_INVALID_PIPELINE",
            message="Invalid pipeline specification",
        )
        spec.exit_code(
            371,
            "ERROR_UNKNOWN_STEP_TYPE",
            message="Unknown step type in pipeline",
        )
        spec.exit_code(
            372,
            "ERROR_STEP_FAILED",
            message="Pipeline step failed",
        )

        # Outline
        spec.outline(
            cls.setup,
            while_(cls.has_more_steps)(
                cls.dispatch_step,
                cls.collect_result,
            ),
            cls.finalize,
        )

    def setup(self):
        """Initialize the WorkChain."""
        self.report("Starting custom pipeline execution")

        # Validate pipeline
        pipeline = self.inputs.pipeline.get_list()

        if not isinstance(pipeline, list):
            self.report("ERROR: Pipeline must be a list")
            return self.exit_codes.ERROR_INVALID_PIPELINE

        if len(pipeline) == 0:
            self.report("WARNING: Empty pipeline")

        # Initialize context
        self.ctx.pipeline = pipeline
        self.ctx.step_index = 0
        self.ctx.current_state = None
        self.ctx.step_results = []

        # Initialize current state from inputs
        if "initial_state" in self.inputs:
            self.ctx.current_state = self.inputs.initial_state

        self.report(f"Pipeline has {len(pipeline)} steps")

    def has_more_steps(self):
        """Check if there are more steps to execute."""
        return self.ctx.step_index < len(self.ctx.pipeline)

    def dispatch_step(self):
        """Submit appropriate CalcJob based on step specification."""
        import importlib
        from aiida.engine import CalcJob

        # Get current step
        step_spec = self.ctx.pipeline[self.ctx.step_index]
        step_type = step_spec.get("step")
        step_inputs = step_spec.get("inputs", {})

        self.report(f"Dispatching step {self.ctx.step_index}: {step_type}")

        # Validate step type
        if step_type not in self.STEP_DISPATCH_TABLE:
            self.report(f"ERROR: Unknown step type: {step_type}")
            return self.exit_codes.ERROR_UNKNOWN_STEP_TYPE

        # Import CalcJob class
        module_name, class_name = self.STEP_DISPATCH_TABLE[step_type]
        try:
            module = importlib.import_module(module_name)
            CalcJobClass = getattr(module, class_name)
            if not issubclass(CalcJobClass, CalcJob):
                raise TypeError(f"{class_name} is not a CalcJob")
        except (ImportError, AttributeError) as e:
            self.report(f"ERROR: Failed to import {class_name}: {e}")
            return self.exit_codes.ERROR_UNKNOWN_STEP_TYPE
        except TypeError as e:
            self.report(f"ERROR: {e}")
            return self.exit_codes.ERROR_UNKNOWN_STEP_TYPE

        # Build inputs
        inputs = {}

        # Add model (from pipeline inputs or step-specific)
        if "model" in step_inputs:
            inputs["model"] = step_inputs["model"]
        elif "model" in self.inputs:
            inputs["model"] = self.inputs.model

        # Add code (from pipeline inputs or step-specific)
        if "code" in step_inputs:
            inputs["code"] = step_inputs["code"]
        elif "code" in self.inputs:
            inputs["code"] = self.inputs.code

        # Add current state as initial MPS (if applicable)
        if self.ctx.current_state is not None:
            # Different CalcJobs use different input names for initial state
            if step_type in ['compress', 'tdvp', 'dmrg', 'imag_time', 'thermal_prop']:
                inputs["initial_mps"] = self.ctx.current_state
            elif step_type == 'expectation':
                inputs["mps"] = self.ctx.current_state

        # Add step-specific inputs
        for key, value in step_inputs.items():
            if key not in ["model", "code"]:  # Already handled
                inputs[key] = value

        # Submit calculation
        try:
            future = self.submit(CalcJobClass, **inputs)
            self.to_context(**{f"step_{self.ctx.step_index}": future})
        except Exception as e:
            self.report(f"ERROR: Failed to submit step {step_type}: {e}")
            return self.exit_codes.ERROR_STEP_FAILED

    def collect_result(self):
        """Collect result from completed step and advance."""
        # Get calculation result
        calc = getattr(self.ctx, f"step_{self.ctx.step_index}")

        if not calc.is_finished_ok:
            self.report(f"ERROR: Step {self.ctx.step_index} failed with exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_STEP_FAILED

        # Extract result (output MPS if available)
        if "output_mps" in calc.outputs:
            self.ctx.current_state = calc.outputs.output_mps
            self.report(f"Step {self.ctx.step_index} completed: new MPS stored")

        # Store step result
        step_result = {
            "step_index": self.ctx.step_index,
            "calculation_uuid": calc.uuid,
            "exit_status": calc.exit_status,
        }
        self.ctx.step_results.append(step_result)

        # Advance to next step
        self.ctx.step_index += 1

    def finalize(self):
        """Finalize pipeline and output results."""
        self.report("Finalizing pipeline execution")

        # Output final state
        if self.ctx.current_state is not None:
            self.out("final_state", self.ctx.current_state)

        # Output statistics
        stats = {
            "n_steps": len(self.ctx.pipeline),
            "completed_steps": self.ctx.step_index,
            "step_results": self.ctx.step_results,
        }

        self.out("output_parameters", orm.Dict(stats))

        self.report(f"Pipeline completed: {self.ctx.step_index}/{len(self.ctx.pipeline)} steps")
