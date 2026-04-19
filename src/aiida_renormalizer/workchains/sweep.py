"""WorkChains for parameter sweep operations."""
from __future__ import annotations

from aiida import orm
from aiida.engine import WorkChain, ToContext

import numpy as np


class ParameterSweepWorkChain(WorkChain):
    """Base WorkChain for parallel parameter sweeps.

    This WorkChain provides a framework for running multiple calculations
    in parallel with different parameter values, then aggregating results.

    Subclasses should implement:
    - get_parameter_values(): Return list of parameter values to sweep
    - build_inputs(param_value): Build inputs dict for each calculation
    - aggregate_results(): Process and aggregate all results

    Inputs:
        base_inputs: Dict - Base inputs for all calculations
        parameter_name: Str - Name of parameter to sweep
        parameter_values: List - List of parameter values
        max_concurrent: Int - Maximum concurrent calculations (optional)
        workchain_class: Str - Entry point name of WorkChain to run

    Outputs:
        sweep_results: ArrayData - Aggregated results
        output_parameters: Dict - Sweep statistics

    Exit Codes:
        380: ERROR_INVALID_SWEEP_PARAMETERS
        381: ERROR_CALCULATION_FAILED
        382: ERROR_AGGREGATION_FAILED
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        # Inputs
        spec.input(
            "base_inputs",
            valid_type=orm.Dict,
            help="Base inputs for all calculations (will be modified per sweep)",
        )
        spec.input(
            "parameter_name",
            valid_type=orm.Str,
            help="Name of parameter to sweep",
        )
        spec.input(
            "parameter_values",
            valid_type=orm.List,
            help="List of parameter values to sweep",
        )
        spec.input(
            "max_concurrent",
            valid_type=orm.Int,
            required=False,
            default=lambda: orm.Int(10),
            help="Maximum number of concurrent calculations",
        )
        spec.input(
            "workchain_class",
            valid_type=orm.Str,
            help="Entry point name of WorkChain to run",
        )

        # Outputs
        spec.output("sweep_results", valid_type=orm.ArrayData, help="Aggregated sweep results")
        spec.output("output_parameters", valid_type=orm.Dict, help="Sweep statistics")

        # Exit codes
        spec.exit_code(
            380,
            "ERROR_INVALID_SWEEP_PARAMETERS",
            message="Invalid sweep parameters",
        )
        spec.exit_code(
            381,
            "ERROR_CALCULATION_FAILED",
            message="One or more calculations failed",
        )
        spec.exit_code(
            382,
            "ERROR_AGGREGATION_FAILED",
            message="Result aggregation failed",
        )

        # Outline
        spec.outline(
            cls.setup,
            cls.launch_sweep,
            cls.collect_results,
            cls.aggregate_results,
        )

    def setup(self):
        """Initialize the sweep."""
        self.report("Setting up parameter sweep")

        param_name = self.inputs.parameter_name.value
        param_values = self.inputs.parameter_values.get_list()

        self.report(f"Sweeping parameter '{param_name}' over {len(param_values)} values")

        self.ctx.param_name = param_name
        self.ctx.param_values = param_values
        self.ctx.calculations = []

    def launch_sweep(self):
        """Launch all calculations in parallel."""
        from aiida.plugins import WorkflowFactory

        # Load WorkChain class
        workchain_entry_point = self.inputs.workchain_class.value
        try:
            WorkChainClass = WorkflowFactory(workchain_entry_point)
        except Exception as e:
            self.report(f"ERROR: Failed to load WorkChain '{workchain_entry_point}': {e}")
            return self.exit_codes.ERROR_INVALID_SWEEP_PARAMETERS

        param_values = self.ctx.param_values
        base_inputs = self.inputs.base_inputs.get_dict()

        # Launch calculations
        for i, param_value in enumerate(param_values):
            # Build inputs for this parameter value
            inputs = dict(base_inputs)
            inputs[self.ctx.param_name] = self._wrap_parameter_value(param_value)

            # Submit calculation
            try:
                future = self.submit(WorkChainClass, **inputs)
                self.to_context(**{f"calc_{i}": future})
                self.ctx.calculations.append(i)
                self.report(f"Launched calculation {i}: {self.ctx.param_name}={param_value}")
            except Exception as e:
                self.report(f"ERROR: Failed to launch calculation {i}: {e}")
                return self.exit_codes.ERROR_CALCULATION_FAILED

    def _wrap_parameter_value(self, value):
        """Wrap parameter value in appropriate AiiDA type."""
        if isinstance(value, float):
            return orm.Float(value)
        elif isinstance(value, int):
            return orm.Int(value)
        elif isinstance(value, str):
            return orm.Str(value)
        else:
            return orm.Str(str(value))

    def collect_results(self):
        """Collect results from all calculations."""
        self.report("Collecting results from sweep")

        self.ctx.results = []
        failed = []

        for i in self.ctx.calculations:
            calc = getattr(self.ctx, f"calc_{i}")

            if not calc.is_finished_ok:
                self.report(f"WARNING: Calculation {i} failed")
                failed.append(i)
                continue

            # Extract result
            result = {
                "param_value": self.ctx.param_values[i],
                "calculation_uuid": calc.uuid,
            }

            # Add outputs
            if "output_parameters" in calc.outputs:
                result["output_params"] = calc.outputs.output_parameters.get_dict()

            self.ctx.results.append(result)

        if failed:
            self.report(f"WARNING: {len(failed)} calculations failed")
            # Could return error here, or continue with partial results

    def aggregate_results(self):
        """Aggregate results into output arrays."""
        self.report("Aggregating results")

        # Create output arrays
        param_values = np.array([r["param_value"] for r in self.ctx.results])

        # Basic aggregation - subclasses can override
        from aiida.orm import ArrayData

        results_data = ArrayData()
        results_data.set_array("parameter_values", param_values)

        # Add any scalar outputs
        for result in self.ctx.results:
            if "output_params" in result:
                # Extract common scalar outputs
                # This is a simple implementation - subclasses can do more
                pass

        self.out("sweep_results", results_data)

        # Output statistics
        stats = {
            "parameter_name": self.ctx.param_name,
            "n_values": len(self.ctx.param_values),
            "n_completed": len(self.ctx.results),
        }

        self.out("output_parameters", orm.Dict(stats))

        self.report(f"Sweep completed: {len(self.ctx.results)}/{len(self.ctx.param_values)} calculations")


class TemperatureSweepWorkChain(ParameterSweepWorkChain):
    """WorkChain for temperature sweep (finite-temperature calculations).

    Sweeps over temperature values and runs a WorkChain for each temperature.

    Inputs:
        base_inputs: Dict - Base inputs (without temperature)
        temperatures: List - List of temperature values
        workchain_class: Str - Entry point name (e.g., 'reno.thermal_state')

    Outputs:
        sweep_results: ArrayData - Temperature vs. observable data
        output_parameters: Dict - Sweep statistics
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        # Override parameter_name and parameter_values with temperature-specific names
        spec.input(
            "temperatures",
            valid_type=orm.List,
            required=False,
            help="List of temperature values",
        )

    def setup(self):
        """Override to set parameter name and values."""
        # Set parameter name
        self.inputs.parameter_name = orm.Str("temperature")

        # Use temperatures if provided, otherwise use parameter_values
        if "temperatures" in self.inputs:
            self.inputs.parameter_values = self.inputs.temperatures

        super().setup()

    def aggregate_results(self):
        """Aggregate temperature sweep results."""
        self.report("Aggregating temperature sweep results")

        # Call parent aggregation
        super().aggregate_results()

        # Add temperature-specific aggregation
        # For example, extract free energies, partition functions, etc.
        results = self.ctx.results

        if results and "output_params" in results[0]:
            # Extract free energies
            free_energies = []
            partition_functions = []

            for result in results:
                params = result.get("output_params", {})
                if "free_energy" in params:
                    free_energies.append(params["free_energy"])
                if "partition_function" in params:
                    partition_functions.append(params["partition_function"])

            # Add to output
            if free_energies:
                sweep_results = self.outputs.sweep_results
                sweep_results.set_array("free_energies", np.array(free_energies))

            if partition_functions:
                sweep_results = self.outputs.sweep_results
                sweep_results.set_array("partition_functions", np.array(partition_functions))


class BondDimensionSweepWorkChain(ParameterSweepWorkChain):
    """WorkChain for bond dimension sweep (convergence testing).

    Sweeps over bond dimension values and runs a WorkChain for each.

    Inputs:
        base_inputs: Dict - Base inputs
        m_values: List - List of bond dimensions
        workchain_class: Str - Entry point name (e.g., 'reno.ground_state')

    Outputs:
        sweep_results: ArrayData - Bond dimension vs. energy data
        output_parameters: Dict - Convergence statistics
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        # Override with bond dimension-specific names
        spec.input(
            "m_values",
            valid_type=orm.List,
            required=False,
            help="List of bond dimension values",
        )

    def setup(self):
        """Override to set parameter name and values."""
        # Set parameter name
        self.inputs.parameter_name = orm.Str("M_max")

        # Use m_values if provided, otherwise use parameter_values
        if "m_values" in self.inputs:
            self.inputs.parameter_values = self.inputs.m_values

        super().setup()

    def aggregate_results(self):
        """Aggregate bond dimension sweep results."""
        self.report("Aggregating bond dimension sweep results")

        # Call parent aggregation
        super().aggregate_results()

        # Extract energies for convergence analysis
        results = self.ctx.results

        if results and "output_params" in results[0]:
            energies = []

            for result in results:
                params = result.get("output_params", {})
                if "energy" in params:
                    energies.append(params["energy"])

            if energies:
                # Add to output
                sweep_results = self.outputs.sweep_results
                sweep_results.set_array("energies", np.array(energies))

                # Compute convergence metrics
                if len(energies) > 1:
                    energy_diffs = np.abs(np.diff(energies))
                    sweep_results.set_array("energy_differences", energy_diffs)

                    # Find converged bond dimension
                    threshold = 1e-6  # Could be an input
                    converged_idx = np.where(energy_diffs < threshold)[0]
                    if len(converged_idx) > 0:
                        converged_m = self.ctx.param_values[converged_idx[0]]
                        self.report(f"Converged at M={converged_m}")


class FrequencySweepWorkChain(ParameterSweepWorkChain):
    """WorkChain for frequency sweep (spectra calculations).

    Sweeps over frequency values and runs a WorkChain for each frequency.

    Inputs:
        base_inputs: Dict - Base inputs
        frequencies: List - List of frequency values
        workchain_class: Str - Entry point name (e.g., 'reno.absorption')

    Outputs:
        sweep_results: ArrayData - Frequency vs. spectrum intensity
        output_parameters: Dict - Sweep statistics
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        # Override with frequency-specific names
        spec.input(
            "frequencies",
            valid_type=orm.List,
            required=False,
            help="List of frequency values",
        )

    def setup(self):
        """Override to set parameter name and values."""
        # Set parameter name
        self.inputs.parameter_name = orm.Str("frequency")

        # Use frequencies if provided, otherwise use parameter_values
        if "frequencies" in self.inputs:
            self.inputs.parameter_values = self.inputs.frequencies

        super().setup()

    def aggregate_results(self):
        """Aggregate frequency sweep results into spectrum."""
        self.report("Aggregating frequency sweep results")

        # Call parent aggregation
        super().aggregate_results()

        # Extract spectrum intensities
        results = self.ctx.results

        if results and "output_params" in results[0]:
            intensities = []

            for result in results:
                params = result.get("output_params", {})
                # Spectrum intensity might be under different keys
                intensity = params.get("intensity", params.get("spectrum", params.get("signal")))
                if intensity is not None:
                    intensities.append(intensity)

            if intensities:
                # Add to output
                sweep_results = self.outputs.sweep_results
                sweep_results.set_array("intensities", np.array(intensities))

                # Could also compute full spectrum array here
                frequencies = np.array([r["param_value"] for r in results])
                spectrum = np.column_stack([frequencies, intensities])
                sweep_results.set_array("spectrum", spectrum)
