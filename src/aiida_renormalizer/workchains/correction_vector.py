"""WorkChain for correction vector frequency-domain spectra."""
from __future__ import annotations

import numpy as np

from aiida import orm
from aiida.engine import WorkChain, ToContext, if_

from aiida_renormalizer.calculations.spectra.correction_vector import CorrectionVectorCalcJob
from aiida_renormalizer.calculations.composite.dmrg import DMRGCalcJob
from aiida_renormalizer.data import ModelData, MPSData, MPOData


class CorrectionVectorWorkChain(WorkChain):
    """WorkChain for correction vector frequency-domain spectra.

    This WorkChain orchestrates the correction vector (CV) spectrum workflow:
    1. Ground state preparation (optional if provided)
    2. Frequency grid preparation (batch or single)
    3. CV spectrum calculation at each frequency
    4. Spectrum aggregation

    It supports:
    - Batch frequency processing (parallel execution)
    - Automatic ground state preparation
    - Zero-temperature and finite-temperature spectra
    - Configurable broadening and convergence

    Inputs:
        model: ModelData - System definition
        mpo: MPOData (optional) - Hamiltonian operator
        ground_state: MPSData (optional) - Pre-computed ground state
        frequencies: ArrayData - Frequency grid
        eta: Float - Broadening parameter
        m_max: Int - Maximum bond dimension
        method: Str - "1site" or "2site" optimization
        spectratype: Str (optional) - "abs", "emi", or None
        gs_method: Str - "dmrg" or "imag_time" (ground state method)
        n_cores: Int - Number of cores for parallel frequency calculation
        code: AbstractCode - Code to use

    Outputs:
        spectrum: ArrayData - Full spectrum (frequency vs intensity)
        ground_state: MPSData - Ground state used
        output_parameters: Dict - Spectrum data and statistics
    """

    @classmethod
    def define(cls, spec):
        """Define the process specification."""
        super().define(spec)

        # Inputs
        spec.input("model", valid_type=ModelData, help="System definition")
        spec.input(
            "mpo",
            valid_type=MPOData,
            required=False,
            help="Hamiltonian MPO (built if not provided)",
        )
        spec.input(
            "ground_state",
            valid_type=MPSData,
            required=False,
            help="Pre-computed ground state (will be computed if not provided)",
        )
        spec.input(
            "frequencies",
            valid_type=orm.ArrayData,
            help="Frequency grid for spectrum calculation",
        )
        spec.input(
            "eta",
            valid_type=orm.Float,
            help="Broadening parameter for spectrum",
        )
        spec.input(
            "m_max",
            valid_type=orm.Int,
            default=lambda: orm.Int(100),
            help="Maximum bond dimension for correction vector MPS",
        )
        spec.input(
            "method",
            valid_type=orm.Str,
            default=lambda: orm.Str("1site"),
            help="Optimization method: '1site' or '2site'",
        )
        spec.input(
            "spectratype",
            valid_type=orm.Str,
            required=False,
            help="Spectrum type: 'abs', 'emi', or None",
        )
        spec.input(
            "gs_method",
            valid_type=orm.Str,
            default=lambda: orm.Str("dmrg"),
            help="Ground state method: 'dmrg' or 'imag_time'",
        )
        spec.input(
            "rtol",
            valid_type=orm.Float,
            default=lambda: orm.Float(1e-5),
            help="Relative tolerance for convergence",
        )
        spec.input(
            "n_cores",
            valid_type=orm.Int,
            default=lambda: orm.Int(1),
            help="Number of cores for parallel frequency calculation",
        )
        spec.input(
            "procedure_cv",
            valid_type=orm.List,
            required=False,
            help="Custom optimization procedure for CV",
        )
        spec.input("config", valid_type=orm.Dict, required=False, help="DMRG configuration")
        spec.input("code", valid_type=orm.AbstractCode, help="Code to use")

        # Outputs
        spec.output("spectrum", valid_type=orm.ArrayData, help="Full spectrum (frequency vs intensity)")
        spec.output("ground_state", valid_type=MPSData, help="Ground state used")
        spec.output("output_parameters", valid_type=orm.Dict, help="Spectrum data and statistics")

        # Exit codes
        spec.exit_code(
            390,
            "ERROR_GROUND_STATE_FAILED",
            message="Ground state calculation failed",
        )
        spec.exit_code(
            391,
            "ERROR_CV_CALCULATION_FAILED",
            message="Correction vector calculation failed",
        )
        spec.exit_code(
            392,
            "ERROR_SPECTRUM_AGGREGATION_FAILED",
            message="Spectrum aggregation failed",
        )

        # Outline
        spec.outline(
            cls.setup,
            if_(cls.needs_ground_state)(
                cls.prepare_ground_state,
                cls.inspect_ground_state,
            ),
            cls.run_cv_calculation,
            cls.inspect_cv_calculation,
            cls.aggregate_spectrum,
            cls.finalize,
        )

    def setup(self):
        """Initialize the WorkChain."""
        self.report("Starting correction vector spectrum calculation")

        # Validate spectratype if provided
        if "spectratype" in self.inputs:
            if self.inputs.spectratype.value not in ["abs", "emi", None]:
                raise ValueError(f"Invalid spectratype: {self.inputs.spectratype.value}")

        # Extract frequency grid
        freqs = self.inputs.frequencies.get_array("frequencies")
        self.ctx.n_frequencies = len(freqs)
        self.report(f"Frequency grid: {self.ctx.n_frequencies} points")

    def needs_ground_state(self):
        """Check if ground state needs to be prepared."""
        return "ground_state" not in self.inputs

    def prepare_ground_state(self):
        """Prepare ground state using DMRG or ImagTime."""
        from aiida_renormalizer.calculations.composite.imag_time import ImagTimeCalcJob

        method = self.inputs.gs_method.value
        self.report(f"Preparing ground state using {method}")

        # Build inputs based on method
        if method == "dmrg":
            inputs = {
                "model": self.inputs.model,
                "code": self.inputs.code,
            }
            if "mpo" in self.inputs:
                inputs["mpo"] = self.inputs.mpo
            if "config" in self.inputs:
                inputs["config"] = self.inputs.config

            future = self.submit(DMRGCalcJob, **inputs)

        elif method == "imag_time":
            inputs = {
                "model": self.inputs.model,
                "code": self.inputs.code,
                "beta": orm.Float(100.0),  # High beta for ground state
            }
            if "mpo" in self.inputs:
                inputs["mpo"] = self.inputs.mpo
            if "config" in self.inputs:
                inputs["config"] = self.inputs.config

            future = self.submit(ImagTimeCalcJob, **inputs)

        else:
            raise ValueError(f"Unknown ground state method: {method}")

        return ToContext(ground_state_calc=future)

    def inspect_ground_state(self):
        """Inspect ground state calculation."""
        calc = self.ctx.ground_state_calc

        if not calc.is_finished_ok:
            self.report(f"Ground state calculation failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_GROUND_STATE_FAILED

        self.ctx.ground_state = calc.outputs.output_mps
        self.report("Ground state prepared successfully")

    def run_cv_calculation(self):
        """Run correction vector calculation."""
        self.report("Running correction vector calculation")

        # Get ground state
        ground_state = self.inputs.ground_state if "ground_state" in self.inputs else self.ctx.ground_state

        # Build inputs
        inputs = {
            "model": self.inputs.model,
            "initial_mps": ground_state,
            "frequencies": self.inputs.frequencies,
            "eta": self.inputs.eta,
            "m_max": self.inputs.m_max,
            "method": self.inputs.method,
            "rtol": self.inputs.rtol,
            "n_cores": self.inputs.n_cores,
            "code": self.inputs.code,
        }

        if "mpo" in self.inputs:
            inputs["mpo"] = self.inputs.mpo
        if "spectratype" in self.inputs:
            inputs["spectratype"] = self.inputs.spectratype
        if "procedure_cv" in self.inputs:
            inputs["procedure_cv"] = self.inputs.procedure_cv

        future = self.submit(CorrectionVectorCalcJob, **inputs)

        return ToContext(cv_calc=future)

    def inspect_cv_calculation(self):
        """Inspect CV calculation."""
        calc = self.ctx.cv_calc

        if not calc.is_finished_ok:
            self.report(f"CV calculation failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_CV_CALCULATION_FAILED

        # Store results
        self.ctx.cv_params = calc.outputs.output_parameters.get_dict()
        self.report("CV calculation completed successfully")

    def aggregate_spectrum(self):
        """Aggregate spectrum from CV calculation results."""
        self.report("Aggregating spectrum")

        try:
            # Extract spectrum data
            params = self.ctx.cv_params

            # Create spectrum ArrayData
            spectrum = orm.ArrayData()

            # Set frequencies
            freqs = self.inputs.frequencies.get_array("frequencies")
            spectrum.set_array("frequencies", freqs)

            # Set spectrum intensities
            # Note: The actual spectrum data structure depends on CorrectionVectorCalcJob output
            if "spectrum" in params:
                intensities = np.array(params["spectrum"])
                spectrum.set_array("intensities", intensities)
            elif "spectrum_real" in params and "spectrum_imag" in params:
                # Complex spectrum
                real_part = np.array(params["spectrum_real"])
                imag_part = np.array(params["spectrum_imag"])
                spectrum.set_array("intensities_real", real_part)
                spectrum.set_array("intensities_imag", imag_part)
                spectrum.set_array("intensities", real_part + 1j * imag_part)

            self.ctx.spectrum = spectrum
            self.report("Spectrum aggregation completed")

        except Exception as e:
            self.report(f"Spectrum aggregation failed: {e}")
            return self.exit_codes.ERROR_SPECTRUM_AGGREGATION_FAILED

    def finalize(self):
        """Collect and output final results."""
        self.report("Finalizing correction vector spectrum calculation")

        # Output ground state
        ground_state = self.inputs.ground_state if "ground_state" in self.inputs else self.ctx.ground_state
        self.out("ground_state", ground_state)

        # Output spectrum
        self.out("spectrum", self.ctx.spectrum)

        # Output parameters
        params = orm.Dict(self.ctx.cv_params)
        self.out("output_parameters", params)

        self.report(
            f"Correction vector spectrum calculation completed: "
            f"{self.ctx.n_frequencies} frequency points, "
            f"eta={self.inputs.eta.value}, M_max={self.inputs.m_max.value}"
        )
