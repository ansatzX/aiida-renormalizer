"""WorkChain for absorption spectrum calculation."""
from __future__ import annotations

from aiida import orm
from aiida.engine import WorkChain, ToContext, if_

from aiida_renormalizer.calculations.composite.dmrg import DMRGCalcJob
from aiida_renormalizer.calculations.composite.imag_time import ImagTimeCalcJob
from aiida_renormalizer.calculations.spectra.spectra_zero_t import SpectraZeroTCalcJob
from aiida_renormalizer.calculations.spectra.spectra_finite_t import SpectraFiniteTCalcJob
from aiida_renormalizer.data import ModelData, MpsData, MpoData


class AbsorptionWorkChain(WorkChain):
    """WorkChain for absorption spectrum calculation.

    This WorkChain orchestrates the full absorption spectrum workflow:
    1. Ground state preparation (optional if provided)
    2. Spectrum calculation (zero-T or finite-T)
    3. Post-processing (optional)

    It supports:
    - Zero-temperature and finite-temperature spectra
    - One-way and two-way propagation
    - Pre-computed ground states

    Inputs:
        model: ModelData - System definition
        mpo: MpoData (optional) - Hamiltonian operator
        ground_state: MpsData (optional) - Pre-computed ground state
        temperature: Float (optional) - Temperature (None = zero-T)
        spectratype: Str - "abs" or "emi"
        propagation: Str - "one_way" or "two_way"
        gs_method: Str - "dmrg" or "imag_time" (ground state method)
        config: ConfigData - Evolution configuration
        code: AbstractCode - Code to use

    Outputs:
        spectrum: ArrayData - Absorption spectrum
        ground_state: MpsData - Ground state used
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
            valid_type=MpoData,
            required=False,
            help="Hamiltonian MPO (built if not provided)",
        )
        spec.input(
            "ground_state",
            valid_type=MpsData,
            required=False,
            help="Pre-computed ground state (will be computed if not provided)",
        )
        spec.input(
            "temperature",
            valid_type=orm.Float,
            required=False,
            help="Temperature (None = zero-temperature)",
        )
        spec.input(
            "spectratype",
            valid_type=orm.Str,
            default=lambda: orm.Str("abs"),
            help="Spectrum type: 'abs' or 'emi'",
        )
        spec.input(
            "propagation",
            valid_type=orm.Str,
            default=lambda: orm.Str("two_way"),
            help="Propagation scheme: 'one_way' or 'two_way'",
        )
        spec.input(
            "gs_method",
            valid_type=orm.Str,
            default=lambda: orm.Str("dmrg"),
            help="Ground state method: 'dmrg' or 'imag_time'",
        )
        spec.input("config", valid_type=orm.Dict, required=False, help="Evolution configuration")
        spec.input("code", valid_type=orm.AbstractCode, help="Code to use")

        # Additional inputs for finite-T
        spec.input("insteps", valid_type=orm.Int, required=False, help="Imaginary time steps")
        spec.input("gs_shift", valid_type=orm.Float, required=False, help="Ground state energy shift")

        # Outputs
        spec.output("spectrum", valid_type=orm.ArrayData, help="Absorption spectrum")
        spec.output("ground_state", valid_type=MpsData, help="Ground state used")
        spec.output("output_parameters", valid_type=orm.Dict, help="Spectrum data")

        # Exit codes
        spec.exit_code(
            330,
            "ERROR_GROUND_STATE_FAILED",
            message="Ground state calculation failed",
        )
        spec.exit_code(
            331,
            "ERROR_SPECTRUM_FAILED",
            message="Spectrum calculation failed",
        )

        # Outline
        spec.outline(
            cls.setup,
            if_(cls.needs_ground_state)(
                cls.prepare_ground_state,
                cls.inspect_ground_state,
            ),
            if_(cls.is_zero_temperature)(
                cls.run_zero_t_spectrum,
            ).else_(
                cls.run_finite_t_spectrum,
            ),
            cls.inspect_spectrum,
            cls.finalize,
        )

    def setup(self):
        """Initialize the WorkChain."""
        self.report("Starting absorption spectrum calculation")

        # Validate spectratype
        if self.inputs.spectratype.value not in ["abs", "emi"]:
            raise ValueError(f"Invalid spectratype: {self.inputs.spectratype.value}")

    def needs_ground_state(self):
        """Check if ground state needs to be prepared."""
        return "ground_state" not in self.inputs

    def prepare_ground_state(self):
        """Prepare ground state using DMRG or ImagTime."""
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

    def is_zero_temperature(self):
        """Check if zero-temperature calculation."""
        return "temperature" not in self.inputs

    def run_zero_t_spectrum(self):
        """Run zero-temperature spectrum calculation."""
        self.report("Running zero-temperature spectrum calculation")

        # Get ground state
        ground_state = self.inputs.ground_state if "ground_state" in self.inputs else self.ctx.ground_state

        # Build inputs
        inputs = {
            "model": self.inputs.model,
            "initial_mps": ground_state,
            "spectratype": self.inputs.spectratype,
            "propagation": self.inputs.propagation,
            "code": self.inputs.code,
        }

        if "mpo" in self.inputs:
            inputs["mpo"] = self.inputs.mpo
        if "config" in self.inputs:
            inputs["config"] = self.inputs.config

        future = self.submit(SpectraZeroTCalcJob, **inputs)

        return ToContext(spectrum_calc=future)

    def run_finite_t_spectrum(self):
        """Run finite-temperature spectrum calculation."""
        self.report("Running finite-temperature spectrum calculation")

        # Get ground state (thermal density matrix for finite-T)
        ground_state = self.inputs.ground_state if "ground_state" in self.inputs else self.ctx.ground_state

        # Build inputs
        inputs = {
            "model": self.inputs.model,
            "initial_mps": ground_state,
            "temperature": self.inputs.temperature,
            "spectratype": self.inputs.spectratype,
            "code": self.inputs.code,
        }

        if "mpo" in self.inputs:
            inputs["mpo"] = self.inputs.mpo
        if "config" in self.inputs:
            inputs["config"] = self.inputs.config
        if "insteps" in self.inputs:
            inputs["insteps"] = self.inputs.insteps
        if "gs_shift" in self.inputs:
            inputs["gs_shift"] = self.inputs.gs_shift

        future = self.submit(SpectraFiniteTCalcJob, **inputs)

        return ToContext(spectrum_calc=future)

    def inspect_spectrum(self):
        """Inspect spectrum calculation."""
        calc = self.ctx.spectrum_calc

        if not calc.is_finished_ok:
            self.report(f"Spectrum calculation failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_SPECTRUM_FAILED

        # Store results
        self.ctx.spectrum_params = calc.outputs.output_parameters.get_dict()
        self.report("Spectrum calculation completed successfully")

    def finalize(self):
        """Collect and output final results."""
        self.report("Finalizing absorption spectrum calculation")

        # Output ground state
        ground_state = self.inputs.ground_state if "ground_state" in self.inputs else self.ctx.ground_state
        self.out("ground_state", ground_state)

        # Output spectrum parameters (contains spectrum data)
        # Note: In a full implementation, we'd create an ArrayData with the spectrum
        # For now, we output the parameters Dict
        params = orm.Dict(self.ctx.spectrum_params)
        self.out("output_parameters", params)

        # Create a simple spectrum ArrayData if autocorrelation is available
        if "autocorrelation" in self.ctx.spectrum_params:
            import numpy as np
            from aiida.orm import ArrayData

            spectrum = ArrayData()
            spectrum.set_array(
                "autocorrelation",
                np.array(self.ctx.spectrum_params["autocorrelation"])
            )
            self.out("spectrum", spectrum)

        self.report(
            f"Absorption spectrum calculation completed: "
            f"type={self.inputs.spectratype.value}, T={'0' if self.is_zero_temperature() else str(self.inputs.temperature.value)}"
        )
