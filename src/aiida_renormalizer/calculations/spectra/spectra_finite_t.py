"""CalcJob for finite-temperature absorption/emission spectra."""
from __future__ import annotations

import os
import tempfile

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import ModelData, MpsData, MpoData


class SpectraFiniteTCalcJob(RenoBaseCalcJob):
    """Finite-temperature absorption/emission spectrum calculation.

    Corresponds to Reno API: SpectraFiniteT

    Inputs:
        model: ModelData - System definition
        mpo: MpoData - Hamiltonian operator (optional)
        temperature: Float - Temperature in atomic units
        spectratype: Str - "abs" for absorption, "emi" for emission
        insteps: Int - Steps for imaginary time propagation
        gs_shift: Float - Ground state energy shift
        config: ConfigData - EvolveConfig for real time propagation
        ievolve_config: ConfigData - EvolveConfig for imaginary time propagation
        icompress_config: ConfigData - CompressConfig for imaginary time propagation

    Outputs:
        output_parameters: Dict - Time series, autocorrelation, temperature info
    """

    _template_name = "spectra_finite_t_driver.py.jinja"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        # Additional inputs
        spec.input(
            "mpo",
            valid_type=MpoData,
            required=False,
            help="Hamiltonian MPO (will be constructed if not provided)",
        )
        spec.input(
            "initial_mps",
            valid_type=MpsData,
            required=False,
            help="Thermal density matrix (MpDm) if precomputed",
        )
        spec.input(
            "temperature",
            valid_type=orm.Float,
            help="Temperature in atomic units",
        )
        spec.input(
            "spectratype",
            valid_type=orm.Str,
            help="Spectrum type: 'abs' for absorption, 'emi' for emission",
        )
        spec.input(
            "insteps",
            valid_type=orm.Int,
            default=lambda: orm.Int(1),
            help="Steps for imaginary time propagation",
        )
        spec.input(
            "gs_shift",
            valid_type=orm.Float,
            required=False,
            help="Ground state energy shift",
        )
        spec.input_namespace(
            "ievolve_config",
            valid_type=orm.Dict,
            required=False,
            help="Config for imaginary time propagation",
        )
        spec.input_namespace(
            "icompress_config",
            valid_type=orm.Dict,
            required=False,
            help="Config for compression during imaginary time propagation",
        )

        # Outputs
        spec.output(
            "output_parameters",
            valid_type=orm.Dict,
            help="Time series, autocorrelation, temperature, spectrum data",
        )

        # Exit codes
        spec.exit_code(
            300,
            "ERROR_EVOLUTION_FAILED",
            message="Time evolution failed to converge",
        )
        spec.exit_code(
            301,
            "ERROR_THERMAL_PROP_FAILED",
            message="Thermal propagation failed",
        )

    def _get_template_context(self) -> dict:
        """Provide context for Jinja2 template rendering."""
        context = super()._get_template_context()
        context["has_mpo"] = "mpo" in self.inputs
        context["has_initial_mps"] = "initial_mps" in self.inputs
        context["has_gs_shift"] = "gs_shift" in self.inputs
        context["has_ievolve_config"] = "ievolve_config" in self.inputs
        context["has_icompress_config"] = "icompress_config" in self.inputs
        context["spectratype"] = self.inputs.spectratype.value
        context["temperature"] = self.inputs.temperature.value
        context["insteps"] = self.inputs.insteps.value
        return context

    def _write_input_files(self, folder) -> None:
        """Write input files for finite-T spectrum calculation."""
        import json

        super()._write_input_files(folder)

        # Write MPO (if provided)
        if "mpo" in self.inputs:
            mpo_data = self.inputs.mpo
            model_data = self.inputs.model
            mpo = mpo_data.load_mpo(model_data)

            with tempfile.TemporaryDirectory() as tmpdir:
                mpo_path = os.path.join(tmpdir, "mpo")
                mpo.dump(mpo_path)
                actual = mpo_path + ".npz" if os.path.exists(mpo_path + ".npz") else mpo_path
                with open(actual, "rb") as src:
                    with folder.open("initial_mpo.npz", "wb") as dst:
                        dst.write(src.read())

        # Write initial thermal state (if provided)
        if "initial_mps" in self.inputs:
            mps_data = self.inputs.initial_mps
            mps = mps_data.load_mps(self.inputs.model)

            with tempfile.TemporaryDirectory() as tmpdir:
                mps_path = os.path.join(tmpdir, "mps")
                mps.dump(mps_path)
                actual = mps_path + ".npz" if os.path.exists(mps_path + ".npz") else mps_path
                with open(actual, "rb") as src:
                    with folder.open("initial_thermal_state.npz", "wb") as dst:
                        dst.write(src.read())

        # Write spectrum parameters
        params = {
            "spectratype": self.inputs.spectratype.value,
            "temperature": self.inputs.temperature.value,
            "insteps": self.inputs.insteps.value,
        }
        if "gs_shift" in self.inputs:
            params["gs_shift"] = self.inputs.gs_shift.value

        with folder.open("input_spectra_params.json", "w") as f:
            json.dump(params, f, indent=2)

        # Write imaginary time configs (if provided)
        if "ievolve_config" in self.inputs:
            with folder.open("input_ievolve_config.json", "w") as f:
                json.dump(self.inputs.ievolve_config.get_dict(), f, indent=2)

        if "icompress_config" in self.inputs:
            with folder.open("input_icompress_config.json", "w") as f:
                json.dump(self.inputs.icompress_config.get_dict(), f, indent=2)
