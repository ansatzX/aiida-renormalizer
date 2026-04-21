"""CalcJob for zero-temperature absorption/emission spectra."""
from __future__ import annotations

import os
import tempfile

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import ModelData, MPSData, MPOData


class SpectraZeroTCalcJob(RenoBaseCalcJob):
    """Zero-temperature absorption/emission spectrum calculation.

    Corresponds to Reno API: SpectraOneWayPropZeroT or SpectraTwoWayPropZeroT

    Inputs:
        model: ModelData - System definition
        mpo: MPOData - Hamiltonian operator (optional, will be constructed if not provided)
        initial_mps: MPSData (optional) - Initial ground state, will be optimized if not provided
        spectratype: Str - "abs" for absorption, "emi" for emission
        propagation: Str - "one_way" or "two_way" propagation scheme
        config: ConfigData - EvolveConfig with time evolution parameters
        offset: Float - Energy offset for Hamiltonian

    Outputs:
        output_parameters: Dict - Time series, autocorrelation function, spectrum
    """

    _template_name = "spectra_zero_t_driver.py.jinja"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        # Additional inputs
        spec.input(
            "mpo",
            valid_type=MPOData,
            required=False,
            help="Hamiltonian MPO (will be constructed if not provided)",
        )
        spec.input(
            "initial_mps",
            valid_type=MPSData,
            required=False,
            help="Initial ground state MPS (will be optimized if not provided)",
        )
        spec.input(
            "spectratype",
            valid_type=orm.Str,
            help="Spectrum type: 'abs' for absorption, 'emi' for emission",
        )
        spec.input(
            "propagation",
            valid_type=orm.Str,
            default=lambda: orm.Str("two_way"),
            help="Propagation scheme: 'one_way' or 'two_way'",
        )
        spec.input(
            "offset",
            valid_type=orm.Float,
            required=False,
            help="Energy offset for Hamiltonian",
        )

        # Outputs
        spec.output(
            "output_parameters",
            valid_type=orm.Dict,
            help="Time series, autocorrelation function, and spectrum data",
        )

        # Exit codes
        spec.exit_code(
            300,
            "ERROR_EVOLUTION_FAILED",
            message="Time evolution failed to converge",
        )

    def _get_template_context(self) -> dict:
        """Provide context for Jinja2 template rendering."""
        context = super()._get_template_context()
        context["has_mpo"] = "mpo" in self.inputs
        context["has_initial_mps"] = "initial_mps" in self.inputs
        context["has_offset"] = "offset" in self.inputs
        context["spectratype"] = self.inputs.spectratype.value
        context["propagation"] = self.inputs.propagation.value
        return context

    def _write_input_files(self, folder) -> None:
        """Write input files for spectrum calculation."""
        import json

        super()._write_input_files(folder)

        # Write MPO (if provided)
        if "mpo" in self.inputs:
            mpo_data = self.inputs.mpo
            model_data = self.inputs.model
            MPO = mpo_data.load_mpo(model_data)

            with tempfile.TemporaryDirectory() as tmpdir:
                mpo_path = os.path.join(tmpdir, "mpo")
                MPO.dump(mpo_path)
                actual = mpo_path + ".npz" if os.path.exists(mpo_path + ".npz") else mpo_path
                with open(actual, "rb") as src:
                    with folder.open("initial_mpo.npz", "wb") as dst:
                        dst.write(src.read())

        # Write initial MPS (if provided)
        if "initial_mps" in self.inputs:
            mps_data = self.inputs.initial_mps
            MPS = mps_data.load_mps(self.inputs.model)

            with tempfile.TemporaryDirectory() as tmpdir:
                mps_path = os.path.join(tmpdir, "mps")
                MPS.dump(mps_path)
                actual = mps_path + ".npz" if os.path.exists(mps_path + ".npz") else mps_path
                with open(actual, "rb") as src:
                    with folder.open("initial_mps.npz", "wb") as dst:
                        dst.write(src.read())

        # Write spectrum parameters
        params = {
            "spectratype": self.inputs.spectratype.value,
            "propagation": self.inputs.propagation.value,
        }
        if "offset" in self.inputs:
            params["offset"] = self.inputs.offset.value

        with folder.open("input_spectra_params.json", "w") as f:
            json.dump(params, f, indent=2)
