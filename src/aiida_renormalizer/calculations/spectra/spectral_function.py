"""CalcJob for spectral function calculation."""
from __future__ import annotations

import os
import tempfile

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import ModelData, MPSData, MPOData


class SpectralFunctionCalcJob(RenoBaseCalcJob):
    """Spectral function calculation for translational invariant systems.

    Corresponds to Reno API: SpectralFunctionZT

    Calculates one-particle retarded Green's function:
        iG_ij(t) = <0|c_i(t) c†_j|0>

    and transforms to k-space for spectral function A(k,ω).

    Inputs:
        model: ModelData - System definition (should be TI1DModel)
        mpo: MPOData - Hamiltonian operator (optional)
        initial_mps: MPSData - Ground state MPS (optional)
        config: ConfigData - EvolveConfig for time evolution
        compress_config: ConfigData - CompressConfig

    Outputs:
        output_parameters: Dict - G array, Gk array, time series, occupations
    """

    _template_name = "spectral_function_driver.py.jinja"

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
            help="Ground state MPS (will be computed if not provided)",
        )
        spec.input_namespace(
            "compress_config",
            valid_type=orm.Dict,
            required=False,
            help="Config for MPS compression",
        )

        # Outputs
        spec.output(
            "output_parameters",
            valid_type=orm.Dict,
            help="G array, Gk array, time series, electron occupations",
        )

        # Exit codes
        spec.exit_code(
            300,
            "ERROR_EVOLUTION_FAILED",
            message="Time evolution failed",
        )

    def _get_template_context(self) -> dict:
        """Provide context for Jinja2 template rendering."""
        context = super()._get_template_context()
        context["has_mpo"] = "mpo" in self.inputs
        context["has_initial_mps"] = "initial_mps" in self.inputs
        context["has_compress_config"] = "compress_config" in self.inputs
        return context

    def _write_input_files(self, folder) -> None:
        """Write input files for spectral function calculation."""
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

        # Write compress config (if provided)
        if "compress_config" in self.inputs:
            payload = self._sanitize_compress_config_payload(self.inputs.compress_config.get_dict())
            with folder.open("input_compress_config.json", "w") as f:
                json.dump(payload, f, indent=2)

    def _get_retrieve_list(self) -> list[str]:
        """Get list of files to retrieve after calculation."""
        retrieve_list = super()._get_retrieve_list()
        # Spectral function doesn't typically output MPS
        if 'output_mps.npz' in retrieve_list:
            retrieve_list.remove('output_mps.npz')
        return retrieve_list
