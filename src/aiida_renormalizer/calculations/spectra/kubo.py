"""CalcJob for Kubo linear response transport."""
from __future__ import annotations

import os
import tempfile

import numpy as np

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import ModelData, MpsData, MpoData


class KuboCalcJob(RenoBaseCalcJob):
    """Kubo formula transport calculation for mobility.

    Corresponds to Reno API: TransportKubo

    Calculates mobility via Green-Kubo formula:
        μ = (1/kT) ∫dt <j(t)j(0)>

    Inputs:
        model: ModelData - System definition
        mpo: MpoData - Hamiltonian operator (optional)
        initial_mps: MpsData - Thermal density matrix if precomputed
        temperature: Float - Temperature in atomic units
        distance_matrix: ArrayData - Distance matrix D_ij = P_i - P_j
        insteps: Int - Steps for imaginary time propagation
        config: ConfigData - EvolveConfig for real time propagation
        ievolve_config: ConfigData - EvolveConfig for imaginary time propagation
        compress_config: ConfigData - CompressConfig

    Outputs:
        output_parameters: Dict - Autocorrelation function, mobility, time series
    """

    _template_name = "kubo_driver.py.jinja"

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
            help="Pre-computed thermal density matrix (MpDm)",
        )
        spec.input(
            "temperature",
            valid_type=orm.Float,
            help="Temperature in atomic units (must be > 0)",
        )
        spec.input(
            "distance_matrix",
            valid_type=orm.ArrayData,
            required=False,
            help="Distance matrix D_ij = P_i - P_j for current operator",
        )
        spec.input(
            "insteps",
            valid_type=orm.Int,
            default=lambda: orm.Int(1),
            help="Steps for imaginary time propagation",
        )
        spec.input_namespace(
            "ievolve_config",
            valid_type=orm.Dict,
            required=False,
            help="Config for imaginary time propagation",
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
            help="Autocorrelation function, mobility, time series",
        )

        # Exit codes
        spec.exit_code(
            300,
            "ERROR_ZERO_TEMPERATURE",
            message="Temperature must be non-zero for Kubo calculation",
        )
        spec.exit_code(
            301,
            "ERROR_THERMAL_PROP_FAILED",
            message="Thermal propagation failed",
        )
        spec.exit_code(
            302,
            "ERROR_EVOLUTION_FAILED",
            message="Real time evolution failed",
        )

    def _get_template_context(self) -> dict:
        """Provide context for Jinja2 template rendering."""
        context = super()._get_template_context()
        context["has_mpo"] = "mpo" in self.inputs
        context["has_initial_mps"] = "initial_mps" in self.inputs
        context["has_distance_matrix"] = "distance_matrix" in self.inputs
        context["has_ievolve_config"] = "ievolve_config" in self.inputs
        context["has_compress_config"] = "compress_config" in self.inputs
        context["temperature"] = self.inputs.temperature.value
        context["insteps"] = self.inputs.insteps.value
        return context

    def _write_input_files(self, folder) -> None:
        """Write input files for Kubo calculation."""
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

        # Write distance matrix (if provided)
        if "distance_matrix" in self.inputs:
            dist_matrix = self.inputs.distance_matrix.get_array("matrix")
            np.save(folder.open("distance_matrix.npy", "wb"), dist_matrix)

        # Write parameters
        params = {
            "temperature": self.inputs.temperature.value,
            "insteps": self.inputs.insteps.value,
        }

        with folder.open("input_kubo_params.json", "w") as f:
            json.dump(params, f, indent=2)

        # Write configs (if provided)
        if "ievolve_config" in self.inputs:
            with folder.open("input_ievolve_config.json", "w") as f:
                json.dump(self.inputs.ievolve_config.get_dict(), f, indent=2)

        if "compress_config" in self.inputs:
            with folder.open("input_compress_config.json", "w") as f:
                json.dump(self.inputs.compress_config.get_dict(), f, indent=2)
