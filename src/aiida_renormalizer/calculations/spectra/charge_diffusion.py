"""CalcJob for charge diffusion dynamics."""
from __future__ import annotations

import os
import tempfile

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import ModelData, MPSData, MPOData


class ChargeDiffusionCalcJob(RenoBaseCalcJob):
    """Charge diffusion dynamics simulation.

    Corresponds to Reno API: ChargeDiffusionDynamics

    Simulates charge carrier diffusion and calculates mean square displacement,
    occupations, and optionally mobility.

    Inputs:
        model: ModelData - System definition (must be HolsteinModel)
        mpo: MPOData - Hamiltonian operator (optional)
        initial_mps: MPSData - Initial thermal state (optional)
        temperature: Float - Temperature in atomic units
        init_electron: Str - "fc" (Franck-Condon) or "relaxed"
        stop_at_edge: Bool - Stop when charge reaches system boundary
        rdm: Bool - Calculate reduced density matrix (expensive)
        config: ConfigData - EvolveConfig
        compress_config: ConfigData - CompressConfig

    Outputs:
        output_parameters: Dict - Time series, occupations, r_square, etc.
    """

    _template_name = "charge_diffusion_driver.py.jinja"

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
            help="Initial thermal state (will be computed if not provided)",
        )
        spec.input(
            "temperature",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.0),
            help="Temperature in atomic units",
        )
        spec.input(
            "init_electron",
            valid_type=orm.Str,
            default=lambda: orm.Str("relaxed"),
            help="Electron initialization: 'fc' or 'relaxed'",
        )
        spec.input(
            "stop_at_edge",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help="Stop when charge reaches system boundary",
        )
        spec.input(
            "rdm",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(False),
            help="Calculate reduced density matrix (expensive)",
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
            help="Time series, occupations, r_square, energies, etc.",
        )

        # Exit codes
        spec.exit_code(
            300,
            "ERROR_EVOLUTION_FAILED",
            message="Time evolution failed",
        )
        spec.exit_code(
            301,
            "ERROR_THERMAL_PROP_FAILED",
            message="Thermal propagation failed for finite temperature",
        )

    def _get_template_context(self) -> dict:
        """Provide context for Jinja2 template rendering."""
        context = super()._get_template_context()
        context["has_mpo"] = "mpo" in self.inputs
        context["has_initial_mps"] = "initial_mps" in self.inputs
        context["has_compress_config"] = "compress_config" in self.inputs
        context["temperature"] = self.inputs.temperature.value
        context["init_electron"] = self.inputs.init_electron.value
        context["stop_at_edge"] = self.inputs.stop_at_edge.value
        context["rdm"] = self.inputs.rdm.value
        return context

    def _write_input_files(self, folder) -> None:
        """Write input files for charge diffusion calculation."""
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

        # Write initial thermal state (if provided)
        if "initial_mps" in self.inputs:
            mps_data = self.inputs.initial_mps
            MPS = mps_data.load_mps(self.inputs.model)

            with tempfile.TemporaryDirectory() as tmpdir:
                mps_path = os.path.join(tmpdir, "mps")
                MPS.dump(mps_path)
                actual = mps_path + ".npz" if os.path.exists(mps_path + ".npz") else mps_path
                with open(actual, "rb") as src:
                    with folder.open("initial_thermal_state.npz", "wb") as dst:
                        dst.write(src.read())

        # Write parameters
        params = {
            "temperature": self.inputs.temperature.value,
            "init_electron": self.inputs.init_electron.value,
            "stop_at_edge": self.inputs.stop_at_edge.value,
            "rdm": self.inputs.rdm.value,
        }

        with folder.open("input_diffusion_params.json", "w") as f:
            json.dump(params, f, indent=2)

        # Write compress config (if provided)
        if "compress_config" in self.inputs:
            payload = self._sanitize_compress_config_payload(self.inputs.compress_config.get_dict())
            with folder.open("input_compress_config.json", "w") as f:
                json.dump(payload, f, indent=2)

    def _get_retrieve_list(self) -> list[str]:
        """Get list of files to retrieve after calculation."""
        retrieve_list = super()._get_retrieve_list()
        # Charge diffusion doesn't typically output MPS
        if 'output_mps.npz' in retrieve_list:
            retrieve_list.remove('output_mps.npz')
        return retrieve_list
