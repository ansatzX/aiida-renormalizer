"""CalcJob for finite-temperature state preparation."""
from __future__ import annotations

import os
import tempfile

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import ModelData, MPSData, MPOData


class ThermalPropCalcJob(RenoBaseCalcJob):
    """Finite-temperature state preparation via imaginary time propagation.

    Corresponds to Reno API: MpDm.thermal_prop()
    Prepares thermal density matrix at finite temperature.

    Inputs:
        model: ModelData - System definition
        mpo: MPOData - Hamiltonian operator
        temperature: Float - Target temperature
        config: ConfigData - EvolveConfig parameters

    Outputs:
        output_mps: MPSData (actually MpDm) - Thermal density matrix
        output_parameters: Dict - Temperature, free energy, etc.
    """

    _template_name = "thermal_prop_driver.py.jinja"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        # Additional inputs
        spec.input(
            "mpo",
            valid_type=MPOData,
            help="Hamiltonian MPO",
        )
        spec.input(
            "temperature",
            valid_type=orm.Float,
            help="Target temperature",
        )
        spec.input(
            "n_iterations",
            valid_type=orm.Int,
            required=False,
            default=lambda: orm.Int(10),
            help="Number of thermal propagation iterations",
        )

        # Input namespace for metadata
        spec.input(
            "is_mpdm",
            valid_type=orm.Bool,
            required=False,
            default=lambda: orm.Bool(True),
            help="Mark output as MpDm (density matrix)",
        )

        # Outputs
        spec.output(
            "output_mps",
            valid_type=MPSData,
            help="Thermal density matrix (MpDm)",
        )

        # Exit codes
        spec.exit_code(
            300,
            "ERROR_NOT_CONVERGED",
            message="Thermal propagation did not converge",
        )

    def _get_template_context(self) -> dict:
        """Provide context for Jinja2 template rendering."""
        context = super()._get_template_context()
        context["n_iterations"] = self.inputs.n_iterations.value
        return context

    def _write_input_files(self, folder) -> None:
        """Write input files for thermal propagation."""
        import json

        super()._write_input_files(folder)

        # Write MPO
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

        # Write thermal parameters
        params = {
            "temperature": self.inputs.temperature.value,
            "n_iterations": self.inputs.n_iterations.value,
        }

        with folder.open("input_thermal_params.json", "w") as f:
            json.dump(params, f, indent=2)
