"""CalcJob for imaginary time evolution."""
from __future__ import annotations

import os
import tempfile

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import ModelData, MPSData, MPOData


class ImagTimeCalcJob(RenoBaseCalcJob):
    """Imaginary time evolution for ground state.

    Corresponds to Reno API: mps.evolve(mpo, -1j * beta) with EvolveConfig

    Inputs:
        model: ModelData - System definition
        mpo: MPOData - Hamiltonian operator
        initial_mps: MPSData (optional) - Initial state, random if not provided
        config: ConfigData - EvolveConfig with imaginary time parameters
        beta: Float - Total imaginary time (inverse temperature)

    Outputs:
        output_mps: MPSData - Ground state approximation
        output_parameters: Dict - Energy, convergence info
    """

    _template_name = "imag_time_driver.py.jinja"

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
            "initial_mps",
            valid_type=MPSData,
            required=False,
            help="Initial MPS (random if not provided)",
        )
        spec.input(
            "beta",
            valid_type=orm.Float,
            help="Total imaginary time (inverse temperature)",
        )
        spec.input(
            "dt",
            valid_type=orm.Float,
            required=False,
            help="Time step (auto-determined if not provided)",
        )

        # Outputs
        spec.output(
            "output_mps",
            valid_type=MPSData,
            help="Ground state from imaginary time evolution",
        )

        # Exit codes
        spec.exit_code(
            300,
            "ERROR_NOT_CONVERGED",
            message="Imaginary time evolution did not converge",
        )

    def _get_template_context(self) -> dict:
        """Provide context for Jinja2 template rendering."""
        context = super()._get_template_context()
        context["has_initial_mps"] = "initial_mps" in self.inputs
        context["has_dt"] = "dt" in self.inputs
        return context

    def _write_input_files(self, folder) -> None:
        """Write input files for imaginary time evolution."""
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

        # Write initial MPS (if provided)
        if "initial_mps" in self.inputs:
            mps_data = self.inputs.initial_mps
            MPS = mps_data.load_mps(model_data)

            with tempfile.TemporaryDirectory() as tmpdir:
                mps_path = os.path.join(tmpdir, "mps")
                MPS.dump(mps_path)
                actual = mps_path + ".npz" if os.path.exists(mps_path + ".npz") else mps_path
                with open(actual, "rb") as src:
                    with folder.open("initial_mps.npz", "wb") as dst:
                        dst.write(src.read())

        # Write evolution parameters
        params = {
            "beta": self.inputs.beta.value,
        }
        if "dt" in self.inputs:
            params["dt"] = self.inputs.dt.value

        with folder.open("input_evolution_params.json", "w") as f:
            json.dump(params, f, indent=2)
