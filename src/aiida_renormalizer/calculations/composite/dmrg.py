"""CalcJob for DMRG ground state optimization."""
from __future__ import annotations

import os
import tempfile

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import ModelData, MpsData, MpoData


class DMRGCalcJob(RenoBaseCalcJob):
    """DMRG variational optimization for ground state.

    Corresponds to Reno API: optimize_mps(mps, mpo, optimize_config)

    Inputs:
        model: ModelData - System definition
        mpo: MpoData - Hamiltonian operator
        initial_mps: MpsData (optional) - Initial guess, random if not provided
        config: ConfigData - OptimizeConfig with convergence criteria

    Outputs:
        output_mps: MpsData - Optimized ground state
        output_parameters: Dict - Energy trajectory, convergence info
    """

    _template_name = "dmrg_driver.py.jinja"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        # Additional inputs
        spec.input(
            "mpo",
            valid_type=MpoData,
            help="Hamiltonian MPO for optimization",
        )
        spec.input(
            "initial_mps",
            valid_type=MpsData,
            required=False,
            help="Initial MPS guess (random if not provided)",
        )
        spec.input(
            "omega",
            valid_type=orm.Float,
            required=False,
            help="Target eigenvalue near omega (for excited states)",
        )

        # Outputs
        spec.output(
            "output_mps",
            valid_type=MpsData,
            help="Optimized ground state MPS",
        )

        # Exit codes
        spec.exit_code(
            300,
            "ERROR_NOT_CONVERGED",
            message="DMRG optimization did not converge",
        )

    def _get_template_context(self) -> dict:
        """Provide context for Jinja2 template rendering."""
        context = super()._get_template_context()
        context["has_initial_mps"] = "initial_mps" in self.inputs
        context["has_omega"] = "omega" in self.inputs
        return context

    def _write_input_files(self, folder) -> None:
        """Write input files for DMRG calculation."""
        import json

        super()._write_input_files(folder)

        # Write MPO
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

        # Write initial MPS (if provided)
        if "initial_mps" in self.inputs:
            mps_data = self.inputs.initial_mps
            mps = mps_data.load_mps(model_data)

            with tempfile.TemporaryDirectory() as tmpdir:
                mps_path = os.path.join(tmpdir, "mps")
                mps.dump(mps_path)
                actual = mps_path + ".npz" if os.path.exists(mps_path + ".npz") else mps_path
                with open(actual, "rb") as src:
                    with folder.open("initial_mps.npz", "wb") as dst:
                        dst.write(src.read())

        # Write omega (if provided)
        if "omega" in self.inputs:
            with folder.open("input_omega.json", "w") as f:
                json.dump({"omega": self.inputs.omega.value}, f)
