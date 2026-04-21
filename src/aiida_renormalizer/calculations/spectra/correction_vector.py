"""CalcJob for correction vector frequency-domain spectra."""
from __future__ import annotations

import os
import tempfile

import numpy as np

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import ModelData, MPSData, MPOData


class CorrectionVectorCalcJob(RenoBaseCalcJob):
    """Correction vector calculation for frequency-domain spectra.

    Corresponds to Reno API: SpectraZtCV

    Uses DDMRG (density matrix renormalization group for dynamical properties)
    to compute frequency-resolved spectra without time propagation.

    Inputs:
        model: ModelData - System definition
        mpo: MPOData - Hamiltonian operator (optional)
        initial_mps: MPSData - Ground state MPS (optional, will be computed if not provided)
        spectratype: Str - "abs", "emi", or None
        frequencies: ArrayData - List of frequencies to compute
        eta: Float - Broadening parameter
        m_max: Int - Maximum bond dimension
        method: Str - "1site" or "2site" optimization
        procedure_cv: List - Optimization procedure for CV
        rtol: Float - Relative tolerance for convergence

    Outputs:
        output_parameters: Dict - Spectrum at each frequency
    """

    _template_name = "correction_vector_driver.py.jinja"

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
            help="Ground state MPS (will be optimized if not provided)",
        )
        spec.input(
            "spectratype",
            valid_type=orm.Str,
            required=False,
            help="Spectrum type: 'abs', 'emi', or None",
        )
        spec.input(
            "frequencies",
            valid_type=orm.ArrayData,
            help="Array of frequencies to compute spectrum",
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
            "procedure_cv",
            valid_type=orm.List,
            required=False,
            help="Custom optimization procedure for CV",
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

        # Outputs
        spec.output(
            "output_parameters",
            valid_type=orm.Dict,
            help="Spectrum values at each frequency",
        )

        # Exit codes
        spec.exit_code(
            300,
            "ERROR_NOT_CONVERGED",
            message="Correction vector optimization did not converge",
        )

    def _get_template_context(self) -> dict:
        """Provide context for Jinja2 template rendering."""
        context = super()._get_template_context()
        context["has_mpo"] = "mpo" in self.inputs
        context["has_initial_mps"] = "initial_mps" in self.inputs
        context["has_spectratype"] = "spectratype" in self.inputs
        context["has_procedure_cv"] = "procedure_cv" in self.inputs
        context["eta"] = self.inputs.eta.value
        context["m_max"] = self.inputs.m_max.value
        context["method"] = self.inputs.method.value
        context["rtol"] = self.inputs.rtol.value
        context["n_cores"] = self.inputs.n_cores.value
        return context

    def _write_input_files(self, folder) -> None:
        """Write input files for correction vector calculation."""
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

        # Write frequencies
        freqs = self.inputs.frequencies.get_array("frequencies")
        with folder.open("frequencies.npy", "wb") as f:
            np.save(f, freqs)

        # Write parameters
        params = {
            "eta": self.inputs.eta.value,
            "m_max": self.inputs.m_max.value,
            "method": self.inputs.method.value,
            "rtol": self.inputs.rtol.value,
            "n_cores": self.inputs.n_cores.value,
        }
        if "spectratype" in self.inputs:
            params["spectratype"] = self.inputs.spectratype.value

        with folder.open("input_cv_params.json", "w") as f:
            json.dump(params, f, indent=2)

        # Write custom procedure (if provided)
        if "procedure_cv" in self.inputs:
            with folder.open("input_procedure_cv.json", "w") as f:
                json.dump(self.inputs.procedure_cv.get_list(), f, indent=2)
