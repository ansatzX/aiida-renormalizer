"""CalcJob for constructing spin-boson bath coefficients from parameters or discrete modes."""

from __future__ import annotations

import json

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob


class BathSpinBosonModelCalcJob(RenoBaseCalcJob):
    """Build spin-boson bath mode payload for downstream symbolic-model construction."""

    _template_name = "bath_spin_boson_model_driver.py.jinja"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        spec.input(
            "construction",
            valid_type=orm.Str,
            default=lambda: orm.Str("param2mollist"),
            help="Construction mode: param2mollist | discrete",
        )
        spec.input("alpha", valid_type=orm.Float, required=False, help="Ohmic strength alpha")
        spec.input(
            "raw_delta", valid_type=orm.Float, required=False, help="Bare tunneling / gap parameter"
        )
        spec.input("omega_c", valid_type=orm.Float, required=False, help="Bath cutoff frequency")
        spec.input(
            "renormalization_p",
            valid_type=orm.Float,
            required=False,
            help="Adiabatic renormalization p",
        )
        spec.input("n_phonons", valid_type=orm.Int, required=False, help="Number of bath modes")
        spec.input(
            "epsilon",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.0),
            help="Spin bias epsilon",
        )
        spec.input(
            "delta_eff",
            valid_type=orm.Float,
            required=False,
            help="Effective delta for discrete mode path",
        )
        spec.input("dipole", valid_type=orm.Float, required=False, help="Optional dipole moment")
        spec.input(
            "omega_k", valid_type=orm.ArrayData, required=False, help="Discrete bath frequencies"
        )
        spec.input(
            "c_j2", valid_type=orm.ArrayData, required=False, help="Discrete bath c_j^2 values"
        )
        spec.input(
            "spectral_density_type", valid_type=orm.Str, default=lambda: orm.Str("ohmic_exp")
        )
        spec.input(
            "beta",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.7),
            help="Reserved for compatibility",
        )

        spec.output(
            "output_parameters",
            valid_type=orm.Dict,
            help="Mode coefficients and construction summary",
        )

        spec.exit_code(
            324,
            "ERROR_BATH_SPIN_BOSON_MODEL_FAILED",
            message="Bath spin-boson payload generation failed",
        )

    def _write_input_files(self, folder) -> None:
        super()._write_input_files(folder)

        params = {
            "construction": self.inputs.construction.value,
            "epsilon": self.inputs.epsilon.value,
            "spectral_density_type": self.inputs.spectral_density_type.value,
            "beta": self.inputs.beta.value,
        }
        optional_scalars = (
            "alpha",
            "raw_delta",
            "omega_c",
            "renormalization_p",
            "n_phonons",
            "delta_eff",
            "dipole",
        )
        for key in optional_scalars:
            if key in self.inputs:
                params[key] = self.inputs[key].value

        with folder.open("input_bath_spin_boson_model.json", "w") as f:
            json.dump(params, f, indent=2)

        if "omega_k" in self.inputs:
            omega_k = self.inputs.omega_k.get_array("omega_k")
            with folder.open("input_omega_k.npy", "wb") as f:
                import numpy as np

                np.save(f, omega_k)

        if "c_j2" in self.inputs:
            c_j2 = self.inputs.c_j2.get_array("c_j2")
            with folder.open("input_c_j2.npy", "wb") as f:
                import numpy as np

                np.save(f, c_j2)

    def _get_retrieve_list(self) -> list[str]:
        retrieve_list = super()._get_retrieve_list()
        for artifact in ("output_mps.npz", "output_mpo.npz", "trajectory.npz"):
            if artifact in retrieve_list:
                retrieve_list.remove(artifact)
        return retrieve_list
