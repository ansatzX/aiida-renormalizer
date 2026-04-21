"""CalcJob for spectral-handler mode generation (ohmic_exp embeds trapz/Wang1)."""
from __future__ import annotations

import json

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob


class OhmicRenormModesCalcJob(RenoBaseCalcJob):
    """Ohmic spectral handler step: adiabatic renormalization + trapz/Wang1."""

    _template_name = "ohmic_renorm_modes_driver.py.jinja"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        spec.input("alpha", valid_type=orm.Float)
        spec.input("s_exponent", valid_type=orm.Float, default=lambda: orm.Float(1.0))
        spec.input("omega_c", valid_type=orm.Float)
        spec.input(
            "spectral_density_type",
            valid_type=orm.Str,
            default=lambda: orm.Str("ohmic_exp"),
        )
        spec.input(
            "ohmic_discretization",
            valid_type=orm.Str,
            default=lambda: orm.Str("wang1"),
        )
        spec.input("raw_delta", valid_type=orm.Float)
        spec.input("renormalization_p", valid_type=orm.Float)
        spec.input("n_modes", valid_type=orm.Int)

        spec.output("output_parameters", valid_type=orm.Dict)

        spec.exit_code(
            323,
            "ERROR_SPECTRAL_MODES_FAILED",
            message="Ohmic renormalized modes generation failed",
        )

    def _get_template_context(self) -> dict:
        return super()._get_template_context()

    def _write_input_files(self, folder) -> None:
        super()._write_input_files(folder)

        params = {
            "alpha": self.inputs.alpha.value,
            "s_exponent": self.inputs.s_exponent.value,
            "omega_c": self.inputs.omega_c.value,
            "spectral_density_type": self.inputs.spectral_density_type.value,
            "ohmic_discretization": self.inputs.ohmic_discretization.value,
            "raw_delta": self.inputs.raw_delta.value,
            "renormalization_p": self.inputs.renormalization_p.value,
            "n_modes": self.inputs.n_modes.value,
        }
        with folder.open("input_ohmic_renorm_modes.json", "w") as f:
            json.dump(params, f, indent=2)

    def _get_retrieve_list(self) -> list[str]:
        retrieve_list = super()._get_retrieve_list()
        if "output_mps.npz" in retrieve_list:
            retrieve_list.remove("output_mps.npz")
        if "output_mpo.npz" in retrieve_list:
            retrieve_list.remove("output_mpo.npz")
        if "trajectory.npz" in retrieve_list:
            retrieve_list.remove("trajectory.npz")
        return retrieve_list
