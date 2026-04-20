"""CalcJob for bath spectral density discretization."""
from __future__ import annotations

import json

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob


class BathDiscretizationCalcJob(RenoBaseCalcJob):
    """Discretize continuous bath spectral density J(omega) into modes."""

    _template_name = "bath_discretization_driver.py.jinja"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        spec.input(
            "omega_grid",
            valid_type=orm.ArrayData,
            help="Frequency grid with array name 'omega_grid'",
        )
        spec.input(
            "j_omega",
            valid_type=orm.ArrayData,
            help="J(omega) values with array name 'j_omega'",
        )
        spec.input(
            "n_modes",
            valid_type=orm.Int,
            default=lambda: orm.Int(16),
            help="Number of discrete bath modes",
        )
        spec.input(
            "method",
            valid_type=orm.Str,
            default=lambda: orm.Str("trapz"),
            help="Discretization method: trapz | wang1_like | equal_area",
        )

        spec.output(
            "output_parameters",
            valid_type=orm.Dict,
            help="Discretized omega_k, c_j2, displacement, and metadata",
        )

        spec.exit_code(
            321,
            "ERROR_DISCRETIZATION_FAILED",
            message="Bath discretization failed",
        )

    def _get_template_context(self) -> dict:
        context = super()._get_template_context()
        context["n_modes"] = self.inputs.n_modes.value
        context["method"] = self.inputs.method.value
        return context

    def _write_input_files(self, folder) -> None:
        super()._write_input_files(folder)

        omega = self.inputs.omega_grid.get_array("omega_grid")
        j_omega = self.inputs.j_omega.get_array("j_omega")

        with folder.open("input_omega_grid.npy", "wb") as f:
            import numpy as np
            np.save(f, omega)
        with folder.open("input_j_omega.npy", "wb") as f:
            import numpy as np
            np.save(f, j_omega)

        params = {
            "n_modes": self.inputs.n_modes.value,
            "method": self.inputs.method.value,
        }
        with folder.open("input_bath_discretization.json", "w") as f:
            json.dump(params, f, indent=2)

    def _get_retrieve_list(self) -> list[str]:
        retrieve_list = super()._get_retrieve_list()
        if "output_mps.npz" in retrieve_list:
            retrieve_list.remove("output_mps.npz")
        return retrieve_list

