"""CalcJob for mapping discretized bath modes to MPO-ready coefficients."""
from __future__ import annotations

import json

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob


class BathToMPOCoeffCalcJob(RenoBaseCalcJob):
    """Map bath mode parameters to MPO coefficient dictionaries.

    The mapping follows Renormalizer SBM conventions around:
    - mode frequencies omega_k
    - c_j^2 quantities
    - displacements d_j = sqrt(c_j^2) / omega_k^2
    """

    _template_name = "bath_to_mpo_coeff_driver.py.jinja"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        spec.input(
            "omega_k",
            valid_type=orm.ArrayData,
            help="Discretized frequencies with array name 'omega_k'",
        )
        spec.input(
            "c_j2",
            valid_type=orm.ArrayData,
            help="Discretized c_j^2 values with array name 'c_j2'",
        )
        spec.input(
            "frequency_scale",
            valid_type=orm.Float,
            default=lambda: orm.Float(1.0),
            help="Global scale applied to frequencies",
        )
        spec.input(
            "coupling_scale",
            valid_type=orm.Float,
            default=lambda: orm.Float(1.0),
            help="Global scale applied to c_j^2",
        )

        spec.output(
            "output_parameters",
            valid_type=orm.Dict,
            help="MPO-ready bath coefficients and summary statistics",
        )

        spec.exit_code(
            322,
            "ERROR_COEFF_MAPPING_FAILED",
            message="Bath-to-MPO coefficient mapping failed",
        )

    def _get_template_context(self) -> dict:
        context = super()._get_template_context()
        context["frequency_scale"] = self.inputs.frequency_scale.value
        context["coupling_scale"] = self.inputs.coupling_scale.value
        return context

    def _write_input_files(self, folder) -> None:
        super()._write_input_files(folder)

        omega_k = self.inputs.omega_k.get_array("omega_k")
        c_j2 = self.inputs.c_j2.get_array("c_j2")

        with folder.open("input_omega_k.npy", "wb") as f:
            import numpy as np
            np.save(f, omega_k)
        with folder.open("input_c_j2.npy", "wb") as f:
            import numpy as np
            np.save(f, c_j2)

        params = {
            "frequency_scale": self.inputs.frequency_scale.value,
            "coupling_scale": self.inputs.coupling_scale.value,
        }
        with folder.open("input_bath_to_mpo_coeff.json", "w") as f:
            json.dump(params, f, indent=2)

    def _get_retrieve_list(self) -> list[str]:
        retrieve_list = super()._get_retrieve_list()
        if "output_mps.npz" in retrieve_list:
            retrieve_list.remove("output_mps.npz")
        return retrieve_list

