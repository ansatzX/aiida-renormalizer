"""CalcJob for bath spectral density generation/normalization."""
from __future__ import annotations

import json

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob


class BathSpectralDensityCalcJob(RenoBaseCalcJob):
    """Generate bath spectral density data J(omega).

    This job supports:
    - Analytic SDF forms used in Renormalizer SBM helpers
    - User-provided tabulated spectrum (omega_grid + j_omega)
    """

    _template_name = "bath_spectral_density_driver.py.jinja"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        spec.input(
            "spectral_density_type",
            valid_type=orm.Str,
            default=lambda: orm.Str("ohmic_exp"),
            help="SDF type: ohmic_exp | debye | cole_davidson | custom",
        )
        spec.input(
            "omega_min",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.0),
            help="Minimum frequency for generated grid",
        )
        spec.input(
            "omega_max",
            valid_type=orm.Float,
            default=lambda: orm.Float(10.0),
            help="Maximum frequency for generated grid",
        )
        spec.input(
            "num_points",
            valid_type=orm.Int,
            default=lambda: orm.Int(512),
            help="Number of frequency points for generated grid",
        )
        spec.input(
            "alpha",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.05),
            help="Ohmic/Cole-Davidson strength parameter",
        )
        spec.input(
            "s_exponent",
            valid_type=orm.Float,
            default=lambda: orm.Float(1.0),
            help="Ohmic exponent s in omega^s",
        )
        spec.input(
            "cutoff",
            valid_type=orm.Float,
            default=lambda: orm.Float(1.0),
            help="Cutoff frequency (omega_c)",
        )
        spec.input(
            "lambda_reorg",
            valid_type=orm.Float,
            default=lambda: orm.Float(1.0),
            help="Reorganization parameter (Debye lambda or Cole-Davidson eta)",
        )
        spec.input(
            "beta",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.7),
            help="Cole-Davidson beta exponent",
        )
        spec.input(
            "custom_omega",
            valid_type=orm.ArrayData,
            required=False,
            help="Custom frequency grid with array name 'omega_grid'",
        )
        spec.input(
            "custom_j_omega",
            valid_type=orm.ArrayData,
            required=False,
            help="Custom J(omega) values with array name 'j_omega'",
        )

        spec.output(
            "output_parameters",
            valid_type=orm.Dict,
            help="Generated/validated omega_grid and j_omega data",
        )

        spec.exit_code(
            320,
            "ERROR_INVALID_SPECTRUM_INPUT",
            message="Invalid spectral density input or parameters",
        )

    def _get_template_context(self) -> dict:
        context = super()._get_template_context()
        context["spectral_density_type"] = self.inputs.spectral_density_type.value
        context["omega_min"] = self.inputs.omega_min.value
        context["omega_max"] = self.inputs.omega_max.value
        context["num_points"] = self.inputs.num_points.value
        context["alpha"] = self.inputs.alpha.value
        context["s_exponent"] = self.inputs.s_exponent.value
        context["cutoff"] = self.inputs.cutoff.value
        context["lambda_reorg"] = self.inputs.lambda_reorg.value
        context["beta"] = self.inputs.beta.value
        context["has_custom_spectrum"] = "custom_omega" in self.inputs and "custom_j_omega" in self.inputs
        return context

    def _write_input_files(self, folder) -> None:
        super()._write_input_files(folder)

        params = {
            "spectral_density_type": self.inputs.spectral_density_type.value,
            "omega_min": self.inputs.omega_min.value,
            "omega_max": self.inputs.omega_max.value,
            "num_points": self.inputs.num_points.value,
            "alpha": self.inputs.alpha.value,
            "s_exponent": self.inputs.s_exponent.value,
            "cutoff": self.inputs.cutoff.value,
            "lambda_reorg": self.inputs.lambda_reorg.value,
            "beta": self.inputs.beta.value,
        }
        with folder.open("input_bath_spectral_density.json", "w") as f:
            json.dump(params, f, indent=2)

        if "custom_omega" in self.inputs and "custom_j_omega" in self.inputs:
            omega = self.inputs.custom_omega.get_array("omega_grid")
            j_omega = self.inputs.custom_j_omega.get_array("j_omega")

            with folder.open("custom_omega_grid.npy", "wb") as f:
                import numpy as np
                np.save(f, omega)
            with folder.open("custom_j_omega.npy", "wb") as f:
                import numpy as np
                np.save(f, j_omega)

    def _get_retrieve_list(self) -> list[str]:
        retrieve_list = super()._get_retrieve_list()
        if "output_mps.npz" in retrieve_list:
            retrieve_list.remove("output_mps.npz")
        return retrieve_list

