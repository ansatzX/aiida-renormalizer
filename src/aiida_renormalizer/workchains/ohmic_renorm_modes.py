"""WorkChain for spectral-function-native mode generation."""
from __future__ import annotations

import numpy as np
from aiida import orm
from aiida.engine import ToContext, WorkChain

from aiida_renormalizer.calculations.bath import OhmicRenormModesCalcJob


class OhmicRenormModesWorkChain(WorkChain):
    """Expose the ohmic renorm-modes handler via CalcJob backend."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("code", valid_type=orm.AbstractCode)
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

        spec.output("bath_modes", valid_type=orm.ArrayData)
        spec.output("output_parameters", valid_type=orm.Dict)

        spec.exit_code(550, "ERROR_INVALID_INPUT", message="Invalid spectral/mode inputs")
        spec.exit_code(551, "ERROR_CALCJOB_FAILED", message="Ohmic renorm modes CalcJob failed")

        spec.outline(cls.setup, cls.run_calcjob, cls.inspect_calcjob, cls.finalize)

    def setup(self):
        if self.inputs.n_modes.value <= 0:
            self.report("n_modes must be > 0")
            return self.exit_codes.ERROR_INVALID_INPUT
        if self.inputs.omega_c.value <= 0:
            self.report("omega_c must be > 0")
            return self.exit_codes.ERROR_INVALID_INPUT
        if self.inputs.raw_delta.value <= 0:
            self.report("raw_delta must be > 0")
            return self.exit_codes.ERROR_INVALID_INPUT
        if self.inputs.renormalization_p.value <= 0:
            self.report("renormalization_p must be > 0")
            return self.exit_codes.ERROR_INVALID_INPUT
        if self.inputs.spectral_density_type.value != "ohmic_exp":
            self.report("Only spectral_density_type='ohmic_exp' is supported in this high-level path")
            return self.exit_codes.ERROR_INVALID_INPUT
        if self.inputs.ohmic_discretization.value not in {"wang1", "trapz"}:
            self.report("ohmic_discretization must be 'wang1' or 'trapz'")
            return self.exit_codes.ERROR_INVALID_INPUT

    def run_calcjob(self):
        inputs = {
            "code": self.inputs.code,
            "alpha": self.inputs.alpha,
            "s_exponent": self.inputs.s_exponent,
            "omega_c": self.inputs.omega_c,
            "spectral_density_type": self.inputs.spectral_density_type,
            "ohmic_discretization": self.inputs.ohmic_discretization,
            "raw_delta": self.inputs.raw_delta,
            "renormalization_p": self.inputs.renormalization_p,
            "n_modes": self.inputs.n_modes,
        }
        return ToContext(calcjob=self.submit(OhmicRenormModesCalcJob, **inputs))

    def inspect_calcjob(self):
        calc = self.ctx.calcjob
        if not calc.is_finished_ok:
            self.report(f"OhmicRenormModesCalcJob failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_CALCJOB_FAILED
        self.ctx.params = calc.outputs.output_parameters.get_dict()

    def finalize(self):
        params = self.ctx.params
        omega_k = np.asarray(params["omega_k"], dtype=float)
        c_j2 = np.asarray(params["c_j2"], dtype=float)

        modes = orm.ArrayData()
        modes.set_array("omega_k", omega_k)
        modes.set_array("c_j2", c_j2)

        self.out("bath_modes", modes)
        self.out("output_parameters", orm.Dict(dict=params))
