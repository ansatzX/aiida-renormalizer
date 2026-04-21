"""WorkChain: spectral parameters -> discretized SBM bath modes."""
from __future__ import annotations

import numpy as np
from aiida import orm
from aiida.engine import WorkChain


class SbmSpectralModesWorkChain(WorkChain):
    """Discretize a spectral density into bath modes and compute renormalized delta."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("alpha", valid_type=orm.Float)
        spec.input("s_exponent", valid_type=orm.Float, default=lambda: orm.Float(1.0))
        spec.input("omega_c", valid_type=orm.Float)
        spec.input("raw_delta", valid_type=orm.Float)
        spec.input("renormalization_p", valid_type=orm.Float)
        spec.input("n_modes", valid_type=orm.Int)
        spec.input("discretization", valid_type=orm.Str, default=lambda: orm.Str("trapz"))

        spec.output("bath_modes", valid_type=orm.ArrayData)
        spec.output("output_parameters", valid_type=orm.Dict)

        spec.exit_code(550, "ERROR_INVALID_INPUT", message="Invalid spectral/mode inputs")

        spec.outline(cls.setup, cls.discretize, cls.finalize)

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
        if self.inputs.discretization.value not in {"trapz", "wang1"}:
            self.report("discretization must be 'trapz' or 'wang1'")
            return self.exit_codes.ERROR_INVALID_INPUT

    def discretize(self):
        from renormalizer.sbm import SpectralDensityFunction
        from renormalizer.utils import Quantity

        sdf = SpectralDensityFunction(
            alpha=self.inputs.alpha.value,
            omega_c=Quantity(self.inputs.omega_c.value),
            s=self.inputs.s_exponent.value,
        )
        delta_eff, omega_l = sdf.adiabatic_renormalization(
            Quantity(self.inputs.raw_delta.value),
            self.inputs.renormalization_p.value,
        )

        if self.inputs.discretization.value == "trapz":
            omega_k, c_j2 = sdf.trapz(self.inputs.n_modes.value, 0.0, omega_l)
        else:
            omega_k, c_j2 = sdf.Wang1(self.inputs.n_modes.value)

        self.ctx.delta_eff = float(delta_eff)
        self.ctx.omega_l = float(omega_l)
        self.ctx.omega_k = np.asarray(omega_k, dtype=float)
        self.ctx.c_j2 = np.asarray(c_j2, dtype=float)

    def finalize(self):
        modes = orm.ArrayData()
        modes.set_array("omega_k", self.ctx.omega_k)
        modes.set_array("c_j2", self.ctx.c_j2)

        params = {
            "alpha": self.inputs.alpha.value,
            "s_exponent": self.inputs.s_exponent.value,
            "omega_c": self.inputs.omega_c.value,
            "raw_delta": self.inputs.raw_delta.value,
            "delta_eff": self.ctx.delta_eff,
            "renormalization_p": self.inputs.renormalization_p.value,
            "renormalization_factor": self.ctx.delta_eff / self.inputs.raw_delta.value,
            "high_frequency_cutoff": self.ctx.omega_l,
            "n_modes": int(self.inputs.n_modes.value),
            "discretization": self.inputs.discretization.value,
        }
        self.out("bath_modes", modes)
        self.out("output_parameters", orm.Dict(dict=params))
