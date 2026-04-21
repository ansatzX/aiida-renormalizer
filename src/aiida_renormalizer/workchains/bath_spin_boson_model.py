"""WorkChain for building SBM bath models from Renormalizer spectral-density semantics."""
from __future__ import annotations

import numpy as np

from aiida import orm
from aiida.engine import WorkChain

from aiida_renormalizer.data import ModelData


class BathSpinBosonModelWorkChain(WorkChain):
    """Build a SpinBosonModel ModelData node from bath parameters.

    This is the high-level bridge from Renormalizer SBM helpers to AiiDA data:
    - `param2mollist` for the canonical ohmic/debye-style construction path
    - explicit `(omega_k, c_j2)` inputs for already-discretized bath modes

    The resulting `bath_model` node can be passed directly to `BuildMPOCalcJob`
    or any other consumer that expects `ModelData`.
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input(
            "construction",
            valid_type=orm.Str,
            default=lambda: orm.Str("param2mollist"),
            help="Construction mode: param2mollist | discrete",
        )
        spec.input("alpha", valid_type=orm.Float, required=False, help="Ohmic strength alpha")
        spec.input("raw_delta", valid_type=orm.Float, required=False, help="Bare tunneling / gap parameter")
        spec.input("omega_c", valid_type=orm.Float, required=False, help="Bath cutoff frequency")
        spec.input("renormalization_p", valid_type=orm.Float, required=False, help="Adiabatic renormalization p")
        spec.input("n_phonons", valid_type=orm.Int, required=False, help="Number of bath modes")
        spec.input("epsilon", valid_type=orm.Float, default=lambda: orm.Float(0.0), help="Spin bias epsilon")
        spec.input("dipole", valid_type=orm.Float, required=False, help="Optional dipole moment")
        spec.input("omega_k", valid_type=orm.ArrayData, required=False, help="Discrete bath frequencies")
        spec.input("c_j2", valid_type=orm.ArrayData, required=False, help="Discrete bath c_j^2 values")
        spec.input("spectral_density_type", valid_type=orm.Str, default=lambda: orm.Str("ohmic_exp"))
        spec.input("beta", valid_type=orm.Float, default=lambda: orm.Float(0.7), help="Reserved for compatibility")

        spec.output("bath_model", valid_type=ModelData, help="Constructed SpinBoson ModelData")
        spec.output("bath_modes", valid_type=orm.ArrayData, help="Bath mode frequencies and couplings")
        spec.output("output_parameters", valid_type=orm.Dict, help="Model construction summary")

        spec.exit_code(480, "ERROR_INVALID_BATH_MODEL_INPUT", message="Invalid bath-model inputs")

        spec.outline(cls.setup, cls.build_model, cls.finalize)

    def setup(self):
        self.report("Starting bath SpinBoson model construction")

        construction = self.inputs.construction.value
        if construction not in ("param2mollist", "discrete"):
            self.report(f"Invalid construction mode: {construction}")
            return self.exit_codes.ERROR_INVALID_BATH_MODEL_INPUT

        if construction == "param2mollist":
            required = ["alpha", "raw_delta", "omega_c", "renormalization_p", "n_phonons"]
            missing = [name for name in required if name not in self.inputs]
            if missing:
                self.report(f"Missing param2mollist inputs: {missing}")
                return self.exit_codes.ERROR_INVALID_BATH_MODEL_INPUT
        else:
            if "omega_k" not in self.inputs or "c_j2" not in self.inputs:
                self.report("Discrete construction requires omega_k and c_j2")
                return self.exit_codes.ERROR_INVALID_BATH_MODEL_INPUT

    def build_model(self):
        """Construct the Renormalizer model in-memory."""
        from renormalizer.model import Phonon, SpinBosonModel
        from renormalizer.sbm import param2mollist
        from renormalizer.utils import Quantity

        construction = self.inputs.construction.value

        if construction == "param2mollist":
            model = param2mollist(
                self.inputs.alpha.value,
                Quantity(self.inputs.raw_delta.value),
                Quantity(self.inputs.omega_c.value),
                self.inputs.renormalization_p.value,
                self.inputs.n_phonons.value,
            )
            ph_list = list(model.ph_list)
            coeff_data = {
                "construction": "param2mollist",
                "spectral_density_type": self.inputs.spectral_density_type.value,
                "alpha": self.inputs.alpha.value,
                "raw_delta": self.inputs.raw_delta.value,
                "omega_c": self.inputs.omega_c.value,
                "renormalization_p": self.inputs.renormalization_p.value,
                "n_phonons": self.inputs.n_phonons.value,
            }
        else:
            omega_k = self.inputs.omega_k.get_array("omega_k")
            c_j2 = self.inputs.c_j2.get_array("c_j2")
            if omega_k.shape != c_j2.shape:
                self.report("omega_k and c_j2 must have the same shape")
                return self.exit_codes.ERROR_INVALID_BATH_MODEL_INPUT

            ph_list = []
            for omega, c2 in zip(omega_k, c_j2):
                if omega <= 0:
                    self.report("Discrete bath frequencies must be positive")
                    return self.exit_codes.ERROR_INVALID_BATH_MODEL_INPUT
                displacement = np.sqrt(max(c2, 0.0)) / (omega ** 2)
                ph_list.append(Phonon.simplest_phonon(Quantity(omega), Quantity(displacement), lam=False))

            dipole = self.inputs.dipole.value if "dipole" in self.inputs else None
            model = SpinBosonModel(Quantity(self.inputs.epsilon.value), Quantity(0.0), ph_list, dipole=dipole)
            coeff_data = {
                "construction": "discrete",
                "spectral_density_type": self.inputs.spectral_density_type.value,
                "omega_k": omega_k.tolist(),
                "c_j2": c_j2.tolist(),
                "n_phonons": len(ph_list),
            }

        self.ctx.model = model
        self.ctx.ph_list = ph_list
        self.ctx.coeff_data = coeff_data

    def finalize(self):
        """Store and output the constructed model."""
        from aiida_renormalizer.data import ModelData

        model_data = ModelData.from_model(self.ctx.model)
        model_data.store()

        bath_modes = orm.ArrayData()
        omega_k = []
        c_j2 = []
        displacement = []
        for ph in self.ctx.ph_list:
            omega = float(ph.omega[0])
            disp = float(ph.dis[1])
            omega_k.append(omega)
            displacement.append(disp)
            c_j2.append(float((disp * omega ** 2) ** 2))

        bath_modes.set_array("omega_k", np.array(omega_k, dtype=float))
        bath_modes.set_array("c_j2", np.array(c_j2, dtype=float))
        bath_modes.set_array("displacement", np.array(displacement, dtype=float))
        coeff_data = dict(self.ctx.coeff_data)
        coeff_data["omega_k"] = omega_k
        coeff_data["c_j2"] = c_j2
        coeff_data["displacement"] = displacement

        self.out("bath_model", model_data)
        self.out("bath_modes", bath_modes)
        self.out("output_parameters", orm.Dict(dict=coeff_data))

        self.report("Bath SpinBoson model construction completed")
