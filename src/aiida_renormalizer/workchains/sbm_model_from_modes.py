"""WorkChain: bath modes + delta -> spin-boson ModelData."""
from __future__ import annotations

import numpy as np
from aiida import orm
from aiida.engine import WorkChain

from aiida_renormalizer.data import ModelData


class SbmModelFromModesWorkChain(WorkChain):
    """Build a spin-boson model from explicit bath modes and effective delta."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("bath_modes", valid_type=orm.ArrayData)
        spec.input("delta_eff", valid_type=orm.Float)
        spec.input("epsilon", valid_type=orm.Float, default=lambda: orm.Float(0.0))
        spec.input("symbol_map", valid_type=orm.Dict, default=lambda: orm.Dict(dict={}))

        spec.output("model", valid_type=ModelData)
        spec.output("output_parameters", valid_type=orm.Dict)

        spec.exit_code(560, "ERROR_INVALID_INPUT", message="Invalid bath mode arrays")

        spec.outline(cls.build_model, cls.finalize)

    def build_model(self):
        from renormalizer.model import Model as RenoModel
        from renormalizer.model import Phonon, SpinBosonModel
        from renormalizer.utils import Quantity

        from aiida_renormalizer.data.op import deserialize_opsum, serialize_opsum

        omega_k = self.inputs.bath_modes.get_array("omega_k")
        c_j2 = self.inputs.bath_modes.get_array("c_j2")

        if omega_k.shape != c_j2.shape or omega_k.ndim != 1:
            self.report("omega_k and c_j2 must be 1D arrays of identical shape")
            return self.exit_codes.ERROR_INVALID_INPUT

        ph_list = []
        disps = []
        for omega, c2 in zip(omega_k, c_j2):
            if omega <= 0:
                self.report("All omega_k values must be > 0")
                return self.exit_codes.ERROR_INVALID_INPUT
            disp = float(np.sqrt(max(float(c2), 0.0)) / (float(omega) ** 2))
            disps.append(disp)
            ph_list.append(
                Phonon.simplest_phonon(
                    Quantity(float(omega)),
                    Quantity(disp),
                    lam=False,
                )
            )

        model = SpinBosonModel(
            Quantity(self.inputs.epsilon.value),
            Quantity(self.inputs.delta_eff.value),
            ph_list,
        )
        symbol_map = {
            str(k): str(v)
            for k, v in self.inputs.symbol_map.get_dict().items()
            if str(k) and str(v)
        }
        if symbol_map:
            ham_data = serialize_opsum(model.ham_terms)
            for term in ham_data:
                old_symbol = str(term["symbol"])
                term["symbol"] = symbol_map.get(old_symbol, old_symbol)
            model = RenoModel(model.basis, deserialize_opsum(ham_data), dipole=model.dipole)

        self.ctx.model_data = ModelData.from_model(model)
        self.ctx.summary = {
            "n_modes": int(len(omega_k)),
            "delta_eff": float(self.inputs.delta_eff.value),
            "epsilon": float(self.inputs.epsilon.value),
            "omega_k": omega_k.tolist(),
            "c_j2": c_j2.tolist(),
            "displacement": disps,
            "symbol_map": symbol_map,
        }

    def finalize(self):
        self.out("model", self.ctx.model_data)
        self.out("output_parameters", orm.Dict(dict=self.ctx.summary))
