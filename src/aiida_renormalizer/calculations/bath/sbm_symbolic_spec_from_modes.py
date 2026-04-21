"""CalcJob for symbolic SBM basis/Hamiltonian payload from bath modes."""
from __future__ import annotations

import json

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob


class SbmSymbolicSpecFromModesCalcJob(RenoBaseCalcJob):
    """Build symbolic SBM basis/hamiltonian payload from explicit bath modes."""

    _template_name = "sbm_symbolic_spec_from_modes_driver.py.jinja"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        spec.input("omega_k", valid_type=orm.ArrayData, help="Bath mode frequencies")
        spec.input("c_j2", valid_type=orm.ArrayData, help="Bath coupling-square values")
        spec.input("delta_eff", valid_type=orm.Float, help="Effective delta")
        spec.input("epsilon", valid_type=orm.Float, default=lambda: orm.Float(0.0))
        spec.input("tree_type", valid_type=orm.Str, default=lambda: orm.Str("binary"))
        spec.input("m_max", valid_type=orm.Int, default=lambda: orm.Int(16))
        spec.input("symbol_map", valid_type=orm.Dict, default=lambda: orm.Dict(dict={}))
        spec.input("min_nbas", valid_type=orm.Float, default=lambda: orm.Float(4.0))
        spec.input("nbas_prefactor", valid_type=orm.Float, default=lambda: orm.Float(16.0))
        spec.input("vib_prefix", valid_type=orm.Str, default=lambda: orm.Str("v_"))

        spec.output(
            "output_parameters",
            valid_type=orm.Dict,
            help="Symbolic payload and construction summary.",
        )

        spec.exit_code(
            562,
            "ERROR_SYMBOLIC_SPEC_FAILED",
            message="Symbolic spec generation from bath modes failed",
        )

    def _write_input_files(self, folder) -> None:
        super()._write_input_files(folder)

        with folder.open("input_sbm_symbolic_spec.json", "w") as handle:
            json.dump(
                {
                    "delta_eff": self.inputs.delta_eff.value,
                    "epsilon": self.inputs.epsilon.value,
                    "tree_type": self.inputs.tree_type.value,
                    "m_max": self.inputs.m_max.value,
                    "symbol_map": {
                        str(k): str(v) for k, v in self.inputs.symbol_map.get_dict().items()
                    },
                    "min_nbas": self.inputs.min_nbas.value,
                    "nbas_prefactor": self.inputs.nbas_prefactor.value,
                    "vib_prefix": self.inputs.vib_prefix.value,
                },
                handle,
                indent=2,
            )

        with folder.open("input_omega_k.npy", "wb") as handle:
            import numpy as np

            np.save(handle, self.inputs.omega_k.get_array("omega_k"))

        with folder.open("input_c_j2.npy", "wb") as handle:
            import numpy as np

            np.save(handle, self.inputs.c_j2.get_array("c_j2"))

    def _get_retrieve_list(self) -> list[str]:
        retrieve_list = super()._get_retrieve_list()
        for artifact in ("output_mps.npz", "output_mpo.npz", "trajectory.npz"):
            if artifact in retrieve_list:
                retrieve_list.remove(artifact)
        return retrieve_list
