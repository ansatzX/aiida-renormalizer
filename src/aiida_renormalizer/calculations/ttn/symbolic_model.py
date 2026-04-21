"""CalcJob for symbolic TTN model payload generation from SBM parameters."""
from __future__ import annotations

import json

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import BasisTreeData


class TTNSymbolicModelCalcJob(RenoBaseCalcJob):
    """Build symbolic TTN inputs from declarative SBM process parameters."""

    _template_name = "ttn_symbolic_model_driver.py.jinja"
    _known_steps = {
        "build_sdf",
        "adiabatic_renormalization",
        "discretize_bath",
        "build_symbolic_hamiltonian",
        "ttns_tdvp",
    }

    @classmethod
    def _validate_process(cls, value: orm.List, _) -> str | None:
        process = value.get_list()
        if not isinstance(process, list) or not process:
            return "process must be a non-empty list."
        unknown = [step for step in process if step not in cls._known_steps]
        if unknown:
            return f"Unknown process steps: {unknown}"
        if "build_symbolic_hamiltonian" not in process:
            return "process must include 'build_symbolic_hamiltonian'."
        return None

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        spec.input("alpha", valid_type=orm.Float, help="SBM coupling strength")
        spec.input("s_exponent", valid_type=orm.Float, default=lambda: orm.Float(1.0))
        spec.input("omega_c", valid_type=orm.Float, help="Bath cutoff frequency")
        spec.input("n_modes", valid_type=orm.Int, help="Number of bath modes")
        spec.input("raw_delta", valid_type=orm.Float, help="Bare tunneling parameter")
        spec.input(
            "renormalization_p",
            valid_type=orm.Float,
            help="Adiabatic renormalization parameter p",
        )
        spec.input("tree_type", valid_type=orm.Str, default=lambda: orm.Str("binary"))
        spec.input("m_max", valid_type=orm.Int, default=lambda: orm.Int(16))
        spec.input("symbol_map", valid_type=orm.Dict, default=lambda: orm.Dict(dict={}))
        spec.input(
            "process",
            valid_type=orm.List,
            validator=cls._validate_process,
            default=lambda: orm.List(
                list=[
                    "build_sdf",
                    "adiabatic_renormalization",
                    "discretize_bath",
                    "build_symbolic_hamiltonian",
                    "ttns_tdvp",
                ]
            ),
        )

        spec.output(
            "output_parameters",
            valid_type=orm.Dict,
            help="Metadata and symbolic input payload.",
        )
        spec.output(
            "output_basis_tree",
            valid_type=BasisTreeData,
            required=False,
            help="Compiled basis tree topology cache for downstream TTN CalcJobs.",
        )

        spec.exit_code(
            531,
            "ERROR_SYMBOLIC_MODEL_BUILD_FAILED",
            message="Symbolic TTN model payload generation failed",
        )

    def _write_input_files(self, folder) -> None:
        super()._write_input_files(folder)

        payload = {
            "alpha": self.inputs.alpha.value,
            "s_exponent": self.inputs.s_exponent.value,
            "omega_c": self.inputs.omega_c.value,
            "n_modes": self.inputs.n_modes.value,
            "raw_delta": self.inputs.raw_delta.value,
            "renormalization_p": self.inputs.renormalization_p.value,
            "tree_type": self.inputs.tree_type.value,
            "m_max": self.inputs.m_max.value,
            "symbol_map": {str(k): str(v) for k, v in self.inputs.symbol_map.get_dict().items()},
            "process": self.inputs.process.get_list(),
        }
        with folder.open("input_ttn_symbolic_model.json", "w") as handle:
            json.dump(payload, handle, indent=2)

    def _get_retrieve_list(self) -> list[str]:
        retrieve_list = super()._get_retrieve_list()
        for artifact in ("output_mps.npz", "output_mpo.npz", "trajectory.npz"):
            if artifact in retrieve_list:
                retrieve_list.remove(artifact)
        return retrieve_list
