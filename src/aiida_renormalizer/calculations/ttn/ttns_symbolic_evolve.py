"""CalcJob for symbolic TTNS time evolution."""
from __future__ import annotations

import json

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import BasisTreeData, TTNSData


class TtnsSymbolicEvolveCalcJob(RenoBaseCalcJob):
    """TTNS evolution from symbolic inputs.

    This CalcJob separates:
    - local setup: symbolic parameters only
    - remote/runtime work: basis/op parsing, tree build, TTNO/TTNS construction, evolution
    """

    _template_name = "ttns_symbolic_evolve_driver.py.jinja"
    _allowed_methods = {
        "tdvp_ps",
        "tdvp_ps2",
        "tdvp_vmf",
        "prop_and_compress_tdrk4",
    }

    @classmethod
    def _validate_symbolic_dict(cls, data: dict) -> str | None:
        if not isinstance(data, dict):
            return "symbolic_inputs must be a dict payload."

        basis = data.get("basis")
        if not isinstance(basis, list) or len(basis) == 0:
            return "symbolic_inputs.basis must be a non-empty list."

        for idx, item in enumerate(basis):
            if not isinstance(item, dict):
                return f"basis[{idx}] must be a dict."
            if item.get("kind") not in {"half_spin", "sho"}:
                return f"basis[{idx}].kind must be 'half_spin' or 'sho'."
            if "dof" not in item:
                return f"basis[{idx}].dof is required."
            if item["kind"] == "sho" and ("omega" not in item or "nbas" not in item):
                return f"basis[{idx}] with kind='sho' requires 'omega' and 'nbas'."

        hamiltonian = data.get("hamiltonian")
        if not isinstance(hamiltonian, list) or len(hamiltonian) == 0:
            return "symbolic_inputs.hamiltonian must be a non-empty list."

        for idx, item in enumerate(hamiltonian):
            if not isinstance(item, dict):
                return f"hamiltonian[{idx}] must be a dict."
            if "symbol" not in item or "dofs" not in item:
                return f"hamiltonian[{idx}] requires 'symbol' and 'dofs'."

        tree_type = data.get("tree_type", "binary")
        if tree_type not in {"binary", "linear"}:
            return "symbolic_inputs.tree_type must be 'binary' or 'linear'."

        m_max = data.get("m_max", 16)
        if not isinstance(m_max, int) or m_max <= 0:
            return "symbolic_inputs.m_max must be a positive integer."

        return None

    @classmethod
    def _validate_symbolic_inputs(cls, value: orm.Dict, _) -> str | None:
        return cls._validate_symbolic_dict(value.get_dict())

    @classmethod
    def _validate_dt(cls, value: orm.Float, _) -> str | None:
        if value.value <= 0:
            return "dt must be > 0."
        return None

    @classmethod
    def _validate_nsteps(cls, value: orm.Int, _) -> str | None:
        if value.value <= 0:
            return "nsteps must be > 0."
        return None

    @classmethod
    def _validate_method(cls, value: orm.Str, _) -> str | None:
        if value.value not in cls._allowed_methods:
            return f"Unsupported method '{value.value}'. Allowed: {sorted(cls._allowed_methods)}"
        return None

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        spec.input(
            "symbolic_inputs",
            valid_type=orm.Dict,
            validator=cls._validate_symbolic_inputs,
            help="Symbolic system definition: basis, Hamiltonian, tree_type, m_max",
        )
        spec.input(
            "dt",
            valid_type=orm.Float,
            validator=cls._validate_dt,
            help="Time step for each evolution step",
        )
        spec.input(
            "nsteps",
            valid_type=orm.Int,
            validator=cls._validate_nsteps,
            help="Number of evolution steps",
        )
        spec.input(
            "method",
            valid_type=orm.Str,
            required=False,
            default=lambda: orm.Str("tdvp_ps"),
            validator=cls._validate_method,
            help="Evolution method name in renormalizer.utils.configs.EvolveMethod",
        )

        spec.output("output_ttns", valid_type=TTNSData, required=False, help="Evolved TTNS state")
        spec.output("output_basis_tree", valid_type=BasisTreeData, required=False, help="Basis tree used")

    def _write_input_files(self, folder) -> None:
        """Write symbolic input payloads."""
        super()._write_input_files(folder)

        with folder.open("input_symbolic.json", "w") as f:
            json.dump(self.inputs.symbolic_inputs.get_dict(), f, indent=2)

        with folder.open("input_evolution_params.json", "w") as f:
            json.dump(
                {
                    "dt": self.inputs.dt.value,
                    "nsteps": self.inputs.nsteps.value,
                    "method": self.inputs.method.value,
                },
                f,
                indent=2,
            )

    def _get_retrieve_list(self) -> list[str]:
        return [
            "output_parameters.json",
            "output_ttns.npz",
            "output_basis_tree.npz",
            "trajectory.npz",
            "aiida.out",
            "aiida.err",
        ]
