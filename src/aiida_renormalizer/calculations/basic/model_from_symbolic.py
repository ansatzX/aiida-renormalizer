"""CalcJob to materialize ModelData from symbolic basis/Hamiltonian payload."""
from __future__ import annotations

import json

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import ModelData


class ModelFromSymbolicSpecCalcJob(RenoBaseCalcJob):
    """Build a model payload consumable by parser-side ModelData construction."""

    _template_name = "model_from_symbolic_spec_driver.py.jinja"

    @classmethod
    def _validate_symbolic_inputs(cls, value: orm.Dict, _) -> str | None:
        payload = value.get_dict()
        if not isinstance(payload, dict):
            return "symbolic_inputs must be a dict."
        basis = payload.get("basis")
        ham = payload.get("hamiltonian")
        if not isinstance(basis, list) or not basis:
            return "symbolic_inputs.basis must be a non-empty list."
        if not isinstance(ham, list) or not ham:
            return "symbolic_inputs.hamiltonian must be a non-empty list."
        return None

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        spec.input(
            "symbolic_inputs",
            valid_type=orm.Dict,
            validator=cls._validate_symbolic_inputs,
            help="Model symbolic payload: basis/hamiltonian/(optional dipole).",
        )
        spec.output(
            "output_model",
            valid_type=ModelData,
            required=False,
            help="ModelData reconstructed by parser from symbolic payload.",
        )
        spec.exit_code(
            572,
            "ERROR_SYMBOLIC_MODEL_INVALID",
            message="Invalid symbolic model payload",
        )

    def _write_input_files(self, folder) -> None:
        super()._write_input_files(folder)
        with folder.open("input_symbolic_model.json", "w") as handle:
            json.dump(self.inputs.symbolic_inputs.get_dict(), handle, indent=2)

    def _get_retrieve_list(self) -> list[str]:
        retrieve_list = super()._get_retrieve_list()
        for artifact in ("output_mps.npz", "output_mpo.npz", "trajectory.npz"):
            if artifact in retrieve_list:
                retrieve_list.remove(artifact)
        return retrieve_list
