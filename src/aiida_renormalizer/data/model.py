"""ModelData node for Renormalizer Model persistence.

Works around Model.to_dict() not serializing basis by using basis_registry.
"""
from __future__ import annotations

import typing as t

from aiida.orm import Data

from aiida_renormalizer.data.utils import read_json_from_repository, write_json_to_repository

if t.TYPE_CHECKING:
    from renormalizer.model import Model


class ModelData(Data):
    """AiiDA Data node storing a complete Renormalizer Model (basis + Hamiltonian + dipole)."""

    @classmethod
    def from_model(cls, model: Model) -> ModelData:
        """Create a ModelData node from a Reno Model."""
        from aiida_renormalizer.data.basis_registry import serialize_basis
        from aiida_renormalizer.data.op import serialize_op, serialize_opsum

        node = cls()

        # Attributes (lightweight metadata for queries)
        node.base.attributes.set("n_sites", len(model.basis))
        node.base.attributes.set("dof_list", [str(b.dof) for b in model.basis])
        node.base.attributes.set("basis_types", [type(b).__name__ for b in model.basis])

        # Repository: basis.json
        basis_data = [serialize_basis(b) for b in model.basis]
        write_json_to_repository(node, basis_data, "basis.json")

        # Repository: ham_opsum.json
        ham_data = serialize_opsum(model.ham_terms)
        write_json_to_repository(node, ham_data, "ham_opsum.json")

        # Repository: dipole.json (optional)
        if model.dipole is not None:
            dipole_data = serialize_opsum(model.dipole)
            write_json_to_repository(node, dipole_data, "dipole.json")

        return node

    def load_model(self) -> Model:
        """Reconstruct a Reno Model (including basis) from stored data."""
        from renormalizer.model import Model as RenoModel

        from aiida_renormalizer.data.basis_registry import deserialize_basis
        from aiida_renormalizer.data.op import deserialize_opsum

        # Load basis
        basis_data = read_json_from_repository(self, "basis.json")
        basis = [deserialize_basis(d) for d in basis_data]

        # Load Hamiltonian
        ham_data = read_json_from_repository(self, "ham_opsum.json")
        ham_terms = deserialize_opsum(ham_data)

        # Load dipole (optional)
        dipole = None
        try:
            dipole_data = read_json_from_repository(self, "dipole.json")
            dipole = deserialize_opsum(dipole_data)
        except FileNotFoundError:
            pass

        return RenoModel(basis, ham_terms, dipole=dipole)
