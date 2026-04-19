"""BasisSetData node for standalone BasisSet persistence."""
from __future__ import annotations

import typing as t

from aiida.orm import Data

from aiida_renormalizer.data.utils import read_json_from_repository, write_json_to_repository

if t.TYPE_CHECKING:
    from renormalizer.model.basis import BasisSet


class BasisSetData(Data):
    """AiiDA Data node storing a list of BasisSet objects (reusable across Models)."""

    @classmethod
    def from_basis_list(cls, basis_list: list[BasisSet]) -> BasisSetData:
        """Create a BasisSetData node from a list of BasisSet instances."""
        from aiida_renormalizer.data.basis_registry import serialize_basis

        node = cls()
        node.base.attributes.set("n_sites", len(basis_list))
        node.base.attributes.set("basis_types", [type(b).__name__ for b in basis_list])
        node.base.attributes.set(
            "dof_list", [str(b.dof) for b in basis_list]
        )

        data = [serialize_basis(b) for b in basis_list]
        write_json_to_repository(node, data, "basis.json")
        return node

    def load_basis_list(self) -> list[BasisSet]:
        """Reconstruct the list of BasisSet instances."""
        from aiida_renormalizer.data.basis_registry import deserialize_basis

        data = read_json_from_repository(self, "basis.json")
        return [deserialize_basis(d) for d in data]
