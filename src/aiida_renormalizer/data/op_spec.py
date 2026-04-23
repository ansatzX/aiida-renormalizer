"""OpSpecData for lightweight Reno-aligned operator specifications."""
from __future__ import annotations

import typing as t

from aiida.orm import Data

from aiida_renormalizer.data.utils import (
    decode_dofs,
    dof_atom_label,
    encode_dofs,
    is_dof_atom,
    read_json_from_repository,
    to_native,
    write_json_to_repository,
)


class OpSpecData(Data):
    """AiiDA Data node storing operator specifications before Reno Op build."""

    @classmethod
    def from_list(cls, op_specs: list[dict[str, t.Any]]) -> "OpSpecData":
        if not isinstance(op_specs, list) or not op_specs:
            raise ValueError("op_specs must be a non-empty list")

        normalized: list[dict[str, t.Any]] = []
        all_dofs: set[str] = set()
        for index, item in enumerate(op_specs):
            if not isinstance(item, dict):
                raise TypeError(f"op_specs[{index}] must be a dict")
            if "symbol" not in item or "dofs" not in item or "factor" not in item:
                raise ValueError(f"op_specs[{index}] must contain symbol, dofs, and factor")

            dofs = item["dofs"]
            if is_dof_atom(dofs):
                dof_list = [dofs]
            elif isinstance(dofs, list) and all(is_dof_atom(dof) for dof in dofs):
                dof_list = list(dofs)
            else:
                raise TypeError(f"op_specs[{index}].dofs must be a supported dof atom or list of dof atoms")

            normalized_item = {
                "symbol": str(item["symbol"]),
                "dofs": encode_dofs(dofs),
                "factor": to_native(item["factor"]),
                "qn": to_native(item.get("qn", 0)),
            }
            normalized.append(normalized_item)
            all_dofs.update(dof_atom_label(dof) for dof in dof_list)

        node = cls()
        node.base.attributes.set("schema", "op_spec_v1")
        node.base.attributes.set("n_terms", len(normalized))
        node.base.attributes.set("dof_list", sorted(all_dofs))
        write_json_to_repository(node, normalized, "op_spec.json")
        return node

    def as_list(self) -> list[dict[str, t.Any]]:
        payload = read_json_from_repository(self, "op_spec.json")
        return [
            {
                "symbol": item["symbol"],
                "dofs": decode_dofs(item["dofs"]),
                "factor": item["factor"],
                "qn": item.get("qn", 0),
            }
            for item in payload
        ]
