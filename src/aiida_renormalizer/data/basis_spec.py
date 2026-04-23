"""BasisSpecData for lightweight Reno-aligned basis specifications."""
from __future__ import annotations

import typing as t

from aiida.orm import Data

from aiida_renormalizer.data.utils import (
    decode_dof_atom,
    dof_atom_label,
    encode_dof_atom,
    is_dof_atom,
    read_json_from_repository,
    to_native,
    write_json_to_repository,
)


class BasisSpecData(Data):
    """AiiDA Data node storing basis specifications before Reno Basis build."""

    @classmethod
    def from_list(cls, basis_specs: list[t.Any]) -> "BasisSpecData":
        if not isinstance(basis_specs, list) or not basis_specs:
            raise ValueError("basis_specs must be a non-empty list")

        normalized: list[dict[str, t.Any]] = []
        kinds: list[str] = []
        dofs: list[str] = []

        def _normalize_dof_atom(raw_dof: t.Any) -> t.Any:
            if isinstance(raw_dof, (str, int)):
                return raw_dof
            if isinstance(raw_dof, (list, tuple)):
                return tuple(_normalize_dof_atom(item) for item in raw_dof)
            raise TypeError(f"unsupported dof atom: {raw_dof!r}")

        for index, item in enumerate(basis_specs):
            if isinstance(item, dict):
                if "kind" not in item or "dof" not in item:
                    raise ValueError(f"basis_specs[{index}] must contain kind and dof")
                normalized_item = {str(key): to_native(value) for key, value in item.items()}
            elif isinstance(item, (list, tuple)) and item:
                kind = str(item[0])
                if kind == "half_spin":
                    if len(item) not in (2, 3):
                        raise ValueError(
                            f"basis_specs[{index}] half_spin form must be [kind, dof] or [kind, dof, sigmaqn]"
                        )
                    normalized_item = {"kind": kind, "dof": item[1]}
                    if len(item) == 3:
                        normalized_item["sigmaqn"] = item[2]
                elif kind == "sho":
                    if len(item) != 4:
                        raise ValueError(f"basis_specs[{index}] sho form must be [kind, dof, omega, nbas]")
                    normalized_item = {
                        "kind": kind,
                        "dof": item[1],
                        "omega": item[2],
                        "nbas": item[3],
                    }
                elif kind == "simple_electron":
                    if len(item) != 2:
                        raise ValueError(f"basis_specs[{index}] simple_electron form must be [kind, dof]")
                    normalized_item = {"kind": kind, "dof": item[1]}
                else:
                    raise ValueError(f"basis_specs[{index}] has unsupported basis kind: {kind}")
                normalized_item = {str(key): to_native(value) for key, value in normalized_item.items()}
            else:
                raise TypeError(f"basis_specs[{index}] must be a dict or list/tuple form")

            dof = _normalize_dof_atom(normalized_item["dof"])
            if not is_dof_atom(dof):
                raise TypeError(f"basis_specs[{index}].dof must be a supported dof atom")

            normalized_item["kind"] = str(normalized_item["kind"])
            normalized_item["dof"] = encode_dof_atom(dof)
            normalized.append(normalized_item)
            kinds.append(normalized_item["kind"])
            dofs.append(dof_atom_label(dof))

        node = cls()
        node.base.attributes.set("schema", "basis_spec_v1")
        node.base.attributes.set("n_items", len(normalized))
        node.base.attributes.set("basis_kinds", kinds)
        node.base.attributes.set("dof_list", dofs)
        write_json_to_repository(node, normalized, "basis_spec.json")
        return node

    def as_list(self) -> list[dict[str, t.Any]]:
        payload = read_json_from_repository(self, "basis_spec.json")
        decoded: list[dict[str, t.Any]] = []
        for item in payload:
            decoded_item = dict(item)
            decoded_item["dof"] = decode_dof_atom(item["dof"])
            decoded.append(decoded_item)
        return decoded
