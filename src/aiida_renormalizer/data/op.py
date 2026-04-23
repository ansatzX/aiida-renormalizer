"""OpData node and Op/OpSum serialization helpers."""
from __future__ import annotations

import io
import json
import typing as t

import numpy as np
from aiida.orm import Data

from aiida_renormalizer.data.utils import (
    decode_dofs,
    dof_atom_label,
    encode_dofs,
    is_dof_atom,
    read_json_from_repository,
    write_json_to_repository,
)

if t.TYPE_CHECKING:
    from renormalizer.model import Op
    from renormalizer.model.op import OpSum


def serialize_op(op: Op) -> dict:
    """Serialize a single Op to a JSON-safe dict."""
    symbol, dofs, factor, qn = op.to_tuple()
    factor_c = complex(factor)
    # Reno Op.to_tuple() returns the operator dofs as a sequence of dof atoms.
    # Even a single x((1, 0)) comes back as ((1, 0),), and b^\dagger b on one
    # site comes back as ("v0", "v0"). Preserve that "many-atoms" structure;
    # treating the tuple as a single dof atom breaks roundtrip semantics.
    if isinstance(dofs, (list, tuple)):
        encoded_dofs = encode_dofs(
            [d.item() if isinstance(d, np.generic) else d for d in dofs]
        )
    else:
        encoded_dofs = encode_dofs(dofs.item() if isinstance(dofs, np.generic) else dofs)
    return {
        "symbol": symbol,
        "dofs": encoded_dofs,
        "factor": {"real": float(factor_c.real), "imag": float(factor_c.imag)},
        "qn": [list(int(x) if isinstance(x, (np.integer,)) else x for x in q) for q in qn] if qn else None,
    }


def deserialize_op(data: dict) -> Op:
    """Reconstruct an Op from a serialized dict."""
    from renormalizer.model import Op as RenoOp

    factor_d = data["factor"]
    factor = complex(factor_d["real"], factor_d["imag"])
    if factor.imag == 0:
        factor = factor.real
    dofs = decode_dofs(data["dofs"])
    qn = data.get("qn")
    return RenoOp(data["symbol"], dofs, factor, qn=qn)


def serialize_opsum(opsum: OpSum) -> list[dict]:
    """Serialize an OpSum (list of Op) to a JSON-safe list."""
    return [serialize_op(op) for op in opsum]


def deserialize_opsum(data: list[dict]) -> OpSum:
    """Reconstruct an OpSum from a serialized list."""
    from renormalizer.model.op import OpSum as RenoOpSum

    return RenoOpSum([deserialize_op(d) for d in data])


class OpData(Data):
    """AiiDA Data node wrapping a Renormalizer Op or OpSum."""

    @classmethod
    def from_serialized_opsum(cls, serialized_opsum: list[dict]) -> OpData:
        """Create an OpData node directly from serialized OpSum payload."""
        node = cls()
        node.base.attributes.set("op_type", "OpSum")
        all_dofs: set[str] = set()
        encoded_payload = []
        for term in serialized_opsum:
            dofs = term["dofs"]
            if is_dof_atom(dofs):
                dof_list = [dofs]
            else:
                dof_list = list(dofs)
            for dof in dof_list:
                all_dofs.add(dof_atom_label(dof))
            encoded_payload.append(
                {
                    "symbol": term["symbol"],
                    "dofs": encode_dofs(dofs),
                    "factor": term["factor"],
                    "qn": term.get("qn"),
                }
            )
        node.base.attributes.set("dofs", sorted(all_dofs))
        node.base.attributes.set("n_terms", len(serialized_opsum))
        write_json_to_repository(node, encoded_payload, "op.json")
        return node

    @classmethod
    def from_op(cls, op: Op) -> OpData:
        """Create an OpData node from a single Op."""
        node = cls()
        node.base.attributes.set("op_type", "Op")
        node.base.attributes.set("dofs", list(op.dofs))
        node.base.attributes.set("n_terms", 1)

        write_json_to_repository(node, serialize_op(op), "op.json")
        return node

    @classmethod
    def from_opsum(cls, opsum: OpSum) -> OpData:
        """Create an OpData node from an OpSum."""
        node = cls()
        node.base.attributes.set("op_type", "OpSum")
        # Collect all unique dofs
        all_dofs: set[str] = set()
        for op in opsum:
            all_dofs.update(str(d) for d in op.dofs)
        node.base.attributes.set("dofs", sorted(all_dofs))
        node.base.attributes.set("n_terms", len(opsum))

        write_json_to_repository(node, serialize_opsum(opsum), "op.json")
        return node

    def load_op(self) -> Op:
        """Load a single Op from this node. Raises if op_type != 'Op'."""
        assert self.base.attributes.get("op_type") == "Op"
        data = read_json_from_repository(self, "op.json")
        return deserialize_op(data)

    def load_opsum(self) -> OpSum:
        """Load an OpSum from this node. Works for both Op and OpSum types."""
        op_type = self.base.attributes.get("op_type")
        data = read_json_from_repository(self, "op.json")
        if op_type == "Op":
            from renormalizer.model.op import OpSum as RenoOpSum

            return RenoOpSum([deserialize_op(data)])
        return deserialize_opsum(data)

    def as_serialized_opsum(self) -> list[dict]:
        """Return serialized OpSum payload without rebuilding Reno objects."""
        op_type = self.base.attributes.get("op_type")
        data = read_json_from_repository(self, "op.json")
        raw = [data] if op_type == "Op" else data
        return [
            {
                "symbol": item["symbol"],
                "dofs": decode_dofs(item["dofs"]),
                "factor": item["factor"],
                "qn": item.get("qn"),
            }
            for item in raw
        ]
