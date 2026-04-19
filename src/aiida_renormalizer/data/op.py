"""OpData node and Op/OpSum serialization helpers."""
from __future__ import annotations

import io
import json
import typing as t

import numpy as np
from aiida.orm import Data

from aiida_renormalizer.data.utils import read_json_from_repository, write_json_to_repository

if t.TYPE_CHECKING:
    from renormalizer.model import Op
    from renormalizer.model.op import OpSum


def serialize_op(op: Op) -> dict:
    """Serialize a single Op to a JSON-safe dict."""
    symbol, dofs, factor, qn = op.to_tuple()
    factor_c = complex(factor)
    return {
        "symbol": symbol,
        "dofs": [d.item() if isinstance(d, np.generic) else d for d in dofs],
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
    dofs = data["dofs"]
    if len(dofs) == 1:
        dofs = dofs[0]
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
