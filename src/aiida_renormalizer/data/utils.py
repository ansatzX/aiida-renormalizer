"""Shared utility functions for data serialization."""
from __future__ import annotations

import enum
import io
import json
import typing as t

import numpy as np
from aiida.orm import Data

if t.TYPE_CHECKING:
    from aiida.orm import Node


def to_native(obj: t.Any) -> t.Any:
    """Recursively convert numpy/enum values to JSON-safe Python natives.

    Handles:
    - enum.Enum -> enum name (str)
    - np.generic -> native Python scalar
    - np.ndarray -> list
    - list/tuple -> recursively converted
    - np.inf -> None (JSON can't represent inf)
    """
    if isinstance(obj, enum.Enum):
        return obj.name
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        converted = [to_native(x) for x in obj]
        return type(obj)(converted) if isinstance(obj, tuple) else converted
    if np.isscalar(obj) and isinstance(obj, float) and np.isinf(obj):
        return None  # JSON can't represent inf; use None sentinel
    return obj


def is_dof_atom(obj: t.Any) -> bool:
    """Return True if obj is a supported Reno dof atom."""
    return isinstance(obj, (str, int)) or (
        isinstance(obj, tuple) and all(is_dof_atom(item) for item in obj)
    )


def encode_dof_atom(dof: t.Any) -> dict[str, t.Any]:
    """Encode a dof atom with enough information to survive JSON roundtrip."""
    if isinstance(dof, str):
        return {"kind": "str", "value": dof}
    if isinstance(dof, int):
        return {"kind": "int", "value": dof}
    if isinstance(dof, tuple) and all(is_dof_atom(item) for item in dof):
        return {"kind": "tuple", "items": [encode_dof_atom(item) for item in dof]}
    raise TypeError(f"unsupported dof atom: {dof!r}")


def decode_dof_atom(payload: dict[str, t.Any]) -> t.Any:
    """Decode a JSON-safe dof atom payload."""
    kind = payload["kind"]
    if kind == "str":
        return str(payload["value"])
    if kind == "int":
        return int(payload["value"])
    if kind == "tuple":
        return tuple(decode_dof_atom(item) for item in payload["items"])
    raise ValueError(f"unsupported dof atom kind: {kind}")


def encode_dofs(dofs: t.Any) -> dict[str, t.Any]:
    """Encode Op dofs, preserving single atom vs multiple atoms."""
    if is_dof_atom(dofs):
        return {"kind": "single", "value": encode_dof_atom(dofs)}
    if isinstance(dofs, list) and all(is_dof_atom(item) for item in dofs):
        return {"kind": "many", "items": [encode_dof_atom(item) for item in dofs]}
    raise TypeError(f"unsupported dofs payload: {dofs!r}")


def decode_dofs(payload: dict[str, t.Any]) -> t.Any:
    """Decode Op dofs payload into Reno-compatible Python objects."""
    kind = payload["kind"]
    if kind == "single":
        return decode_dof_atom(payload["value"])
    if kind == "many":
        return [decode_dof_atom(item) for item in payload["items"]]
    raise ValueError(f"unsupported dofs payload kind: {kind}")


def dof_atom_label(dof: t.Any) -> str:
    """Stable string label for metadata attributes."""
    return dof if isinstance(dof, str) else repr(dof)


def write_json_to_repository(node: Data, data: t.Any, path: str) -> None:
    """Write JSON-serializable data to a node's repository."""
    payload = json.dumps(data).encode("utf-8")
    node.base.repository.put_object_from_filelike(io.BytesIO(payload), path)


def read_json_from_repository(node: Data, path: str) -> t.Any:
    """Read JSON data from a node's repository."""
    with node.base.repository.open(path, "rb") as f:
        return json.load(f)


def get_linked_node(uuid: str) -> Node:
    """Get a linked AiiDA node by UUID."""
    from aiida.orm import load_node

    return load_node(uuid)
