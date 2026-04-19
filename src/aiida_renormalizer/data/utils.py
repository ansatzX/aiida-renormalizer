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
