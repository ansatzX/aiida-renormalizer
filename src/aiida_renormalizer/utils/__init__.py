"""Utility functions for aiida-renormalizer."""

from __future__ import annotations

from typing import Any, Callable, TypeVar

import numpy as np
from aiida import orm
from aiida.engine import run_get_node
from plumpy.ports import PortNamespace

T = TypeVar("T")


def _normalize_python_literal(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return [_normalize_python_literal(item) for item in value.tolist()]
    if isinstance(value, list):
        return [_normalize_python_literal(item) for item in value]
    if isinstance(value, tuple):
        return [_normalize_python_literal(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _normalize_python_literal(item) for key, item in value.items()}
    return value


def _coerce_aiida_value(value: Any, valid_type: Any) -> Any:
    if isinstance(value, orm.Data) or valid_type is None:
        return value

    types = valid_type if isinstance(valid_type, tuple) else (valid_type,)
    python_value = _normalize_python_literal(value)

    for dtype in types:
        if not isinstance(dtype, type):
            continue
        if issubclass(dtype, orm.Bool) and isinstance(python_value, bool):
            return orm.Bool(python_value)
        if issubclass(dtype, orm.Int) and isinstance(python_value, int) and not isinstance(python_value, bool):
            return orm.Int(python_value)
        if issubclass(dtype, orm.Float) and isinstance(python_value, (int, float)) and not isinstance(python_value, bool):
            return orm.Float(float(python_value))
        if issubclass(dtype, orm.Str) and isinstance(python_value, str):
            return orm.Str(python_value)
        if issubclass(dtype, orm.List) and isinstance(python_value, list):
            return orm.List(list=python_value)
        if issubclass(dtype, orm.Dict) and isinstance(python_value, dict):
            return orm.Dict(dict=python_value)
    return value


def _coerce_inputs_for_ports(inputs: dict[str, Any], ports: PortNamespace) -> dict[str, Any]:
    converted: dict[str, Any] = {}
    for name, value in inputs.items():
        port = ports.get(name)
        if isinstance(port, PortNamespace) and isinstance(value, dict):
            converted[name] = _coerce_inputs_for_ports(value, port)
            continue
        valid_type = getattr(port, "valid_type", None) if port is not None else None
        converted[name] = _coerce_aiida_value(value, valid_type)
    return converted


def run_process(process: Callable[..., T], *, debug_provenance: bool = False, **inputs) -> tuple[T, orm.ProcessNode]:
    """Run an AiiDA process with automatic coercion from Python/numpy scalars to AiiDA Data types."""
    spec = getattr(process, "spec", None)
    converted_inputs = _coerce_inputs_for_ports(inputs, spec().inputs) if callable(spec) else dict(inputs)
    outputs, node = run_get_node(process, **converted_inputs)
    return outputs, node


__all__ = ["run_process"]
