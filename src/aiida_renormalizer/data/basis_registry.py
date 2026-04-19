"""Registry for serializing/deserializing Renormalizer BasisSet subclasses.

Each built-in BasisSet subclass has a registered (param_extractor, constructor)
pair. The extractor pulls constructor-compatible params from a live object;
the constructor rebuilds it from those params.
"""
from __future__ import annotations

import typing as t

import numpy as np

from aiida_renormalizer.data.utils import to_native

if t.TYPE_CHECKING:
    from renormalizer.model.basis import BasisSet

# class_name -> (param_extractor, constructor_class)
_REGISTRY: dict[str, tuple[t.Callable[..., dict], type]] = {}


def register_basis(
    cls_name: str,
    param_extractor: t.Callable[..., dict],
    constructor: type,
) -> None:
    """Register a BasisSet subclass for serialization."""
    _REGISTRY[cls_name] = (param_extractor, constructor)


def serialize_basis(basis: BasisSet) -> dict:
    """Serialize a BasisSet instance to a JSON-safe dict."""
    cls_name = type(basis).__name__
    if cls_name not in _REGISTRY:
        raise ValueError(
            f"Unknown basis type: {cls_name}. "
            f"Register it via register_basis() or the aiida.reno.basis_serializers entry point."
        )
    extractor, _ = _REGISTRY[cls_name]
    params = extractor(basis)
    # Ensure all values are JSON-serializable (convert numpy scalars)
    params = to_native(params)
    dof = basis.dof
    if isinstance(dof, np.generic):
        dof = dof.item()
    elif isinstance(dof, tuple):
        dof = list(dof)
    return {"type": cls_name, "dof": dof, "params": params}


def deserialize_basis(data: dict) -> BasisSet:
    """Reconstruct a BasisSet instance from a serialized dict."""
    cls_name = data["type"]
    if cls_name not in _REGISTRY:
        raise ValueError(
            f"Unknown basis type: {cls_name}. "
            f"Register it via register_basis() or the aiida.reno.basis_serializers entry point."
        )
    _, constructor = _REGISTRY[cls_name]
    params = dict(data["params"])
    # Convert sigmaqn back to numpy array if present
    if "sigmaqn" in params and params["sigmaqn"] is not None:
        params["sigmaqn"] = np.array(params["sigmaqn"])
    dof = data["dof"]
    if isinstance(dof, list):
        dof = tuple(dof)
    return constructor(dof, **params)




def _extract_sho(b: BasisSet) -> dict:
    return {
        "omega": b.omega,
        "nbas": b.nbas,
        "x0": b.x0,
        "dvr": b.dvr,
        "general_xp_power": b.general_xp_power,
    }


def _extract_sine_dvr(b: BasisSet) -> dict:
    return {
        "nbas": b.nbas,
        "xi": b.xi,
        "xf": b.xf,
        "quadrature": getattr(b, "quadrature", False),
        "dvr": b.dvr,
    }


def _extract_sigmaqn(b: BasisSet) -> dict:
    sigmaqn = b.sigmaqn
    if sigmaqn is not None and isinstance(sigmaqn, np.ndarray):
        return {"sigmaqn": sigmaqn.tolist()}
    return {"sigmaqn": sigmaqn}


def _extract_sigmaqn_required(b: BasisSet) -> dict:
    return {"sigmaqn": b.sigmaqn.tolist()}


def _extract_empty(b: BasisSet) -> dict:
    return {}


def _extract_nbas(b: BasisSet) -> dict:
    return {"nbas": b.nbas}


def _extract_dummy(b: BasisSet) -> dict:
    params: dict[str, t.Any] = {"nbas": b.nbas}
    if b.sigmaqn is not None and isinstance(b.sigmaqn, np.ndarray):
        params["sigmaqn"] = b.sigmaqn.tolist()
    else:
        params["sigmaqn"] = b.sigmaqn
    return params


def _register_builtins() -> None:
    from renormalizer.model.basis import (
        BasisDummy,
        BasisHalfSpin,
        BasisHopsBoson,
        BasisMultiElectron,
        BasisMultiElectronVac,
        BasisSHO,
        BasisSimpleElectron,
        BasisSineDVR,
    )

    register_basis("BasisSHO", _extract_sho, BasisSHO)
    register_basis("BasisSineDVR", _extract_sine_dvr, BasisSineDVR)
    register_basis("BasisHalfSpin", _extract_sigmaqn, BasisHalfSpin)
    register_basis("BasisSimpleElectron", _extract_sigmaqn, BasisSimpleElectron)
    register_basis("BasisMultiElectron", _extract_sigmaqn_required, BasisMultiElectron)
    register_basis("BasisMultiElectronVac", _extract_empty, BasisMultiElectronVac)
    register_basis("BasisDummy", _extract_dummy, BasisDummy)
    register_basis("BasisHopsBoson", _extract_nbas, BasisHopsBoson)


_register_builtins()
