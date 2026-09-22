"""Validation for the construction choices supported by this case."""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Complex, Integral

import numpy as np


def normalize_initial_state(initial_state):
    """Return JSON-native occupations or normalized real local amplitude vectors.

    This converts representation only; it never rescales the supplied state.
    Squared vector norms must equal one within rtol=1e-10 and atol=1e-12.
    Complex containers are accepted only when every imaginary component is zero,
    since Reno's Hartree-product constructor stores real local tensors.
    """
    if not isinstance(initial_state, Mapping) or any(
        not isinstance(dof, str) or not dof for dof in initial_state
    ):
        raise ValueError("initial_state must map non-empty string dofs to local states")
    normalized = {}
    for dof, state in initial_state.items():
        label = f"initial_state[{dof!r}]"
        if isinstance(state, Integral) and not isinstance(state, (bool, np.bool_)):
            normalized[dof] = int(state)
            continue
        if isinstance(state, np.ndarray):
            if state.ndim != 1:
                raise ValueError(f"{label} must be a one-dimensional local amplitude vector")
            state = state.tolist()
        if not isinstance(state, (list, tuple)) or not state:
            raise ValueError(f"{label} must be an occupation integer or a local amplitude vector")
        amplitudes = []
        for value in state:
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, Complex):
                raise ValueError(f"{label} amplitudes must be finite real numbers")
            if value.imag != 0:
                raise ValueError(
                    f"{label} complex amplitudes are not supported by the real Hartree path"
                )
            amplitude = float(value.real)
            if not math.isfinite(amplitude):
                raise ValueError(f"{label} amplitudes must be finite real numbers")
            amplitudes.append(amplitude)
        norm_squared = math.fsum(value * value for value in amplitudes)
        if not math.isclose(norm_squared, 1.0, rel_tol=1e-10, abs_tol=1e-12):
            raise ValueError(
                f"{label} amplitude vector must be normalized; no automatic rescaling is applied"
            )
        normalized[dof] = amplitudes
    return normalized


def validate_construction(basis, topology, initial_state, compression, expansion):
    required = {
        "kind",
        "bath_dofs",
        "root_dofs",
        "contract_primitive",
        "contract_labels",
        "dummy_label",
    }
    if set(topology) != required or topology["kind"] != "binary_mctdh":
        raise ValueError("topology must explicitly specify the binary_mctdh construction")
    dofs = [item["dof"] for item in basis]
    selected = topology["bath_dofs"] + topology["root_dofs"]
    if (
        len(set(dofs)) != len(dofs)
        or len(set(selected)) != len(selected)
        or set(selected) != set(dofs)
    ):
        raise ValueError("topology must assign every basis dof exactly once")
    if len(topology["bath_dofs"]) < 2 or not topology["root_dofs"]:
        raise ValueError("topology needs at least two bath dofs and a root child")
    if type(topology["contract_primitive"]) is not bool:
        raise ValueError("contract_primitive must be a bool")
    labels = topology["contract_labels"]
    if topology["contract_primitive"]:
        if (
            not isinstance(labels, list)
            or len(labels) != len(topology["bath_dofs"])
            or any(type(value) is not bool for value in labels)
        ):
            raise ValueError("contract_labels must supply one bool per bath dof")
    elif labels is not None:
        raise ValueError("contract_labels must be None when primitive contraction is disabled")
    if not isinstance(topology["dummy_label"], str) or not topology["dummy_label"]:
        raise ValueError("dummy_label must be non-empty")
    initial_state = normalize_initial_state(initial_state)
    if set(initial_state) != set(dofs):
        raise ValueError("initial_state must explicitly specify every physical dof")
    for item in basis:
        size = 2 if item["kind"] == "half_spin" else item["nbas"]
        state = initial_state[item["dof"]]
        if type(size) is not int or size < 1:
            raise ValueError("basis size must be positive")
        if isinstance(state, int):
            if not 0 <= state < size:
                raise ValueError(
                    f"initial_state[{item['dof']!r}] occupation is outside the local basis"
                )
            continue
        if len(state) != size:
            raise ValueError(
                f"initial_state[{item['dof']!r}] amplitude dimension must match basis size {size}"
            )
        quantum_numbers = np.asarray(item.get("sigmaqn", [0] * size))
        if quantum_numbers.ndim not in (1, 2) or len(quantum_numbers) != size:
            raise ValueError("basis quantum numbers must have one entry per local basis state")
        occupied_qn = quantum_numbers[np.flatnonzero(state)]
        if not np.all(occupied_qn == occupied_qn[0]):
            raise ValueError(f"initial_state[{item['dof']!r}] mixes quantum-number sectors")
    if set(compression) != {"criteria", "max_bonddim"} or compression["criteria"] != "fixed":
        raise ValueError("compression must explicitly select fixed max_bonddim")
    if type(compression["max_bonddim"]) is not int or compression["max_bonddim"] < 1:
        raise ValueError("max_bonddim must be positive")
    if set(expansion) != {"enabled", "hint", "coefficient"} or expansion["hint"] != "hamiltonian":
        raise ValueError("expansion must explicitly select a Hamiltonian hint and coefficient")
    if (
        type(expansion["enabled"]) is not bool
        or not math.isfinite(expansion["coefficient"])
        or expansion["coefficient"] <= 0
    ):
        raise ValueError("expansion requires enabled bool and positive finite coefficient")
    return {
        "topology": topology,
        "initial_state": initial_state,
        "compression": compression,
        "expansion": expansion,
    }
