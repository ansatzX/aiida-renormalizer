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
