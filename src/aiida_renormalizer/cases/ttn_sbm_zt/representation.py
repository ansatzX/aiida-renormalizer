"""Render the canonical representations of the native Reno objects used by this case."""

from __future__ import annotations

import math


def normalize_operator_terms(raw_terms):
    """Decode recorded numeric coefficients without adding scientific defaults."""
    if not isinstance(raw_terms, list) or not raw_terms:
        raise ValueError("supply at least one operator term")
    terms = []
    for item in raw_terms:
        if set(item) != {"symbol", "dofs", "factor", "qn"}:
            raise ValueError("operator terms require symbol, dofs, factor, and quantum numbers")
        dofs = item["dofs"]
        if (
            not isinstance(dofs, list)
            or not dofs
            or any(not isinstance(dof, str) or not dof for dof in dofs)
        ):
            raise ValueError("this case supports non-empty string dofs")
        factor = item["factor"]
        factor = complex(factor["real"], factor["imag"])
        if not math.isfinite(factor.real) or not math.isfinite(factor.imag):
            raise ValueError("operator coefficients must be finite")
        terms.append({**item, "factor": factor.real if factor.imag == 0 else factor})
    return terms


def render_operator_terms(terms):
    calls = [
        f"Op({item['symbol']!r}, {item['dofs']!r}, factor={item['factor']!r}, qn={item['qn']!r})"
        for item in terms
    ]
    return "ham_terms.extend([\n        " + ",\n        ".join(calls) + "\n    ])"


def render_basis(basis):
    calls = []
    for item in basis:
        dof = item["dof"]
        if not isinstance(dof, str) or not dof:
            raise ValueError("this case supports non-empty string dofs")
        if item["kind"] == "half_spin":
            calls.append(f"ba.BasisHalfSpin({dof!r}, sigmaqn={item['sigmaqn']!r})")
        elif item["kind"] == "sho":
            omega = item["omega"]
            if not isinstance(omega, (int, float)) or not math.isfinite(omega) or omega <= 0:
                raise ValueError("SHO frequency must be finite and positive")
            calls.append(f"ba.BasisSHO({dof!r}, omega={omega!r}, nbas={item['nbas']!r})")
        else:
            raise ValueError(f"unsupported basis kind: {item['kind']}")
    return "basis = [\n        " + ",\n        ".join(calls) + "\n    ]"
