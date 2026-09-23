"""CD bath algorithms with their selection and cutoffs supplied by the script."""

from __future__ import annotations

import math

import numpy as np
from aiida import orm
from aiida.engine import calcfunction
from renormalizer.sbm import ColeDavidsonSDF
from scipy.integrate import quad


@calcfunction
def discretize_cole_davidson_spectrum(
    ita: orm.Float,
    omega_c: orm.Float,
    beta: orm.Float,
    upper_limit: orm.Float,
    n_modes: orm.Int,
    method: orm.Str,
) -> orm.ArrayData:
    """Discretize the Cole-Davidson spectral density into oscillator modes.

    ``ita`` is the spectral-density prefactor; ``omega_c`` is its characteristic
    frequency and ``beta`` its dimensionless shape exponent. ``upper_limit``
    bounds the discretization interval. Wang1 uses 10^7 quadrature samples.
    Return recorded arrays ``omega_k`` (mode frequencies) and ``c_j2`` (squared
    bare coordinate-coupling amplitudes c_j, before the script applies the
    impurity-number factor of two and any thermofield transformation).
    """
    values = {
        "ita": ita.value,
        "omega_c": omega_c.value,
        "beta": beta.value,
        "upper_limit": upper_limit.value,
    }
    if any(not math.isfinite(value) or value <= 0 for value in values.values()):
        raise ValueError("CD parameters must be finite and positive")
    if n_modes.value < 2:
        raise ValueError("this binary TTN case requires at least two bath modes")
    if method.value != "Wang1":
        raise ValueError("this case currently supports CD discretization method Wang1 only")
    sdf = ColeDavidsonSDF(values["ita"], values["omega_c"], values["beta"], values["upper_limit"])
    omega, coupling_squared = sdf.Wang1(n_modes.value)
    node = orm.ArrayData()
    node.set_array("omega_k", np.asarray(omega, dtype=float))
    node.set_array("c_j2", np.asarray(coupling_squared, dtype=float))
    node.base.attributes.set_many({**values, "n_modes": n_modes.value, "method": method.value})
    return node


@calcfunction
def cd_renormalization_factor(
    bath: orm.ArrayData, lower_cutoff: orm.Float, upper_cutoff: orm.Float
) -> orm.Float:
    """Return exp[-2/pi integral J(w)/w^2 dw] over explicit integration bounds."""
    lower, upper = lower_cutoff.value, upper_cutoff.value
    if not (math.isfinite(lower) and math.isfinite(upper) and 0 < lower < upper):
        raise ValueError("renormalization cutoffs must satisfy 0 < lower < upper")
    attrs = bath.base.attributes
    sdf = ColeDavidsonSDF(
        attrs.get("ita"), attrs.get("omega_c"), attrs.get("beta"), attrs.get("upper_limit")
    )
    integral, _ = quad(lambda omega: sdf.func(omega) / omega**2, lower, upper)
    result = orm.Float(float(np.exp(-2 * integral / np.pi)))
    result.base.attributes.set_many({"lower_cutoff": lower, "upper_cutoff": upper})
    return result
