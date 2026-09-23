"""Recorded native Ohmic quadrature, with adiabatic-renormalization inputs explicit."""

import numpy as np
from aiida import orm
from aiida.engine import calcfunction
from renormalizer.sbm import SpectralDensityFunction
from renormalizer.utils import Quantity


@calcfunction
def discretize_ohmic_spectrum(
    alpha: orm.Float,
    cutoff_frequency: orm.Float,
    bare_delta: orm.Float,
    renormalization_p: orm.Float,
    n_modes: orm.Int,
    sort_modes: orm.Bool,
) -> orm.ArrayData:
    if bare_delta.value <= 0 or n_modes.value < 1:
        raise ValueError("positive bare delta and mode count required")
    spectrum = SpectralDensityFunction(alpha.value, Quantity(cutoff_frequency.value), s=1)
    renormalized_delta, max_frequency = spectrum.adiabatic_renormalization(
        Quantity(bare_delta.value), renormalization_p.value
    )
    frequencies, coupling_squared = spectrum.trapz(n_modes.value, 0.0, max_frequency)
    frequencies, displacements = spectrum.post_process(
        frequencies, coupling_squared, ifsort=sort_modes.value
    )
    result = orm.ArrayData()
    result.set_array("frequencies", np.asarray([v.as_au() for v in frequencies]))
    result.set_array("displacements", np.asarray([v.as_au() for v in displacements]))
    result.base.attributes.set(
        "renormalization_factor", float(renormalized_delta / bare_delta.value)
    )
    result.base.attributes.set("max_frequency", float(max_frequency))
    return result
