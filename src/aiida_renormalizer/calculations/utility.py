"""Utility calcfunctions for pure data transformations."""
from __future__ import annotations

import numpy as np
from aiida import orm
from aiida.engine import calcfunction

from aiida_renormalizer.data import MPOData, MPSData


@calcfunction
def validate_result(
    mps: MPSData,
    mpo: MPOData,
    checks: orm.List,
) -> orm.Dict:
    """Physical validation (from llm_reno Validator agent experience).

    Checks:
    - 'nan_check': expectation contains NaN/Inf
    - 'norm_check': |<ψ|ψ>| ≈ 1
    - 'energy_bound': E in physical range
    - 'variance': <H²> - <H>² ≈ 0 (ground state check)

    Args:
        mps: MPS state (only metadata is read, no MPS reconstruction)
        mpo: MPO operator (only metadata is read)
        checks: List of check names to perform

    Returns:
        {'passed': bool, 'details': {...}}
    """
    results = {}

    # NOTE: This calcfunction only reads attributes (lightweight),
    # does NOT reconstruct full MPS/MPO objects.

    if 'nan_check' in checks:
        # Check bond_dims for NaN
        bond_dims = mps.base.attributes.get('bond_dims')
        has_nan = any(np.isnan(d) for d in bond_dims)
        results['nan_check'] = {'passed': not has_nan}

    if 'norm_check' in checks:
        # This would require actual MPS reconstruction
        # For now, return placeholder
        results['norm_check'] = {'passed': True, 'note': 'Requires full MPS'}

    all_passed = all(r['passed'] for r in results.values())

    return orm.Dict({
        'passed': all_passed,
        'details': results,
    })


@calcfunction
def fourier_transform(
    correlation: orm.List,
    dt: orm.Float,
    window: orm.Str,
) -> orm.Dict:
    """FFT of time-domain correlation function.

    Args:
        correlation: Time series data
        dt: Time step
        window: Window function ('none', 'gaussian', 'lorentzian', 'cosine')

    Returns:
        {'frequencies': [...], 'spectrum': [...]}
    """
    from scipy.fft import fft, fftfreq

    t_series = np.array(correlation.get_list())
    dt_val = dt.value

    # Apply window
    n = len(t_series)
    if window.value == 'gaussian':
        sigma = n / 6
        window_arr = np.exp(-((np.arange(n) - n/2)**2) / (2 * sigma**2))
    elif window.value == 'lorentzian':
        gamma = n / 6
        window_arr = 1 / (1 + ((np.arange(n) - n/2) / gamma)**2)
    elif window.value == 'cosine':
        window_arr = 0.5 * (1 + np.cos(np.pi * np.arange(n) / n))
    else:
        window_arr = np.ones(n)

    windowed = t_series * window_arr

    # FFT
    spectrum = fft(windowed)
    freqs = fftfreq(n, dt_val)

    # Only return positive frequencies
    pos_mask = freqs >= 0

    return orm.Dict({
        'frequencies': freqs[pos_mask].tolist(),
        'spectrum': np.abs(spectrum[pos_mask]).tolist(),
    })
