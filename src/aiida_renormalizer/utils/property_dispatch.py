"""PropertySpec dispatch to L1 CalcJobs."""
from __future__ import annotations

import typing as t

from aiida import orm

from aiida_renormalizer.data import ModelData, MpsData


def dispatch_observable(
    obs_spec: dict,
    mps: MpsData,
    model: ModelData,
) -> orm.Dict:
    """Dispatch an observable calculation to the appropriate L1 CalcJob.

    Args:
        obs_spec: {'name': str, ...additional params}
        mps: MPS state
        model: Model

    Returns:
        Result from the CalcJob (orm.Dict)
    """
    name = obs_spec['name']

    # Registry of observable names → CalcJob classes
    # (In real implementation, this would be a proper registry with entry points)
    dispatch_table = {
        'occupations': _dispatch_occupations,
        'msd': _dispatch_msd,
        'sigma_z': _dispatch_sigma_z,
        'sigma_x': _dispatch_sigma_x,
        'correlation': _dispatch_correlation,
        'entanglement_entropy': _dispatch_entropy,
        'rdm': _dispatch_rdm,
        'autocorrelation': _dispatch_autocorrelation,
        'energy': _dispatch_energy,
    }

    if name not in dispatch_table:
        raise ValueError(f"Unknown observable: {name}")

    return dispatch_table[name](obs_spec, mps, model)


def _dispatch_occupations(spec, mps, model):
    """Dispatch to ComputeOccupationsCalcJob."""
    from aiida.engine import submit
    from aiida_renormalizer.calculations.lego import ComputeOccupationsCalcJob

    # Extract parameters from spec
    dof_type = spec.get('dof_type', 'all')

    # Submit CalcJob (simplified — real version needs proper input construction)
    # For now, return placeholder
    return orm.Dict({
        'dof_names': ['placeholder'],
        'occupations': [0.0],
    })


def _dispatch_msd(spec, mps, model):
    """Dispatch to ComputeMsdCalcJob."""
    # Placeholder
    return orm.Dict({'msd': 0.0})


def _dispatch_sigma_z(spec, mps, model):
    """Dispatch to ComputeSigmaZCalcJob."""
    # Placeholder
    return orm.Dict({'sigma_z': 0.0})


def _dispatch_sigma_x(spec, mps, model):
    """Dispatch to ComputeSigmaXCalcJob."""
    # Placeholder
    return orm.Dict({'sigma_x': 0.0})


def _dispatch_correlation(spec, mps, model):
    """Dispatch to ComputeCorrelationCalcJob."""
    # Placeholder
    return orm.Dict({'correlation': []})


def _dispatch_entropy(spec, mps, model):
    """Dispatch to ComputeEntropyCalcJob."""
    # Placeholder
    return orm.Dict({'entropy': 0.0})


def _dispatch_rdm(spec, mps, model):
    """Dispatch to ComputeRdmCalcJob."""
    # Placeholder
    return orm.Dict({'rdm': []})


def _dispatch_autocorrelation(spec, mps, model):
    """Dispatch to AutocorrelationCalcJob."""
    # Placeholder
    return orm.Dict({'autocorrelation': []})


def _dispatch_energy(spec, mps, model):
    """Dispatch to ExpectationCalcJob for energy calculation."""
    # Placeholder
    return orm.Dict({'energy': 0.0})
