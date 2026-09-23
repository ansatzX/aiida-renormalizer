# ruff: noqa: F401
# Native imports below are also used by the recorded sections appended during rendering.
"""Standalone native MPS operations; state objects survive each calculation phase."""

from __future__ import annotations

import json
import math
from pathlib import Path

from renormalizer import Model, Mpo, Mps, optimize_mps
from renormalizer.model import Op
from renormalizer.model import basis as ba
from renormalizer.utils import CompressConfig, EvolveConfig, OptimizeConfig, configs


def real_value(value):
    value = complex(value)
    if not math.isfinite(value.real) or not math.isfinite(value.imag) or abs(value.imag) > 1e-9:
        raise ValueError(f"expected finite real expectation, got {value}")
    return value.real


def complex_matrix(matrix):
    return {"real": matrix.real.tolist(), "imag": matrix.imag.tolist()}


def observe_state(
    state, observations, *, one_site_rdms=(), bond_entropy=False, electronic_rdm=False
):
    values = {
        label: real_value(state.expectation(operator)) for label, operator in observations.items()
    }
    if one_site_rdms:
        values["one_site_rdms"] = {
            str(site): complex_matrix(rdm)
            for site, rdm in state.calc_1site_rdm(idx=list(one_site_rdms)).items()
        }
    if bond_entropy:
        values["bond_entropy"] = [float(value) for value in state.calc_entropy("bond")]
    if electronic_rdm:
        values["electronic_rdm"] = complex_matrix(state.calc_edof_rdm())
    return values


def optimize_state(state, hamiltonian, *, optimize_config, copy_initial_state):
    """Optimize the supplied MPS; the caller owns any branch copy decision."""
    if len(state.model.basis) != len(hamiltonian.model.basis) or any(
        left is not right for left, right in zip(state.model.basis, hamiltonian.model.basis)
    ):
        raise ValueError("state and MPO must use the same ordered native basis objects")
    if optimize_config.nroots != 1 or not optimize_config.procedure:
        raise ValueError("single-root nonempty optimization procedure required")
    working_state = state.copy() if copy_initial_state else state
    working_state.optimize_config = optimize_config
    energies, final_state = optimize_mps(working_state, hamiltonian)
    energies = [real_value(energy) for energy in energies]
    return final_state, {
        "sweep_energies": energies,
        "lowest_sweep_energy": min(energies),
        "final_state_energy": real_value(final_state.expectation(hamiltonian)),
        "planned_sweeps": len(optimize_config.procedure),
        "completed_sweeps": len(energies),
    }
