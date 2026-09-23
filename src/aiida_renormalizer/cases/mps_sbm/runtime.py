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


def evolve_segment(
    state,
    hamiltonian,
    *,
    evolve_config,
    dt,
    nsteps,
    normalize,
    observations,
    one_site_rdms=(),
    bond_entropy=False,
    electronic_rdm=False,
    observe_initial=True,
    observe_every=1,
    start_time=0.0,
    start_step=0,
):
    """Continue an MPS without rebuilding it; returned state is authoritative."""
    if len(state.model.basis) != len(hamiltonian.model.basis) or any(
        left is not right for left, right in zip(state.model.basis, hamiltonian.model.basis)
    ):
        raise ValueError("state and MPO must use the same ordered native basis objects")
    if (
        not isinstance(nsteps, int)
        or nsteps < 0
        or not isinstance(observe_every, int)
        or observe_every < 1
    ):
        raise ValueError("invalid segment step count or observation cadence")
    evolve_config.check_valid_dt(dt)
    rows = []

    def observe(step):
        row = {"step": start_step + step, "time": start_time + step * dt}
        row.update(
            observe_state(
                state,
                observations,
                one_site_rdms=one_site_rdms,
                bond_entropy=bond_entropy,
                electronic_rdm=electronic_rdm,
            )
        )
        rows.append(row)

    if observe_initial:
        observe(0)
    if nsteps:
        state.evolve_config = evolve_config
    for step in range(1, nsteps + 1):
        state = state.evolve(hamiltonian, dt, normalize=normalize)
        if step % observe_every == 0:
            observe(step)
    return state, rows
