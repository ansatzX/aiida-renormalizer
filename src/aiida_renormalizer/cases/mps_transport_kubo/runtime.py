"""Standalone native Reno state preparation and same-process segment operations."""

from __future__ import annotations

# Imports also serve the native constructors appended by the case renderer.
# ruff: noqa: F401
import json
import math
from copy import deepcopy
from pathlib import Path

from renormalizer.model import Model, Op, Phonon
from renormalizer.model.basis import BasisSHO, BasisSimpleElectron
from renormalizer.mps import MpDm, Mpo, Mps, ThermalProp
from renormalizer.mps.backend import np
from renormalizer.mps.mps import BraKetPair
from renormalizer.utils import CompressConfig, EvolveConfig, Quantity
from renormalizer.utils.configs import OFS, BondDimDistri, CompressCriteria, EvolveMethod
from renormalizer.utils.constant import mobility2au

RESULT_NAME = "transport_kubo_result.json"


def plain(value):
    if isinstance(value, complex):
        return {"real": value.real, "imag": value.imag}
    if hasattr(value, "tolist"):
        return plain(value.tolist())
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def validate_segment(dt, nsteps, start_time, start_step):
    if isinstance(nsteps, bool) or not isinstance(nsteps, int) or nsteps < 0:
        raise ValueError("nsteps must be a nonnegative integer")
    if isinstance(start_step, bool) or not isinstance(start_step, int) or start_step < 0:
        raise ValueError("start_step must be a nonnegative integer")
    if not math.isfinite(float(start_time)):
        raise ValueError("start_time must be finite")
    if not math.isfinite(float(dt)) or dt <= 0:
        raise ValueError("dt must be finite and positive")


def make_thermal_state(
    model, *, temperature, thermal_steps, imaginary_config, compress_config, thermal_sector
):
    """Return the live native one-electron thermal MPDM; no checkpoint/cache path."""
    if thermal_sector != "one_electron":
        raise ValueError("this case supports only thermal_sector=one_electron")
    if temperature.as_au() <= 0 or thermal_steps < 1:
        raise ValueError("Kubo thermal state requires positive temperature and thermal_steps")
    state = MpDm.max_entangled_ex(model)
    state.compress_config = deepcopy(compress_config)
    thermal = ThermalProp(init_mpdm=state, evolve_config=deepcopy(imaginary_config))
    thermal.evolve(evolve_dt=None, nsteps=thermal_steps, evolve_time=temperature.to_beta() / 2j)
    return thermal.latest_mps


def prepare_correlation(
    thermal_state, *, current_terms, evolve_config, compress_config, subtract_initial_energy
):
    """Return live (bra, kets), shifted Hamiltonian, native currents and offset."""
    if not current_terms or any(not group for group in current_terms):
        raise ValueError("each current component must contain native Op terms")
    energy = thermal_state.expectation(Mpo(model=thermal_state.model))
    hamiltonian = Mpo(
        model=thermal_state.model, offset=Quantity(energy if subtract_initial_energy else 0.0)
    )
    currents = [Mpo(model=thermal_state.model, terms=group) for group in current_terms]
    thermal_state.evolve_config = deepcopy(evolve_config)
    thermal_state.compress_config = deepcopy(compress_config)
    bra = thermal_state.copy()
    kets = [current.contract(thermal_state).normalize("mps_norm_to_coeff") for current in currents]
    return (bra, kets), hamiltonian, currents, energy


def observe_correlation(state, currents, *, time, step, correlation_prefactor):
    bra, kets = state
    # The caller supplies the phase convention for its explicit current generators.
    components = [
        correlation_prefactor * BraKetPair(bra, ket, current).ft
        for current in currents
        for ket in kets
    ]
    return {
        "time": time,
        "step": step,
        "auto_correlation": sum(components),
        "current_components": components,
    }


def tail_converged(rows, *, window, relative_tolerance):
    if len(rows) < window:
        return False
    correlations = np.asarray([row["auto_correlation"] for row in rows])
    threshold = relative_tolerance * abs(correlations[0])
    tail = correlations[-window:]
    return bool(abs(tail.mean()) < threshold and tail.std() < threshold)


def evolve_segment(
    state,
    hamiltonian,
    currents,
    *,
    dt,
    nsteps,
    evolve_config,
    compress_config,
    correlation_prefactor,
    observe_initial=True,
    stop_at_tail=True,
    tail_window=10,
    tail_relative_tolerance=1e-5,
    start_time=0.0,
    start_step=0,
    history=(),
):
    """Continue native bra/ket states and retain full history for the tail criterion.

    Pass the previous returned rows as history and observe_initial=False when
    continuing at its final time/step. Copy bra and each ket before branching.
    """
    validate_segment(dt, nsteps, start_time, start_step)
    if isinstance(correlation_prefactor, bool) or not math.isfinite(abs(correlation_prefactor)):
        raise ValueError("correlation_prefactor must be a finite scalar")
    bra, kets = state
    if not currents or len(kets) != len(currents):
        raise ValueError("each current needs its corresponding live ket")
    if isinstance(tail_window, bool) or not isinstance(tail_window, int) or tail_window < 1:
        raise ValueError("tail_window must be a positive integer")
    if not math.isfinite(tail_relative_tolerance) or tail_relative_tolerance <= 0:
        raise ValueError("tail_relative_tolerance must be finite and positive")
    if any(item.model.basis != hamiltonian.model.basis for item in [bra, *kets, *currents]):
        raise ValueError("bra, kets, currents and Hamiltonian require compatible bases")
    rows = list(history)
    if rows and (rows[-1]["time"] != start_time or rows[-1]["step"] != start_step):
        raise ValueError("history must end at the segment's start_time/start_step")
    if rows and observe_initial:
        raise ValueError("continuation history already contains the initial observation")
    if nsteps:
        for item in [bra, *kets]:
            item.evolve_config = deepcopy(evolve_config)
            item.compress_config = deepcopy(compress_config)
    if observe_initial:
        rows.append(
            observe_correlation(
                (bra, kets),
                currents,
                time=start_time,
                step=start_step,
                correlation_prefactor=correlation_prefactor,
            )
        )
    for step in range(1, nsteps + 1):
        if stop_at_tail and tail_converged(
            rows, window=tail_window, relative_tolerance=tail_relative_tolerance
        ):
            break
        kets = [ket.evolve(hamiltonian, dt) for ket in kets]
        bra = bra.evolve(hamiltonian, dt)
        rows.append(
            observe_correlation(
                (bra, kets),
                currents,
                time=start_time + step * dt,
                step=start_step + step,
                correlation_prefactor=correlation_prefactor,
            )
        )
    return (bra, kets), rows


def mobility_from_rows(rows, *, temperature):
    """Finite-time trapezoidal Kubo integral; convergence is evaluated separately."""
    times = [row["time"] for row in rows]
    values = [complex(row["auto_correlation"]).real for row in rows]
    integral = sum(
        (t1 - t0) * (v1 + v0) / 2 for t0, t1, v0, v1 in zip(times, times[1:], values, values[1:])
    )
    mobility_au = integral / temperature.as_au()
    return {"mobility_au": mobility_au, "mobility_cm2_per_Vs": mobility_au / mobility2au}
