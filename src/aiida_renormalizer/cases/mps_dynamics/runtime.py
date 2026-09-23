"""Standalone native Reno state preparation and same-process segment operations."""

from __future__ import annotations

# Imports also serve the native constructors appended by the case renderer.
# ruff: noqa: F401
import json
import math
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

from renormalizer.model import Model, Op, Phonon
from renormalizer.model.basis import BasisSHO, BasisSimpleElectron
from renormalizer.mps import MpDm, Mpo, Mps, ThermalProp
from renormalizer.mps.backend import np
from renormalizer.utils import CompressConfig, EvolveConfig, Quantity
from renormalizer.utils.configs import OFS, BondDimDistri, CompressCriteria, EvolveMethod

RESULT_NAME = "dynamics_result.json"


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


class GroundThermalModel(Model):
    """Supply native exact-GS propagator metadata for the explicit scheme-3 Model.

    Mpo.exact_propagator iterates molecules and reads oscillator omega/pbond.
    It does not accept a plain Model even though all operators are explicit.
    No Hamiltonian or displaced-oscillator choice is inferred here.
    """

    scheme = 3

    def __init__(self, basis, ham_terms):
        super().__init__(basis=basis, ham_terms=ham_terms)
        molecules = []
        for primitive in basis:
            if isinstance(primitive, BasisSimpleElectron):
                molecules.append(SimpleNamespace(ph_list=[]))
            elif isinstance(primitive, BasisSHO) and molecules:
                molecules[-1].ph_list.append(
                    SimpleNamespace(omega=[primitive.omega, primitive.omega], pbond=primitive.nbas)
                )
            else:
                raise ValueError("exact GS thermal route requires electron-then-local-SHO layout")
        self._molecules = molecules
        self.mol_num = self.n_edofs

    def __iter__(self):
        return iter(self._molecules)

    def copy(self):
        model = type(self)(self.basis.copy(), self.ham_terms)
        model.mpos = self.mpos.copy()
        return model


def make_ground_state(model, *, temperature, thermal_steps, exact_ground_thermal):
    """Return a live native MPS at T=0 or exact ground-sector thermal MPDM."""
    if temperature.as_au() == 0:
        return Mps.ground_state(model=model, max_entangled=False)
    if temperature.as_au() < 0 or thermal_steps < 1:
        raise ValueError("positive temperature and thermal_steps are required")
    if not exact_ground_thermal:
        raise ValueError("this case implements only the exact ground-sector thermal route")
    thermal_model = GroundThermalModel(model.basis, model.ham_terms)
    state = MpDm.max_entangled_gs(thermal_model)
    thermal = ThermalProp(init_mpdm=state, exact=True, space="GS")
    thermal.evolve(evolve_dt=None, nsteps=thermal_steps, evolve_time=temperature.to_beta() / 2j)
    return thermal.latest_mps


def excite_electron(state, *, site, relaxed_phonons):
    """Apply explicitly selected displaced local modes, then create an electron.

    relaxed_phonons is a sequence of (native phonon dof, native Phonon).
    Passing an empty sequence selects the Franck-Condon excitation.
    """
    if relaxed_phonons and any(dim != 1 for dim in state.bond_dims):
        raise ValueError("relaxed excitation requires an unentangled ground-sector state")
    state = state.copy()
    for dof, phonon in relaxed_phonons:
        index = state.model.dof_to_siteidx[dof]
        tensor = state[index].array
        physical = tensor[0, ..., 0]
        displaced = phonon.get_displacement_evecs().dot(physical)
        state[index] = displaced.reshape(tensor.shape)
    creation = Mpo.onsite(model=state.model, opera=r"a^\dagger", dof_set={site})
    return creation.apply(state)


def prepare_dynamics(
    state,
    *,
    evolve_config,
    compress_config,
    expand_bonds,
    expansion_coefficient,
    include_ex,
    subtract_initial_energy,
):
    """Shift by the initial energy, configure the live state, and optionally expand."""
    energy = state.expectation(Mpo(model=state.model))
    hamiltonian = Mpo(
        model=state.model, offset=Quantity(energy if subtract_initial_energy else 0.0)
    )
    state.evolve_config = deepcopy(evolve_config)
    state.compress_config = deepcopy(compress_config)
    if expand_bonds:
        state = state.expand_bond_dimension(
            hint_mpo=hamiltonian, coef=expansion_coefficient, include_ex=include_ex
        )
    state.canonicalise()
    return state, hamiltonian, energy


def observe_state(state, hamiltonian, *, time, step, rdm, momentum):
    occupations = state.e_occupations
    positions = np.arange(len(occupations))
    variance = (
        0.0
        if np.allclose(occupations, 0)
        else float(
            np.average(positions**2, weights=occupations)
            - np.average(positions, weights=occupations) ** 2
        )
    )
    row = {
        "step": step,
        "time": time,
        "energy": state.expectation(hamiltonian),
        "electron_occupations": occupations,
        "phonon_occupations": state.ph_occupations,
        "r_square": variance,
        "bond_entropy": state.calc_bond_entropy(),
    }
    if rdm:
        reduced = state.calc_edof_rdm()
        eigenvalues = np.linalg.eigvalsh(reduced)
        positive = eigenvalues[eigenvalues > 0]
        row.update(
            electron_rdm=reduced,
            eph_entropy=float(-np.sum(positive * np.log(positive))),
            coherent_length=float(np.abs(reduced).sum() - np.trace(reduced).real),
        )
        if momentum:
            n = len(occupations)
            transform = np.exp(
                -1j * (np.arange(-n, n, 2) / n * np.pi)[:, None] * np.arange(n)[None, :]
            ) / np.sqrt(n)
            row["k_occupations"] = np.diag(transform @ reduced @ transform.conj().T).real
    return row


def evolve_segment(
    state,
    hamiltonian,
    *,
    dt,
    nsteps,
    evolve_config,
    compress_config,
    observe_initial=True,
    rdm=False,
    momentum=False,
    normalize=True,
    edge_site=None,
    edge_threshold=1e-4,
    start_time=0.0,
    start_step=0,
):
    """Return the actual final MPS/MPDM and rows for further same-process phases.

    The caller supplies offsets when continuing. Copy the state before branching;
    the native evolution and configuration objects may be mutated by Reno.
    """
    validate_segment(dt, nsteps, start_time, start_step)
    if momentum and not rdm:
        raise ValueError("momentum occupations require rdm=True")
    if state.model.basis != hamiltonian.model.basis:
        raise ValueError("state and Hamiltonian must use compatible native bases")
    if edge_site is not None and not 0 <= edge_site < state.model.n_edofs:
        raise ValueError("edge_site is outside the electronic chain")
    if nsteps:
        state.evolve_config = deepcopy(evolve_config)
        state.compress_config = deepcopy(compress_config)
    rows = []

    def observe(step):
        rows.append(
            observe_state(
                state,
                hamiltonian,
                time=start_time + step * dt,
                step=start_step + step,
                rdm=rdm,
                momentum=momentum,
            )
        )

    if observe_initial:
        observe(0)
    for step in range(1, nsteps + 1):
        if edge_site is not None and state.e_occupations[edge_site] > edge_threshold:
            break
        state = state.evolve(hamiltonian, dt, normalize=normalize)
        observe(step)
    return state, rows


def summarize_dynamics(state, rows, *, initial_energy, planned_steps):
    """Keep scalar series accessible while retaining rows for continuation analysis."""
    completed = rows[-1]["step"] if rows else 0
    result = {
        "initial_energy": initial_energy,
        "rows": rows,
        "time_series": [row["time"] for row in rows],
        "final_time": rows[-1]["time"] if rows else 0.0,
        "model": {
            "basis_count": len(state.model.basis),
            "hamiltonian_term_count": len(state.model.ham_terms),
            "mol_num": state.model.n_edofs,
        },
        "execution_runtime": {
            "planned_steps": planned_steps,
            "completed_steps": completed,
            "termination_reason": "left_edge_population_threshold"
            if completed < planned_steps
            else "requested_steps_completed",
        },
        "restart_plan": {"mode": "replay-only", "checkpoint_restart": False},
    }
    series = {
        "energies": "energy",
        "r_square": "r_square",
        "e_occupations": "electron_occupations",
        "ph_occupations": "phonon_occupations",
        "bond_entropy": "bond_entropy",
        "reduced_density_matrices": "electron_rdm",
        "k_occupations": "k_occupations",
        "eph_entropy": "eph_entropy",
        "coherent_length": "coherent_length",
    }
    for name, key in series.items():
        if rows and key in rows[0]:
            result[name] = [row[key] for row in rows]
    return result
