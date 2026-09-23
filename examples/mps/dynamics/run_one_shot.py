#!/usr/bin/env python
"""Dry-run native MPS dynamics example; scientific choices remain here."""

from __future__ import annotations

import math
import os
from pathlib import Path

import yaml
from aiida import load_profile, orm
from renormalizer.model import Model, Op, Phonon
from renormalizer.model.basis import BasisSHO, BasisSimpleElectron
from renormalizer.utils import CompressConfig, EvolveConfig, Quantity
from renormalizer.utils.configs import CompressCriteria, EvolveMethod

from aiida_renormalizer.cases.mps_dynamics.api import (
    assemble_script,
    record_evolution,
    record_initial_state,
    record_model,
)
from aiida_renormalizer.cases.mps_dynamics.artifacts import write_dry_run

SMOKE_TEST = os.getenv("AIIDA_RENO_MPS_DYNAMICS_SMOKE", "0").lower() not in {
    "0",
    "false",
    "no",
    "off",
}

# std.yaml remains the authoritative physical input; frozen source needs no sidecar.
PARAM_FILE = Path(__file__).with_name("std.yaml")


def main(output_dir=None):
    load_profile()
    parameter_file = orm.SinglefileData(file=str(PARAM_FILE))
    parameters = yaml.safe_load(parameter_file.get_content())
    molecule_count = int(parameters["mol num"])
    transfer_integral = Quantity(*parameters["j constant"])
    temperature = Quantity(*parameters["temperature"])
    phonon_modes = [
        (Quantity(*omega), Quantity(*displacement))
        for omega, displacement in parameters["ph modes"]
    ]
    dt = float(parameters["evolve dt"])
    nsteps = parameters.get("evolve nsteps")
    evolve_time = parameters.get("evolve time")
    if dt <= 0 or not math.isfinite(dt):
        raise ValueError("evolve dt must be finite and positive")
    if nsteps is None:
        # Preserve Reno dynamics' original floor(duration/dt)+1 duration rule.
        if evolve_time is None or not math.isfinite(evolve_time) or evolve_time < 0:
            raise ValueError("evolve time must be finite and nonnegative")
        nsteps = int(evolve_time // dt) + 1
    if SMOKE_TEST:
        molecule_count, dt, nsteps = 3, 1.0, 1
    evolve_config = EvolveConfig(
        method=EvolveMethod.tdvp_ps,
        adaptive=True,
        guess_dt=1.0 if SMOKE_TEST else 2.0,
        adaptive_rtol=5e-4,
    )
    compress_config = CompressConfig(
        criteria=CompressCriteria.threshold, threshold=1e-3, max_bonddim=4 if SMOKE_TEST else 16
    )
    if isinstance(nsteps, bool) or not isinstance(nsteps, int) or nsteps < 0:
        raise ValueError("evolve nsteps must be a nonnegative integer")
    if molecule_count < 2:
        raise ValueError("this transport chain requires at least two molecules")
    phonons = [
        Phonon.simplest_phonon(
            omega=omega, displacement=displacement, temperature=temperature, lam=False, max_pdim=128
        )
        for omega, displacement in phonon_modes
    ]
    basis, hamiltonian = [], []
    for site in range(molecule_count):
        basis.append(BasisSimpleElectron(dof=site))
        reorganization = sum(0.5 * ph.omega[0] ** 2 * ph.dis[1] ** 2 for ph in phonons)
        hamiltonian.append(Op(symbol=r"a^\dagger a", dof=site, factor=reorganization))
        for mode, phonon in enumerate(phonons):
            dof, omega, displacement = (site, mode), phonon.omega[0], phonon.dis[1]
            basis.append(BasisSHO(dof=dof, omega=omega, nbas=phonon.n_phys_dim))
            hamiltonian += [
                Op(symbol="p^2", dof=dof, factor=0.5),
                Op(symbol="x^2", dof=dof, factor=0.5 * omega**2),
                Op(symbol=r"a^\dagger a", dof=site)
                * Op(symbol="x", dof=dof, factor=-(omega**2) * displacement),
            ]
        if site + 1 < molecule_count:
            hamiltonian += [
                Op(symbol=r"a^\dagger a", dof=[site, site + 1], factor=transfer_integral.as_au()),
                Op(symbol=r"a^\dagger a", dof=[site + 1, site], factor=transfer_integral.as_au()),
            ]
    model = Model(basis=basis, ham_terms=hamiltonian)
    model_section = record_model(model, source_file=parameter_file)
    center = molecule_count // 2
    initial = record_initial_state(
        temperature=temperature,
        thermal_steps=max(20, len(basis)),
        electron_site=center,
        relaxed_phonons=[((center, mode), phonon) for mode, phonon in enumerate(phonons)],
        evolve_config=evolve_config,
        compress_config=compress_config,
        expand_bonds=True,
        expansion_coefficient=1e-10,
        include_ex=True,
        subtract_initial_energy=True,
        exact_ground_thermal=True,
    )
    evolution = record_evolution(
        evolve_config=evolve_config,
        compress_config=compress_config,
        dt=dt,
        nsteps=nsteps,
        observe_initial=True,
        rdm=False,
        momentum=False,
        normalize=True,
        edge_site=0,
        edge_threshold=1e-4,
    )
    script = assemble_script(
        model=model_section,
        initial_state=initial,
        evolution=evolution,
        run_metadata=orm.Dict(
            dict={
                "case": "mps_dynamics",
                "smoke": SMOKE_TEST,
                "excitation": "relaxed",
                "coordinate": "electron-chain index",
                "restart": "replay-only",
            }
        ),
    )
    output_dir = (
        Path(output_dir) if output_dir is not None else Path(__file__).parent / "generated_scripts"
    )
    path = write_dry_run(script, output_dir)
    print(f"Standalone Reno script: {path}")
    return path


if __name__ == "__main__":
    main()
