#!/usr/bin/env python
"""Dry-run native MPS transport_kubo example; scientific choices remain here."""

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

from aiida_renormalizer.cases.mps_transport_kubo.api import (
    assemble_script,
    record_evolution,
    record_initial_state,
    record_model,
)
from aiida_renormalizer.cases.mps_transport_kubo.artifacts import write_dry_run

SMOKE_TEST = os.getenv("AIIDA_RENO_MPS_TRANSPORT_KUBO_SMOKE", "0").lower() not in {
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
        if evolve_time is None or not math.isfinite(evolve_time) or evolve_time < 0:
            raise ValueError("evolve time must be finite and nonnegative")
        nsteps = round(evolve_time / dt)
        if not math.isclose(nsteps * dt, evolve_time, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Kubo evolve time must be an integer multiple of evolve dt")
    elif evolve_time is not None and not math.isclose(nsteps * dt, evolve_time):
        raise ValueError("evolve nsteps and evolve time disagree")
    if SMOKE_TEST:
        molecule_count, temperature, dt, nsteps = 2, Quantity(100000, "K"), 0.1, 1
    evolve_config = EvolveConfig(
        method=EvolveMethod.prop_and_compress, adaptive=True, guess_dt=0.1 if SMOKE_TEST else 2.0
    )
    imaginary_config = EvolveConfig(
        method=EvolveMethod.prop_and_compress, adaptive=True, guess_dt=temperature.to_beta() / 1000j
    )
    compress_config = CompressConfig(
        criteria=CompressCriteria.threshold, threshold=1e-2 if SMOKE_TEST else 1e-4
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
    basis, hamiltonian, current = [], [], []
    distances = [[i - j for j in range(molecule_count)] for i in range(molecule_count)]
    distances[0][-1], distances[-1][0] = 1, -1
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
            for origin, destination in [(site, site + 1), (site + 1, site)]:
                hopping = Op(
                    symbol=r"a^\dagger a",
                    dof=[origin, destination],
                    factor=transfer_integral.as_au(),
                )
                hamiltonian.append(hopping)
                # Omit -i here; the runtime restores its squared minus sign in J(t)J(0).
                current.append(hopping * distances[origin][destination])
    model = Model(basis=basis, ham_terms=hamiltonian)
    model_section = record_model(model, source_file=parameter_file)
    initial = record_initial_state(
        temperature=temperature,
        thermal_steps=1,
        thermal_sector="one_electron",
        imaginary_config=imaginary_config,
        evolve_config=evolve_config,
        compress_config=compress_config,
        current_terms=[current],
        subtract_initial_energy=True,
    )
    evolution = record_evolution(
        evolve_config=evolve_config,
        compress_config=compress_config,
        temperature=temperature,
        dt=dt,
        nsteps=nsteps,
        observe_initial=True,
        correlation_prefactor=-1.0,  # The explicit real current generator omits -i.
        stop_at_tail=True,
        tail_window=10,
        tail_relative_tolerance=1e-5,
        smoke=SMOKE_TEST,
    )
    script = assemble_script(
        model=model_section,
        initial_state=initial,
        evolution=evolution,
        run_metadata=orm.Dict(
            dict={
                "case": "mps_transport_kubo",
                "smoke": SMOKE_TEST,
                "temperature_K": temperature.as_au() / Quantity(1.0, "K").as_au(),
                "current": "real generator omitting -i",
                "distance": "chain index with signed boundary entries",
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
