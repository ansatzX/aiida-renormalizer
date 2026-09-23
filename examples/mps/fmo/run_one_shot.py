#!/usr/bin/env python
"""Dry-run native MPS fmo example; scientific choices remain here."""

from __future__ import annotations

import json
import os
from pathlib import Path

from aiida import load_profile, orm
from renormalizer.model import Model, Op, Phonon
from renormalizer.model.basis import BasisSHO, BasisSimpleElectron
from renormalizer.mps.backend import np
from renormalizer.utils import CompressConfig, EvolveConfig, Quantity
from renormalizer.utils.configs import CompressCriteria, EvolveMethod
from renormalizer.utils.constant import cm2au

from aiida_renormalizer.cases.mps_fmo.api import (
    assemble_script,
    record_evolution,
    record_initial_state,
    record_model,
)
from aiida_renormalizer.cases.mps_fmo.artifacts import write_dry_run

SMOKE_TEST = os.getenv("AIIDA_RENO_MPS_FMO_SMOKE", "0").lower() not in {"0", "false", "no", "off"}


N_PHONONS = 2 if SMOKE_TEST else 35  # Uniform frequency samples between 2 and 300 cm^-1.
TOTAL_HR = 0.42  # Sum of Huang-Rhys factors for each molecule.
ACTIVE_SITE_ORDER = [7, 5, 3, 1, 2, 4, 6]  # Physical site 8 is excluded.
J_MATRIX_CM = [
    [310, -98, 6, -6, 7, -12, -10, 38],
    [-98, 230, 30, 7, 2, 12, 5, 8],
    [6, 30, 0, -59, -2, -10, 5, 2],
    [-6, 7, -59, 180, -65, -17, -65, -2],
    [7, 2, -2, -65, 405, 89, -6, 5],
    [-12, 11, -10, -17, 89, 320, 32, -10],
    [-10, 5, 5, -64, -6, 32, 270, -11],
    [38, 8, 2, -2, 5, -10, -11, 505],
]
DT, NSTEPS = (1.0, 1) if SMOKE_TEST else (160.0, 250)
EVOLVE_CONFIG = EvolveConfig(method=EvolveMethod.tdvp_ps, guess_dt=DT)
COMPRESS_CONFIG = CompressConfig(
    criteria=CompressCriteria.fixed, max_bonddim=4 if SMOKE_TEST else 32
)


def main(output_dir=None):
    load_profile()
    sdf = orm.SinglefileData(file=str(Path(__file__).with_name("fmo_sdf.json")))
    values = np.asarray(json.loads(sdf.get_content()), dtype=float)
    if values.ndim != 2 or values.shape[1] != 2 or np.any(np.diff(values[:, 0]) < 0):
        raise ValueError("SDF must be sorted [frequency_cm^-1, relative_Huang-Rhys_weight] rows")
    frequencies_cm = np.linspace(2.0, 300.0, N_PHONONS)
    hr_weights = np.interp(frequencies_cm, values[:, 0], values[:, 1])
    if hr_weights.sum() <= 0:
        raise ValueError("SDF must have positive interpolated weight")
    hr_weights *= TOTAL_HR / hr_weights.sum()
    phonons = [
        Phonon.simplest_phonon(
            omega=Quantity(float(omega)), displacement=Quantity(float(hr * omega)), lam=True
        )
        for omega, hr in zip(frequencies_cm * cm2au, hr_weights)
    ]
    raw_coupling = np.asarray(J_MATRIX_CM, dtype=float)
    # Preserve the deliberate pair-average correction to asymmetric upstream entries.
    coupling = 0.5 * (raw_coupling + raw_coupling.T) * cm2au
    order = np.asarray(ACTIVE_SITE_ORDER) - 1
    coupling = coupling[order][:, order]
    basis, hamiltonian = [], []
    for site in range(len(ACTIVE_SITE_ORDER)):
        basis.append(BasisSimpleElectron(dof=site))
        reorganization = sum(0.5 * ph.omega[0] ** 2 * ph.dis[1] ** 2 for ph in phonons)
        hamiltonian.append(
            Op(symbol=r"a^\dagger a", dof=site, factor=float(coupling[site, site]) + reorganization)
        )
        for other in range(len(ACTIVE_SITE_ORDER)):
            if other != site:
                hamiltonian.append(
                    Op(
                        symbol=r"a^\dagger a",
                        dof=[site, other],
                        factor=float(coupling[site, other]),
                    )
                )
        for mode, phonon in enumerate(phonons):
            dof, omega, displacement = (site, mode), phonon.omega[0], phonon.dis[1]
            basis.append(BasisSHO(dof=dof, omega=omega, nbas=phonon.n_phys_dim))
            hamiltonian += [
                Op(symbol="p^2", dof=dof, factor=0.5),
                Op(symbol="x^2", dof=dof, factor=0.5 * omega**2),
                Op(symbol=r"a^\dagger a", dof=site)
                * Op(symbol="x", dof=dof, factor=-(omega**2) * displacement),
            ]
    model = Model(basis=basis, ham_terms=hamiltonian)
    model_section = record_model(model, source_file=sdf)
    initial = record_initial_state(
        temperature=Quantity(0.0, "K"),
        thermal_steps=max(20, len(basis)),
        electron_site=len(ACTIVE_SITE_ORDER) // 2,
        relaxed_phonons=[],
        evolve_config=EVOLVE_CONFIG,
        compress_config=COMPRESS_CONFIG,
        expand_bonds=True,
        expansion_coefficient=1e-10,
        include_ex=True,
        subtract_initial_energy=True,
        exact_ground_thermal=True,
    )
    evolution = record_evolution(
        evolve_config=EVOLVE_CONFIG,
        compress_config=COMPRESS_CONFIG,
        dt=DT,
        nsteps=NSTEPS,
        observe_initial=True,
        rdm=False,
        momentum=False,
        normalize=True,
        edge_site=None,
        edge_threshold=1e-4,
    )
    script = assemble_script(
        model=model_section,
        initial_state=initial,
        evolution=evolution,
        run_metadata=orm.Dict(
            dict={
                "case": "mps_fmo",
                "smoke": SMOKE_TEST,
                "active_physical_sites": ACTIVE_SITE_ORDER,
                "excitation": "Franck-Condon",
                "coupling_policy": "symmetrize-pair-average",
                "total_hr": TOTAL_HR,
                "coordinate": "electron-chain index, not a physical distance",
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
