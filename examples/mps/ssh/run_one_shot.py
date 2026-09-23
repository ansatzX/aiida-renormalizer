#!/usr/bin/env python
"""Single-polaron optical SSH chain with interleaved electron/phonon MPS sites.

The one-electron sector is intentional: this example does not supply the
Jordan-Wigner strings needed for a general many-fermion problem. Atomic units.
The launcher records a standalone DMRG calculation without running it.
"""

import os
from pathlib import Path

from aiida import load_profile
from renormalizer.model import Op
from renormalizer.model.basis import BasisSHO, BasisSimpleElectron
from renormalizer.utils import OptimizeConfig

from aiida_renormalizer.cases.mps_ssh import (
    assemble_script,
    record_dmrg,
    record_model,
    record_observations,
    record_random_state,
    write_dry_run,
)

# Preserve the existing small-run environment switch; ordinary execution only generates code.
SMOKE_TEST = os.getenv("AIIDA_RENO_MPS_SSH_SMOKE", "0").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

N_SITES = 2 if SMOKE_TEST else 4
HOPPING = -1.0
# Hopping modulation per dimensionless phonon displacement.
SSH_COUPLING = 0.1 if SMOKE_TEST else 0.7
PHONON_FREQUENCY = 0.5
N_PHONON_LEVELS = 2 if SMOKE_TEST else 4  # SHO basis size, including the vacuum.
MAX_BOND_DIMENSION = 4 if SMOKE_TEST else 16
N_SWEEPS = 2 if SMOKE_TEST else 10
PERIODIC = not SMOKE_TEST  # A two-site periodic chain would cancel the SSH coupling.


# Native Reno sweep rows: [bond dimension, sector exploration fraction].
SWEEP_PROCEDURE = (
    [[MAX_BOND_DIMENSION, 0.2]] + [[MAX_BOND_DIMENSION, 0.0] for _ in range(N_SWEEPS - 1)]
    if SMOKE_TEST
    else [
        [max(1, MAX_BOND_DIMENSION // 4), 0.4],
        [max(1, MAX_BOND_DIMENSION // 2), 0.2],
        [max(1, 3 * MAX_BOND_DIMENSION // 4), 0.1],
        *[[MAX_BOND_DIMENSION, 0.0] for _ in range(N_SWEEPS - 3)],
    ]
)


def main(output_dir=None):
    load_profile()
    if PERIODIC and N_SITES < 3:
        raise ValueError("periodic SSH requires at least three sites")
    bonds = [(site, site + 1) for site in range(N_SITES - 1)]
    if PERIODIC:
        bonds.append((N_SITES - 1, 0))
    hamiltonian = [
        Op(r"b^\dagger b", (site, 0), factor=PHONON_FREQUENCY) for site in range(N_SITES)
    ]
    for left, right in bonds:
        for source, target in ((left, right), (right, left)):
            hopping = Op(r"a^\dagger a", [source, target])
            hamiltonian.extend(
                [
                    HOPPING * hopping,
                    SSH_COUPLING * hopping * Op(r"b^\dagger+b", (right, 0)),
                    -SSH_COUPLING * hopping * Op(r"b^\dagger+b", (left, 0)),
                ]
            )
    basis = []
    for site in range(N_SITES):
        basis.extend(
            [
                BasisSimpleElectron(site),
                BasisSHO((site, 0), omega=PHONON_FREQUENCY, nbas=N_PHONON_LEVELS),
            ]
        )
    model = record_model(hamiltonian=hamiltonian, basis=basis)
    initial_state = record_random_state(
        quantum_number=1, bond_dimension=MAX_BOND_DIMENSION, percent=1.0
    )
    optimizer = OptimizeConfig(procedure=SWEEP_PROCEDURE)
    optimizer.method = "2site"
    dmrg = record_dmrg(optimize_config=optimizer, copy_initial_state=True)

    operators = {}
    for site in range(N_SITES):
        operators[f"phonon_occupation_{site}"] = Op(r"b^\dagger b", (site, 0))
        operators[f"phonon_displacement_{site}"] = Op(r"b^\dagger+b", (site, 0))
    for left in range(N_SITES):
        number_left = Op(r"a^\dagger a", [left, left])
        for right in range(N_SITES):
            # n_i^2 = n_i for a single electronic orbital.
            operators[f"n_{left}_n_{right}"] = (
                number_left if left == right else number_left * Op(r"a^\dagger a", [right, right])
            )
    observations = record_observations(operators=operators, electronic_rdm=True)
    script = assemble_script(model, initial_state, dmrg, observations)
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).with_name("generated_scripts")
    )
    return write_dry_run(script, output_dir)


if __name__ == "__main__":
    print(main())
