#!/usr/bin/env python
"""Open Hubbard chain: DMRG followed by imaginary-time refinement.

Reno '+' annihilates and '-' creates an electron. Spin orbitals are ordered
0-up, 0-down, 1-up, 1-down, ...; all energies and times use atomic units.
Running this launcher records and writes code, without executing the solver.
"""

import os
from pathlib import Path

from aiida import load_profile
from renormalizer.model import Op
from renormalizer.model.basis import BasisHalfSpin
from renormalizer.utils import EvolveConfig, EvolveMethod, OptimizeConfig

from aiida_renormalizer.cases.mps_hubbard import (
    assemble_script,
    record_dmrg,
    record_imaginary_time,
    record_model,
    record_observations,
    record_random_state,
    write_dry_run,
)

# Preserve the existing small-run environment switch; ordinary execution only generates code.
SMOKE_TEST = os.getenv("AIIDA_RENO_MPS_HUBBARD_SMOKE", "0").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

N_SITES = 3 if SMOKE_TEST else 10  # Number of spatial sites in the open chain.
HOPPING = -1.0  # Nearest-neighbor hopping coefficient.
ONSITE_REPULSION = 1.0 if SMOKE_TEST else 4.0  # Energy cost for double occupation.
N_ELECTRONS = [1, 1] if SMOKE_TEST else [5, 5]  # Conserved [spin-up, spin-down] electron counts.
MAX_BOND_DIMENSION = 16 if SMOKE_TEST else 100
MAX_IMAGINARY_STEPS = 2 if SMOKE_TEST else 1000
ENERGY_CHANGE_ATOL = 1e-5  # Stop refinement when consecutive energies agree.


def main(output_dir=None):
    load_profile()
    up = {"+": [-1, 0], "-": [1, 0], "Z": [0, 0]}
    down = {"+": [0, -1], "-": [0, 1], "Z": [0, 0]}
    hamiltonian = []
    for orbital in range(2 * (N_SITES - 1)):
        spin, opposite = (up, down) if orbital % 2 == 0 else (down, up)
        # Explicit Jordan-Wigner strings fix the fermionic sign convention.
        dofs = [orbital, orbital, orbital + 1, orbital + 2]
        hamiltonian.extend(
            [
                Op(
                    "Z + Z -",
                    dofs,
                    factor=HOPPING,
                    qn=[spin["Z"], spin["+"], opposite["Z"], spin["-"]],
                ),
                Op(
                    "Z - Z +",
                    dofs,
                    factor=-HOPPING,
                    qn=[spin["Z"], spin["-"], opposite["Z"], spin["+"]],
                ),
            ]
        )
    for orbital in range(0, 2 * N_SITES, 2):
        hamiltonian.append(
            Op(
                "- + - +",
                [orbital, orbital, orbital + 1, orbital + 1],
                factor=ONSITE_REPULSION,
                qn=[up["-"], up["+"], down["-"], down["+"]],
            )
        )
    basis = [
        BasisHalfSpin(orbital, sigmaqn=[[0, 0], [1, 0] if orbital % 2 == 0 else [0, 1]])
        for orbital in range(2 * N_SITES)
    ]
    model = record_model(hamiltonian=hamiltonian, basis=basis)
    initial_state = record_random_state(
        quantum_number=N_ELECTRONS, bond_dimension=MAX_BOND_DIMENSION, percent=1.0
    )

    # Reno's native sweep rows are [bond dimension, sector exploration fraction].
    optimizer = OptimizeConfig(
        procedure=[
            [MAX_BOND_DIMENSION // 4 if SMOKE_TEST else MAX_BOND_DIMENSION, 0.4],
            [MAX_BOND_DIMENSION // 2 if SMOKE_TEST else MAX_BOND_DIMENSION, 0.2],
            [3 * MAX_BOND_DIMENSION // 4 if SMOKE_TEST else MAX_BOND_DIMENSION, 0.1],
            *[[MAX_BOND_DIMENSION, 0.0] for _ in range(4)],
        ]
    )
    optimizer.method = "2site"
    dmrg = record_dmrg(optimize_config=optimizer, copy_initial_state=True)
    refinement = record_imaginary_time(
        evolve_config=EvolveConfig(
            EvolveMethod.tdvp_ps,
            adaptive=True,
            guess_dt=-0.001j,
            adaptive_rtol=5e-4,
            ivp_solver="RK45",
        ),
        dt=-0.5j,
        max_steps=MAX_IMAGINARY_STEPS,
        energy_change_atol=ENERGY_CHANGE_ATOL,
        normalize=True,
    )
    observations = record_observations(
        operators={
            "n_up_0": Op("- +", 0, qn=[up["-"], up["+"]]),
            "n_down_0": Op("- +", 1, qn=[down["-"], down["+"]]),
        }
    )
    # Refinement receives the DMRG result; the final native MPS is returned at runtime.
    script = assemble_script(model, initial_state, dmrg, refinement, observations)
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).with_name("generated_scripts")
    )
    return write_dry_run(script, output_dir)


if __name__ == "__main__":
    print(main())
