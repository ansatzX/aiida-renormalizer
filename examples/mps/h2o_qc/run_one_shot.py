#!/usr/bin/env python
"""Water/STO-3G: record an explicit spin-orbital Hamiltonian and generate MPS DMRG.

All energies are in hartree. The FCIDUMP contains 10 electrons in 7 spatial
orbitals. Reno expands its integrals into 14 interleaved alpha/beta spin orbitals.
Generating this file does not run DMRG; execute the emitted Python program to solve it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from aiida import load_profile, orm
from renormalizer.model import Op, h_qc
from renormalizer.model.basis import BasisHalfSpin
from renormalizer.utils import OptimizeConfig

from aiida_renormalizer.cases.mps_h2o_qc import (
    assemble_script,
    prepare_fcidump,
    record_mps_model,
    record_optimization,
    record_random_state,
    write_dry_run,
)
from aiida_renormalizer.utils import run_process

SMOKE_TEST = os.getenv("AIIDA_RENO_MPS_H2O_QC_SMOKE", "0") == "1"
OUTPUT_DIR = Path(__file__).with_name("generated_scripts")  # Choose a fresh directory.
FCIDUMP_FILE = Path(__file__).with_name("h2o_fcidump.txt")
N_SPATIAL_ORBITALS = 7  # Each spatial orbital supplies one alpha and one beta orbital.
ELECTRON_NUMBERS = [5, 5]  # Conserved [N_alpha, N_beta], not total occupation per site.
MAX_BOND_DIMENSION = 10 if SMOKE_TEST else 50

# Native DMRG sweep schedule: [bond dimension, fraction of extra symmetry-sector states].
OPTIMIZE_CONFIG = OptimizeConfig(procedure=(
    [[MAX_BOND_DIMENSION, 0.2], [MAX_BOND_DIMENSION, 0.0]] if SMOKE_TEST else
    [[MAX_BOND_DIMENSION, mixing] for mixing in (0.4, 0.2, 0.1, 0.0, 0.0, 0.0, 0.0)]
))
OPTIMIZE_CONFIG.method = "2site"
OPTIMIZE_CONFIG.algo = "davidson"
OPTIMIZE_CONFIG.e_rtol = 1e-6  # Native sweep-energy stopping tolerance.
OPTIMIZE_CONFIG.e_atol = 1e-8  # Absolute energy tolerance in hartree.
REFERENCE_TOTAL_ENERGY = -75.008697516450  # STO-3G full-configuration-interaction reference.


def main(output_dir: str | Path | None = None) -> Path:
    """Record inputs and write a runnable, self-contained Reno program plus manifest."""
    destination = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"Choose a fresh output directory: {destination}")
    load_profile()

    # 1. Record the actual FCIDUMP, then use Reno's antisymmetrized spin-orbital integrals.
    integrals, _ = run_process(
        prepare_fcidump,
        fcidump=orm.SinglefileData(file=str(FCIDUMP_FILE)),
        spatial_norbs=N_SPATIAL_ORBITALS,
    )
    h1e = integrals.get_array("h1e")
    h2e = integrals.get_array("h2e")
    nuclear_repulsion = integrals.base.attributes.get("nuclear_repulsion")
    n_spin_orbitals = len(h1e)

    # 2. Site order alpha_0, beta_0, alpha_1, beta_1, ...; local states empty/occupied.
    basis = []
    for orbital in range(n_spin_orbitals):
        occupied_qn = [1, 0] if orbital % 2 == 0 else [0, 1]
        basis.append(BasisHalfSpin(orbital, sigmaqn=[[0, 0], occupied_qn]))

    # Jordan-Wigner: a_j = Z_0...Z_(j-1) sigma_+[j]; creation uses sigma_-.
    annihilate, create = h_qc.generate_ladder_operator(n_spin_orbitals)
    hamiltonian = []
    for p, q in np.argwhere(h1e != 0):
        term = h_qc.simplify_op(
            Op.product([create[p], annihilate[q]]), n_spin_orbitals, conserve_qn=True,
        )
        hamiltonian.append(term * float(h1e[p, q]))
    # Reno's h2e already includes antisymmetry factors: no additional 1/2 or 1/4 here.
    for p, q, r, s in np.argwhere(h2e != 0):
        term = h_qc.simplify_op(
            Op.product([create[p], create[q], annihilate[r], annihilate[s]]),
            n_spin_orbitals, conserve_qn=True,
        )
        hamiltonian.append(term * float(h2e[p, q, r, s]))
    model_section = record_mps_model(basis=basis, hamiltonian=hamiltonian, integrals=integrals)

    # 3. MPS uses Model -> Mpo and a number-conserving random Mps, not a TTN basis tree.
    state_section = record_random_state(
        quantum_number=ELECTRON_NUMBERS,
        bond_dimension=MAX_BOND_DIMENSION,
        sector_mixing=1.0,  # Reno Mps.random(percent=...); uses Reno's initialized RNG.
    )
    optimization_section = record_optimization(
        optimize_config=OPTIMIZE_CONFIG,
        preserve_initial_state=True,  # Native optimize_mps mutates its input; pass a copy.
        nuclear_repulsion=nuclear_repulsion,  # Add only when reporting total energy.
        reference_energy=REFERENCE_TOTAL_ENERGY,
        check_reference=not SMOKE_TEST,  # Two short sweeps are a runtime check, not convergence.
        reference_rtol=1e-5,
        reference_atol=1e-8,
    )

    # 4. The runtime returns its optimized MPS for further work in the same Python process.
    script, _ = run_process(
        assemble_script, model_section=model_section, state_section=state_section,
        optimization_section=optimization_section,
    )
    return write_dry_run(script, destination)


if __name__ == "__main__":
    print(main())
