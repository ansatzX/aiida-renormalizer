#!/usr/bin/env python
"""Zero-temperature spin-boson dynamics with a Cole-Davidson bath and a TTN.

H = epsilon*sigma_z + delta_eff*sigma_x
    + sum_j [p_j**2/2 + omega_j**2*x_j**2/2 + c_j*sigma_z*x_j].

Use hbar = 1 and unit oscillator masses. Energies and angular frequencies share
one chosen unit; times use its inverse. These plain numbers imply no eV/ps conversion.
This file records the choices and generates a standalone Reno program; executing
the generated program performs the tensor-network evolution.
"""

from __future__ import annotations

import os
from pathlib import Path

from aiida import load_profile
from renormalizer.model import Op
from renormalizer.model.basis import BasisHalfSpin, BasisSHO
from renormalizer.utils import EvolveConfig, EvolveMethod

from aiida_renormalizer.cases.ttn_sbm_zt import (
    assemble_script,
    cd_renormalization_factor,
    discretize_cole_davidson_spectrum,
    record_binary_tree_model,
    record_evolution,
    write_dry_run,
)
from aiida_renormalizer.utils import run_process

SMOKE_TEST = os.getenv("AIIDA_RENO_SBM_ZT_SMOKE", "0") == "1"  # A small runtime check.
OUTPUT_DIR = Path(__file__).with_name("generated_scripts")  # Must be a fresh directory.

# Cole-Davidson spectral density: eta * sin(beta*atan(w/wc)) / (1 + (w/wc)**2)**(beta/2).
CD_STRENGTH = 1.0  # eta (Reno calls it "ita"): spectral-density prefactor, in energy units.
OMEGA_C = 0.1  # Characteristic bath frequency; the spectral density extends above it.
CD_BETA = 0.5  # Dimensionless spectral-shape exponent, not inverse temperature.
N_BATH_MODES = 4 if SMOKE_TEST else 1000  # Number of discrete harmonic oscillators.
BATH_FREQUENCY_LIMIT = 30.0  # Upper frequency included in Wang1 discretization.

# Bare spin Hamiltonian: epsilon*sigma_z + delta*sigma_x, with no extra factor of 1/2.
EPSILON = 0.0  # Half the energy bias between the two sigma_z basis states.
DELTA = 1.0  # Bare tunneling coefficient; bath renormalization is applied below.
SPIN_DOF = "spin"  # Degree-of-freedom label shared by operators, basis, and initial state.

# Tensor-network accuracy and real-time propagation.
MAX_BOND_DIMENSION = 4 if SMOKE_TEST else 20  # Virtual TTN bond size, not local SHO basis size.
TIME_STEP = 0.2  # Evolution interval in inverse energy units (hbar = 1).
N_STEPS = 1 if SMOKE_TEST else 200  # Total duration = N_STEPS * TIME_STEP.
EVOLVE_CONFIG = EvolveConfig(
    method=EvolveMethod.tdvp_ps,  # One-site TDVP projector splitting.
)
# For tdvp_vmf, ivp_rtol/ivp_atol set integration tolerances; reg_epsilon regularizes inversion.

# Unentangled initial state; this is not the interacting spin-bath ground state.
INITIAL_SPIN_STATE = 0  # 0 = +z, 1 = -z; [2**-0.5, 2**-0.5] gives +x.
INITIAL_BATH_STATE = 0  # SHO occupation: 0 is each undisplaced oscillator's vacuum.
EXTRA_SYSTEM_TERMS = []  # Optional native Reno Op terms, added to the spin Hamiltonian.


def main(output_dir: str | Path | None = None) -> Path:
    """Record preprocessing and generate the runtime program; return its Python file path."""
    destination = Path(output_dir) if output_dir else OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"Choose a fresh output directory: {destination}")
    load_profile()

    # 1. Replace the continuous CD spectrum with mode frequencies omega_j and couplings c_j.
    bath, _ = run_process(
        discretize_cole_davidson_spectrum,
        ita=CD_STRENGTH,
        omega_c=OMEGA_C,
        beta=CD_BETA,
        upper_limit=BATH_FREQUENCY_LIMIT,
        n_modes=N_BATH_MODES,
        method="Wang1",  # Mode density proportional to J(omega)/omega.
    )
    mode_frequencies = bath.get_array("omega_k")
    mode_couplings_squared = bath.get_array("c_j2")  # c_j**2 for c_j*sigma_z*x_j.

    # 2. Apply the high-frequency tail correction to the bare tunneling coefficient.
    # These bounds match Reno's CD example; they are separate from the discretization limit.
    tail_start_frequency = float(mode_frequencies[-1])  # Highest discrete mode frequency.
    renormalization, _ = run_process(
        cd_renormalization_factor,
        bath=bath,
        lower_cutoff=tail_start_frequency,
        upper_cutoff=1000 * tail_start_frequency,
    )
    delta_eff = DELTA * renormalization.value  # Effective coefficient in the simulated Hamiltonian.

    # 3. Build the explicit Hamiltonian and local basis. All quantum-number labels are zero.
    hamiltonian = [
        Op("sigma_z", SPIN_DOF, factor=EPSILON, qn=0),
        Op("sigma_x", SPIN_DOF, factor=delta_eff, qn=0),
        *EXTRA_SYSTEM_TERMS,
    ]
    basis = [BasisHalfSpin(SPIN_DOF, sigmaqn=[0, 0])]
    bath_dofs = []  # Mode labels in the order used to construct the bath tree.
    contract_labels = []  # True: give this mode its own primitive-contraction leaf.
    initial_state = {SPIN_DOF: INITIAL_SPIN_STATE}

    for index, (omega, c_squared) in enumerate(zip(mode_frequencies, mode_couplings_squared)):
        mode_dof = f"v_{index}"
        omega = float(omega)
        c_squared = float(c_squared)
        coupling = c_squared**0.5
        # Reno's empirical truncation: keep at least four SHO states, occupations 0..size-1.
        # This is a starting basis choice; convergence still requires checking larger sizes.
        basis_size = int(round(max(16 * c_squared / max(omega, 1e-12) ** 3, 4.0)))
        basis.append(BasisSHO(mode_dof, omega=omega, nbas=basis_size))
        bath_dofs.append(mode_dof)
        contract_labels.append(basis_size > MAX_BOND_DIMENSION)
        initial_state[mode_dof] = INITIAL_BATH_STATE
        hamiltonian.extend(
            [
                Op("p^2", mode_dof, factor=0.5, qn=0),  # Kinetic energy, mass = 1.
                Op("x^2", mode_dof, factor=0.5 * omega**2, qn=0),  # Harmonic potential.
                Op("sigma_z x", [SPIN_DOF, mode_dof], factor=coupling, qn=[0, 0]),
            ]
        )

    # 4. Record a binary bath tree with the spin attached at its root, and the product state.
    # The sections below are recorded code; the live TTNS is created by the generated program.
    model_section = record_binary_tree_model(
        hamiltonian=hamiltonian,
        basis=basis,
        bath_dofs=bath_dofs,
        root_dofs=[SPIN_DOF],
        contract_primitive=True,
        contract_labels=contract_labels,
        dummy_label="n",  # Prefix for internal tree-node labels; not a physical mode.
        initial_state=initial_state,
        compression_criteria="fixed",
        max_bond_dimension=MAX_BOND_DIMENSION,
        expand_bonds=True,  # Seed virtual bonds first: one-site TDVP cannot enlarge them.
        expansion_coefficient=1e-10,  # Small Hamiltonian-guided expansion amplitude.
    )

    # 5. Record a real-time segment and its sampled observables.
    evolution_section = record_evolution(
        dt=TIME_STEP,
        nsteps=N_STEPS,
        evolve_config=EVOLVE_CONFIG,
        observations={
            "sigma_z": Op("sigma_z", SPIN_DOF, qn=0),  # Population difference.
            "sigma_x": Op("sigma_x", SPIN_DOF, qn=0),  # Twice the real spin coherence.
        },
        observe_initial=False,  # First observation follows the first evolution step.
        observe_every=1,  # Measure every local step of this segment.
        start_time=0.0,  # Time label offset; does not prepare or reset the state.
        start_step=0,  # Step label offset for continuation within one runtime process.
    )

    # 6. Assemble and write the standalone Python file plus the CalcJob execution manifest.
    script, _ = run_process(
        assemble_script,
        environment=bath,
        renormalization=renormalization,
        model_section=model_section,
        calculation_section=evolution_section,
        restart_contract={"mode": "replay-only", "checkpoint_restart": False},
    )
    return write_dry_run(script, destination)


if __name__ == "__main__":
    print(main())
