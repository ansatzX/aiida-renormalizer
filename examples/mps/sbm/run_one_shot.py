#!/usr/bin/env python
"""Ohmic spin-boson MPS dynamics: explicit bath, Hartree state and compression.

All frequencies, energies, displacements and times use Reno atomic units.
The launcher records and writes an executable calculation, without evolving it.
"""

import os
from pathlib import Path

from aiida import load_profile
from renormalizer.model import Op, Phonon
from renormalizer.model.basis import BasisHalfSpin, BasisSHO
from renormalizer.utils import (
    CompressConfig,
    CompressCriteria,
    EvolveConfig,
    EvolveMethod,
    Quantity,
)

from aiida_renormalizer.cases.mps_sbm import (
    assemble_script,
    record_evolution,
    record_model,
    record_observations,
    record_product_state,
    write_dry_run,
)
from aiida_renormalizer.cases.mps_sbm.bath import discretize_ohmic_spectrum
from aiida_renormalizer.utils import run_process

# Preserve the existing small-run environment switch; ordinary execution only generates code.
SMOKE_TEST = os.getenv("AIIDA_RENO_MPS_SBM_SMOKE", "0").strip().lower() not in {
    "0",
    "false",
    "no",
    "off",
}

ALPHA = 0.05  # Dimensionless Ohmic spectral-density strength.
DELTA = 1.0  # Bare coefficient of sigma_x.
CUTOFF_FREQUENCY = 20.0
N_BATH_MODES = 4 if SMOKE_TEST else 300
RENORMALIZATION_P = 1.0  # Self-consistent bath cutoff = p * renormalized delta.
MAX_PHONON_LEVELS = 128  # Cap on Reno's displaced-vacuum basis-size estimate.
MAX_BOND_DIMENSION = 32
TIME_STEP = 0.1
TOTAL_TIME = 20.0
# Retain Reno's original inclusive-endpoint time-grid convention.
N_STEPS = 1 if SMOKE_TEST else int(abs(TOTAL_TIME) // abs(TIME_STEP)) + 1
SPIN_DOF = "spin"


def main(output_dir=None):
    load_profile()
    bath, _ = run_process(
        discretize_ohmic_spectrum,
        alpha=ALPHA,
        cutoff_frequency=CUTOFF_FREQUENCY,
        bare_delta=DELTA,
        renormalization_p=RENORMALIZATION_P,
        n_modes=N_BATH_MODES,
        sort_modes=True,
    )
    # Apply the bath correction visibly; the bath library never changes H itself.
    delta_eff = DELTA * bath.base.attributes.get("renormalization_factor")
    hamiltonian = [Op("sigma_z", SPIN_DOF, factor=0.0), Op("sigma_x", SPIN_DOF, factor=delta_eff)]
    basis = [BasisHalfSpin(SPIN_DOF, sigmaqn=[0, 0])]
    for mode, (omega, displacement) in enumerate(
        zip(bath.get_array("frequencies"), bath.get_array("displacements"))
    ):
        # Native zero-temperature displaced-vacuum estimate controls SHO truncation.
        phonon = Phonon.simplest_phonon(
            Quantity(omega),
            Quantity(displacement),
            temperature=Quantity(0),
            max_pdim=MAX_PHONON_LEVELS,
        )
        basis.append(
            BasisSHO(
                mode, omega=omega, nbas=phonon.n_phys_dim, x0=0.0, dvr=False, general_xp_power=False
            )
        )
        hamiltonian.extend(
            [
                Op("p^2", mode, factor=0.5),
                Op("x^2", mode, factor=0.5 * omega**2),
                Op("sigma_z x", [SPIN_DOF, mode], factor=-(omega**2) * displacement),
            ]
        )
    model = record_model(hamiltonian=hamiltonian, basis=basis)
    # Spin up (sigma_z=+1) and each oscillator vacuum, independently editable.
    local_states = {SPIN_DOF: 0, **{mode: 0 for mode in range(N_BATH_MODES)}}
    initial_state = record_product_state(local_states=local_states)
    observations = record_observations(
        operators={"sigma_x": Op("sigma_x", SPIN_DOF), "sigma_z": Op("sigma_z", SPIN_DOF)},
        one_site_rdms=[0],
        bond_entropy=True,
    )
    compression = CompressConfig(
        criteria=CompressCriteria.threshold,
        threshold=1e-4,
        max_bonddim=MAX_BOND_DIMENSION,
        vmethod="2site",
        vrtol=1e-5,
        vguess_m=(5, 5),
        vprocedure=[[MAX_BOND_DIMENSION, fraction] for fraction in [0.5, 0.3, 0.1] + [0.0] * 10],
    )
    evolution = EvolveConfig(
        EvolveMethod.prop_and_compress,
        adaptive=True,
        guess_dt=0.1,
        adaptive_rtol=5e-4,
        taylor_order=5,
        rk_solver="C_RK4",
    )
    dynamics = record_evolution(
        compress_config=compression,
        evolve_config=evolution,
        dt=TIME_STEP,
        nsteps=N_STEPS,
        normalize=True,
        # Hamiltonian-guided expansion is needed when selecting a TDVP method.
        expand_bonds=evolution.is_tdvp,
        expansion_coefficient=1e-16,
        expansion_include_ex=False,
        observe_initial=True,
        observe_every=1,
    )
    script = assemble_script(model, initial_state, observations, dynamics)
    output_dir = (
        Path(output_dir)
        if output_dir is not None
        else Path(__file__).with_name("generated_scripts")
    )
    return write_dry_run(script, output_dir)


if __name__ == "__main__":
    print(main())
