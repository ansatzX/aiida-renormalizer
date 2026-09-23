#!/usr/bin/env python
"""Finite-temperature CD spin-boson dynamics through explicit thermofield doubling."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from aiida import load_profile
from renormalizer.model import Op
from renormalizer.model.basis import BasisHalfSpin, BasisSHO
from renormalizer.tn import BasisTree, TreeNodeBasis
from renormalizer.utils import CompressCriteria, EvolveConfig, EvolveMethod

from aiida_renormalizer.cases.ttn_sbm_ft import (
    assemble_script,
    cd_renormalization_factor,
    discretize_cole_davidson_spectrum,
    record_evolution,
    record_model,
    write_dry_run,
)
from aiida_renormalizer.utils import run_process

SMOKE_TEST = os.getenv("AIIDA_RENO_TTN_SBM_FT_SMOKE", "0") == "1"
OUTPUT_DIR = Path(__file__).with_name("generated_scripts")
CD_AMPLITUDE = 1.0
BATH_FREQUENCY_SCALE = 1.0
CD_SHAPE_EXPONENT = 0.25  # Spectral shape, not inverse temperature.
THERMAL_ENERGY = 2.0  # k_B T in the same energy convention as the Hamiltonian.
N_BATH_MODES = 2 if SMOKE_TEST else 1000  # Before p/q doubling.
DISCRETIZATION_CUTOFF = 30.0
SPIN_BIAS = 0.0
BARE_TUNNELING = 1.0  # Coefficient of sigma_x, with no factor 1/2.
MAX_BOND_DIMENSION = 4 if SMOKE_TEST else 20
TIME_STEP = 0.1
EVOLUTION_STEPS = 1 if SMOKE_TEST else 400


def main(output_dir: str | Path | None = None) -> Path:
    destination = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"Choose a fresh output directory: {destination}")
    if not np.isfinite(THERMAL_ENERGY) or THERMAL_ENERGY <= 0:
        raise ValueError("THERMAL_ENERGY must be finite and positive")
    load_profile()
    bath, _ = run_process(
        discretize_cole_davidson_spectrum,
        ita=CD_AMPLITUDE,
        omega_c=BATH_FREQUENCY_SCALE,
        beta=CD_SHAPE_EXPONENT,
        upper_limit=DISCRETIZATION_CUTOFF,
        n_modes=N_BATH_MODES,
        method="Wang1",
    )
    frequencies = bath.get_array("omega_k")
    coupling_squared = bath.get_array("c_j2")
    cutoff = float(frequencies[-1])
    renormalization, _ = run_process(
        cd_renormalization_factor,
        bath=bath,
        lower_cutoff=cutoff,
        upper_cutoff=1000 * cutoff,
    )
    effective_tunneling = BARE_TUNNELING * renormalization.value

    # Thermal preparation is a Bogoliubov transformation of H, not imaginary-time evolution.
    # The p oscillator has positive energy, q has negative auxiliary energy.
    theta = np.arctanh(np.exp(-frequencies / (2 * THERMAL_ENERGY)))
    hamiltonian = [
        Op("sigma_z", "spin", factor=SPIN_BIAS, qn=0),
        Op("sigma_x", "spin", factor=effective_tunneling, qn=0),
    ]
    spin_basis = BasisHalfSpin("spin", sigmaqn=[0, 0])
    bath_basis, contract_labels = [], []
    initial_state = {"spin": 0}  # +z spin; every doubled oscillator starts in its vacuum.
    for index, (omega, squared_coupling, angle) in enumerate(
        zip(frequencies, coupling_squared, theta)
    ):
        omega, squared_coupling = float(omega), float(squared_coupling)
        # Retain the upstream empirical cutoff: cap at 512, round, then double.
        basis_size = 2 * int(
            round(min(max(16 * squared_coupling / omega**3 * np.cosh(angle) ** 2, 4), 512))
        )
        for branch, energy_sign, weight in (("p", 1, np.cosh(angle)), ("q", -1, np.sinh(angle))):
            dof = f"v_{index}_{branch}"
            bath_basis.append(BasisSHO(dof, omega=omega, nbas=basis_size))
            contract_labels.append(basis_size > MAX_BOND_DIMENSION)
            initial_state[dof] = 0
            hamiltonian.extend(
                [
                    Op("p^2", dof, factor=0.5 * energy_sign, qn=0),
                    Op("x^2", dof, factor=0.5 * energy_sign * omega**2, qn=0),
                    Op(
                        "sigma_z x",
                        ["spin", dof],
                        factor=float(weight * squared_coupling**0.5),
                        qn=[0, 0],
                    ),
                ]
            )
    root = BasisTree.binary_mctdh(
        bath_basis,
        contract_primitive=True,
        contract_label=contract_labels,
        dummy_label="n",
    ).root
    root.add_child(TreeNodeBasis([spin_basis]))
    model = record_model(
        hamiltonian=hamiltonian,
        basis_tree=BasisTree(root),
        initial_state=initial_state,
        compression_criteria=CompressCriteria.fixed,
        max_bond_dimension=MAX_BOND_DIMENSION,
        expand_bonds=True,
        expansion_coefficient=1e-10,
    )
    evolution = record_evolution(
        dt=TIME_STEP,
        nsteps=EVOLUTION_STEPS,
        evolve_config=EvolveConfig(EvolveMethod.tdvp_ps),
        observations={
            "sigma_z": Op("sigma_z", "spin", qn=0),
            "sigma_x": Op("sigma_x", "spin", qn=0),
        },
        observe_initial=False,
        observe_every=1,
    )
    script, _ = run_process(
        assemble_script,
        environment=bath,
        renormalization=renormalization,
        model_section=model,
        calculation_section=evolution,
        analysis_section={},
    )
    return write_dry_run(script, destination)


if __name__ == "__main__":
    print(main())
