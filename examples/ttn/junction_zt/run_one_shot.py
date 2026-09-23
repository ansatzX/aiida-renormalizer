#!/usr/bin/env python
"""Zero-temperature junction: explicit reservoirs, Jordan-Wigner strings, phonons, and TTN state."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
from aiida import load_profile
from renormalizer import Quantity
from renormalizer.model import Op
from renormalizer.model.basis import BasisDummy, BasisHalfSpin, BasisSHO
from renormalizer.model.op import OpSum
from renormalizer.tn import BasisTree, TreeNodeBasis
from renormalizer.utils import CompressCriteria, EvolveConfig, EvolveMethod, constant

from aiida_renormalizer.cases.ttn_junction_zt import (
    assemble_script,
    cd_renormalization_factor,
    discretize_cole_davidson_spectrum,
    record_analysis,
    record_evolution,
    record_model,
    write_dry_run,
)
from aiida_renormalizer.utils import run_process

SMOKE_TEST = os.getenv("AIIDA_RENO_TTN_JUNCTION_ZT_SMOKE", "0") == "1"
OUTPUT_DIR = Path(__file__).with_name("generated_scripts")
N_PHONON_MODES = 2 if SMOKE_TEST else 500
N_ELECTRON_MODES_PER_SIDE = 2 if SMOKE_TEST else 160
PHONON_FREQUENCY_CM = 500.0
PHONON_AMPLITUDE_CM = 2000.0  # Upstream CD prefactor is half this converted energy.
CD_SHAPE_EXPONENT = 0.5  # Not inverse temperature.
PHONON_CUTOFF_EV = 5.0
ELECTRON_BAND_PARAMETER_EV = 1.0  # The bare grid spans (-2*band, 2*band).
HYBRIDIZATION_EV = 0.2
BIAS_EV = 0.1

INITIAL_OCCUPIED = True
MAX_BOND_DIMENSION = 4 if SMOKE_TEST else 32
TIME_STEP_FS = 0.1 if SMOKE_TEST else 0.5
OBSERVATION_POINTS = 2 if SMOKE_TEST else 100  # Includes t=0; evolve one fewer step.
AU_TO_MICROAMP = 6.623618237510e3


def analyze_transport(rows, *, au_to_microamp, fs_to_au):
    """The real primitive current generator represents I = i <generator>."""

    def value(item):
        return complex(item["real"], item["imag"]) if isinstance(item, dict) else complex(item)

    def encoded(item):
        return {"real": item.real, "imag": item.imag}

    result = []
    for row in rows:
        left = 1j * value(row["I_L_generator"])
        right = 1j * value(row["I_R_generator"])
        current = (right - left) / 2
        result.append(
            {
                "observation_point": row["step"],
                "evolution_step": row["step"],
                "time_au": row["time"],
                "time_fs": row["time"] / fs_to_au,
                "bond_dims": row["bond_dims"],
                "I_L": encoded(left),
                "I_R": encoded(right),
                "N_L": encoded(value(row["N_L"])),
                "N_R": encoded(value(row["N_R"])),
                "N_S": encoded(value(row["N_S"])),
                "current": encoded(current),
                "I_L_microamp": encoded(left * au_to_microamp),
                "I_R_microamp": encoded(right * au_to_microamp),
                "current_microamp": encoded(current * au_to_microamp),
            }
        )
    return result


def main(output_dir: str | Path | None = None) -> Path:
    destination = Path(output_dir) if output_dir is not None else OUTPUT_DIR
    if destination.exists():
        raise FileExistsError(f"Choose a fresh output directory: {destination}")
    if N_ELECTRON_MODES_PER_SIDE < 2 or type(INITIAL_OCCUPIED) is not bool:
        raise ValueError(
            "at least two electronic modes per reservoir and an occupation bool are required"
        )
    load_profile()
    bath, _ = run_process(
        discretize_cole_davidson_spectrum,
        ita=Quantity(PHONON_AMPLITUDE_CM, "cm-1").as_au() / 2,
        omega_c=Quantity(PHONON_FREQUENCY_CM, "cm-1").as_au(),
        beta=CD_SHAPE_EXPONENT,
        upper_limit=Quantity(PHONON_CUTOFF_EV, "eV").as_au(),
        n_modes=N_PHONON_MODES,
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
    # The script applies the CD tail correction to electronic bandwidth, hopping and bias.
    band = Quantity(ELECTRON_BAND_PARAMETER_EV, "eV").as_au() * renormalization.value
    hybridization = Quantity(HYBRIDIZATION_EV, "eV").as_au() * renormalization.value
    mu_left = Quantity(BIAS_EV * renormalization.value / 2, "eV").as_au()
    mu_right = -mu_left
    energies = (
        np.arange(1, N_ELECTRON_MODES_PER_SIDE + 1) / (N_ELECTRON_MODES_PER_SIDE + 1) * 4 * band
        - 2 * band
    )
    density_of_states = 1 / float(energies[1] - energies[0])
    modes = [(f"L{i}", float(energy - mu_left), mu_left) for i, energy in enumerate(energies)]
    modes += [(f"R{i}", float(energy - mu_right), mu_right) for i, energy in enumerate(energies)]
    modes.sort(key=lambda item: item[1])

    # Negative/positive shifted-energy partitions, independently of physical L/R reservoirs.
    ordered_dofs = []
    for dof, shifted_energy, _ in modes:
        if shifted_energy > 0 and "s" not in ordered_dofs:
            ordered_dofs.append("s")
        ordered_dofs.append(dof)
    if "s" not in ordered_dofs:
        raise ValueError("no positive shifted-energy partition")
    impurity_index = ordered_dofs.index("s")
    electron_basis = [BasisHalfSpin(dof, sigmaqn=[0, 0]) for dof in ordered_dofs]
    # index 0 = occupied; index 1 = empty. QN blocks are zero; JW strings supply fermion signs.
    initial_state = {dof: int(index > impurity_index) for index, dof in enumerate(ordered_dofs)}
    initial_state["s"] = 0 if INITIAL_OCCUPIED else 1
    hamiltonian, current_left, current_right = [], [], []
    for dof, shifted_energy, chemical_potential in modes:
        onsite_energy = shifted_energy + chemical_potential
        hamiltonian.append(Op("+ -", dof, factor=onsite_energy, qn=[0, 0]))
        hopping_squared = (
            hybridization**2
            / band**2
            * np.sqrt(4 * band**2 - onsite_energy**2)
            / (2 * np.pi * density_of_states)
        )
        hopping = float(np.sqrt(hopping_squared))
        index = ordered_dofs.index(dof)
        z_dofs = ordered_dofs[min(index, impurity_index) + 1 : max(index, impurity_index)]
        operator_dofs = [dof, *z_dofs, "s"]
        qn = [0] * len(operator_dofs)
        forward = Op("+ " + "Z " * len(z_dofs) + "-", operator_dofs, factor=hopping, qn=qn)
        reverse = Op("- " + "Z " * len(z_dofs) + "+", operator_dofs, factor=hopping, qn=qn)
        hamiltonian.extend([forward, reverse])
        current = current_left if dof.startswith("L") else current_right
        current.extend(reverse - forward)  # The missing factor i is applied by analyze_transport.

    if INITIAL_OCCUPIED:
        hamiltonian.append(
            Op("+ -", "s", factor=float(-4 * (coupling_squared / frequencies**2).sum()), qn=[0, 0])
        )
    phonon_basis = []
    for index, (omega, squared_coupling) in enumerate(zip(frequencies, coupling_squared)):
        omega, squared_coupling = float(omega), float(squared_coupling)
        dof = f"v_{index}"
        basis_size = int(round(max(16 * squared_coupling / omega**3, 4)))
        phonon_basis.append(BasisSHO(dof, omega=omega, nbas=basis_size))
        initial_state[dof] = 0
        hamiltonian.extend(
            [Op("p^2", dof, factor=0.5, qn=0), Op("x^2", dof, factor=0.5 * omega**2, qn=0)]
        )
        impurity_number = Op("+ -", "s", qn=[0, 0])
        transformed_number = (
            impurity_number - Op.identity("s") if INITIAL_OCCUPIED else impurity_number
        )
        product = transformed_number * Op("x", dof, factor=2 * squared_coupling**0.5, qn=0)
        hamiltonian.extend([product] if isinstance(product, Op) else product)
    # The occupied-state energy shift and n_s-1 coupling above explicitly choose the polaron frame.
    phonon_tree = BasisTree.binary_mctdh(
        phonon_basis,
        contract_primitive=False,
        contract_label=None,
        dummy_label="phonon-dummy",
    )

    # Electronic subtrees partition shifted energies, not the physical L/R leads.
    negative_tree = BasisTree.binary_mctdh(
        electron_basis[:impurity_index],
        contract_primitive=False,
        contract_label=None,
        dummy_label="EL-dummy",
    )
    positive_tree = BasisTree.binary_mctdh(
        electron_basis[impurity_index + 1 :],
        contract_primitive=False,
        contract_label=None,
        dummy_label="ER-dummy",
    )
    electrodes = TreeNodeBasis([BasisDummy("dummy")])
    electrodes.add_child([negative_tree.root, positive_tree.root])
    root = TreeNodeBasis([electron_basis[impurity_index]])
    root.add_child([electrodes, phonon_tree.root])
    model = record_model(
        hamiltonian=hamiltonian,
        basis_tree=BasisTree(root),
        initial_state=initial_state,
        compression_criteria=CompressCriteria.fixed,
        max_bond_dimension=MAX_BOND_DIMENSION,
        expand_bonds=True,
        expansion_coefficient=1e-10,
    )
    observations = {
        "I_L_generator": OpSum(current_left),
        "I_R_generator": OpSum(current_right),
        "N_L": OpSum([Op("+ -", f"L{i}", qn=[0, 0]) for i in range(N_ELECTRON_MODES_PER_SIDE)]),
        "N_R": OpSum([Op("+ -", f"R{i}", qn=[0, 0]) for i in range(N_ELECTRON_MODES_PER_SIDE)]),
        "N_S": Op("+ -", "s", qn=[0, 0]),
    }
    evolution = record_evolution(
        dt=TIME_STEP_FS * constant.fs2au,
        nsteps=OBSERVATION_POINTS - 1,
        evolve_config=EvolveConfig(EvolveMethod.tdvp_ps),
        observations=observations,
        observe_initial=True,
        observe_every=1,
        record_bond_dimensions=True,
    )
    analysis = record_analysis(
        analyze_transport, au_to_microamp=AU_TO_MICROAMP, fs_to_au=float(constant.fs2au)
    )
    script, _ = run_process(
        assemble_script,
        environment=bath,
        renormalization=renormalization,
        model_section=model,
        calculation_section=evolution,
        analysis_section=analysis,
    )
    return write_dry_run(script, destination)


if __name__ == "__main__":
    print(main())
