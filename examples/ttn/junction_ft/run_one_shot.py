#!/usr/bin/env python
"""Finite-temperature junction with explicit reservoirs, phonons, and TTN state."""

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

from aiida_renormalizer.cases.ttn_junction_ft import (
    assemble_script,
    cd_renormalization_factor,
    discretize_cole_davidson_spectrum,
    record_analysis,
    record_evolution,
    record_model,
    write_dry_run,
)
from aiida_renormalizer.utils import run_process

SMOKE_TEST = os.getenv("AIIDA_RENO_TTN_JUNCTION_FT_SMOKE", "0") == "1"
OUTPUT_DIR = Path(__file__).with_name("generated_scripts")
N_PHONON_MODES = 2 if SMOKE_TEST else 1000
N_ELECTRON_MODES_PER_SIDE = 2 if SMOKE_TEST else 320
PHONON_FREQUENCY_CM = 500.0
PHONON_AMPLITUDE_CM = 2000.0  # Upstream CD prefactor is half this converted energy.
CD_SHAPE_EXPONENT = 0.5  # Not inverse temperature.
PHONON_CUTOFF_EV = 10.0
ELECTRON_BAND_PARAMETER_EV = 1.0  # The bare grid spans (-2*band, 2*band).
HYBRIDIZATION_EV = 0.2
BIAS_EV = 0.1
TEMPERATURE_K = 100.0
INITIAL_OCCUPIED = True
MAX_BOND_DIMENSION = 4 if SMOKE_TEST else 32
TIME_STEP_FS = 0.1 if SMOKE_TEST else 0.5
OBSERVATION_POINTS = 2 if SMOKE_TEST else 200  # Includes t=0; evolve one fewer step.
AU_TO_MICROAMP = 6.623618237510e3


def analyze_transport(rows, *, au_to_microamp, fs_to_au):
    """Keep the upstream convention: take real(i <generator>) and (I_R-I_L)/2."""

    def value(item):
        return complex(item["real"], item["imag"]) if isinstance(item, dict) else complex(item)

    result = []
    for row in rows:
        left = (1j * value(row["i_l_generator"])).real
        right = (1j * value(row["i_r_generator"])).real
        current = (right - left) / 2
        result.append(
            {
                "point": row["step"],
                "time_au": row["time"],
                "time_fs": row["time"] / fs_to_au,
                "i_l_au": left,
                "i_r_au": right,
                "n_s": value(row["n_s"]).real,
                "current_au": current,
                "current_microamp": current * au_to_microamp,
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

    if not np.isfinite(TEMPERATURE_K) or TEMPERATURE_K <= 0:
        raise ValueError("TEMPERATURE_K must be finite and positive")
    inverse_temperature = Quantity(TEMPERATURE_K, "K").to_beta()
    # Thermofield preparation transforms H; p/q fermions both start empty, not Fermi-filled.
    ordered_dofs = []
    for dof, shifted_energy, _ in modes:
        if shifted_energy > 0 and "s" not in ordered_dofs:
            ordered_dofs.append("s")
        ordered_dofs.extend([f"{dof}_p", f"{dof}_q"])
    if "s" not in ordered_dofs:
        raise ValueError("no positive shifted-energy partition")
    impurity_index = ordered_dofs.index("s")
    electron_basis = [BasisHalfSpin(dof, sigmaqn=[0, 0]) for dof in ordered_dofs]
    initial_state = {dof: 1 for dof in ordered_dofs}  # Basis index 1 is fermion vacuum.
    initial_state["s"] = 0 if INITIAL_OCCUPIED else 1
    hamiltonian, current_left, current_right = [], [], []
    for mode, shifted_energy, chemical_potential in modes:
        onsite_energy = shifted_energy + chemical_potential
        hopping_squared = (
            hybridization**2
            / band**2
            * np.sqrt(4 * band**2 - onsite_energy**2)
            / (2 * np.pi * density_of_states)
        )
        hopping = float(np.sqrt(hopping_squared))
        theta_e = float(np.arctan(np.exp(-inverse_temperature * shifted_energy / 2)))
        for branch, sign, weight in (("p", 1, np.cos(theta_e)), ("q", -1, np.sin(theta_e))):
            dof = f"{mode}_{branch}"
            hamiltonian.append(Op("+ -", dof, factor=sign * onsite_energy, qn=[0, 0]))
            index = ordered_dofs.index(dof)
            z_dofs = ordered_dofs[min(index, impurity_index) + 1 : max(index, impurity_index)]
            operator_dofs = [dof, *z_dofs, "s"]
            qn = [0] * len(operator_dofs)
            # q hopping is a pair creation/annihilation term in the doubled representation.
            first, second = ("+", "-") if branch == "p" else ("-", "+")
            forward = Op(
                first + " " + "Z " * len(z_dofs) + "-",
                operator_dofs,
                factor=float(hopping * weight),
                qn=qn,
            )
            reverse = Op(
                second + " " + "Z " * len(z_dofs) + "+",
                operator_dofs,
                factor=float(hopping * weight),
                qn=qn,
            )
            hamiltonian.extend([forward, reverse])
            current = current_left if mode.startswith("L") else current_right
            current.extend(reverse - forward)

    if INITIAL_OCCUPIED:
        hamiltonian.append(
            Op("+ -", "s", factor=float(-4 * (coupling_squared / frequencies**2).sum()), qn=[0, 0])
        )
    theta_b = np.arctanh(np.exp(-frequencies * inverse_temperature / 2))
    phonon_basis, contract_labels = [], []
    for index, (omega, squared_coupling, angle) in enumerate(
        zip(frequencies, coupling_squared, theta_b)
    ):
        omega, squared_coupling = float(omega), float(squared_coupling)
        basis_size = int(
            round(min(max(16 * squared_coupling / omega**3 * np.cosh(angle) ** 2, 4), 512))
        )
        impurity_number = Op("+ -", "s", qn=[0, 0])
        transformed_number = (
            impurity_number - Op.identity("s") if INITIAL_OCCUPIED else impurity_number
        )
        for branch, sign, weight in (("p", 1, np.cosh(angle)), ("q", -1, np.sinh(angle))):
            dof = f"v_{index}_{branch}"
            phonon_basis.append(BasisSHO(dof, omega=omega, nbas=basis_size))
            contract_labels.append(basis_size > MAX_BOND_DIMENSION)
            initial_state[dof] = 0  # Boson thermofield vacuum; temperature is encoded in H.
            hamiltonian.extend(
                [
                    Op("p^2", dof, factor=0.5 * sign, qn=0),
                    Op("x^2", dof, factor=0.5 * sign * omega**2, qn=0),
                ]
            )
            product = transformed_number * Op(
                "x", dof, factor=float(2 * squared_coupling**0.5 * weight), qn=0
            )
            hamiltonian.extend([product] if isinstance(product, Op) else product)
    phonon_tree = BasisTree.binary_mctdh(
        phonon_basis,
        contract_primitive=True,
        contract_label=contract_labels,
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
        "i_l_generator": OpSum(current_left),
        "i_r_generator": OpSum(current_right),
        "n_s": Op("+ -", "s", qn=[0, 0]),
    }
    evolution = record_evolution(
        dt=TIME_STEP_FS * constant.fs2au,
        nsteps=OBSERVATION_POINTS - 1,
        evolve_config=EvolveConfig(EvolveMethod.tdvp_ps),
        observations=observations,
        observe_initial=True,
        observe_every=1,
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
