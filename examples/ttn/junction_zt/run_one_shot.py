#!/usr/bin/env python
"""TTN junction zero-temperature script-generation example."""

from __future__ import annotations

import numpy
from aiida import load_profile, orm
from renormalizer import Quantity
from renormalizer.model import Op
from renormalizer.sbm import ColeDavidsonSDF

from aiida_renormalizer.calcfunction.calcfunction_ttn_junction_zt import (
    build_bundle_manifest,
    build_topology,
    build_ttn_script,
    define_basis,
    define_hamiltonian_terms,
)
from aiida_renormalizer.data.utils import to_native
from aiida_renormalizer.example_support import materialize_python_script_bundle_preview
from aiida_renormalizer.utils import run_process
from aiida_renormalizer.workchains.bundle_runner import BundleRunnerWorkChain

load_profile()

CODE = "reno-script-clean@localhost"
WORK_DIR = "generated_scripts"
REAL_RUN = False
DEBUG_PROVENANCE = False
FAIL_FAST = True
MAX_RETRIES = 0
RESUME_FROM_STAGE = 1

# INPUT: spectral density / electronic bath setup.
N_PH_MODE = 500
N_E_MODE = 160
OMEGA_C_CM = 500.0
ITA_CM = 2000.0
BETA = 0.5
UPPER_LIMIT_EV = 5.0

# INPUT: impurity / transport setup.
INITIAL_OCCUPIED = True
M_MAX = 32

# CALC: dynamics settings.
DT_FS = 0.5
NSTEPS = 100
METHOD = "tdvp_ps"


# Workflow wiring below this line.
def main() -> None:
    input_params = {
        "n_ph_mode": N_PH_MODE,
        "n_e_mode": N_E_MODE,
        "omega_c_cm": OMEGA_C_CM,
        "ita_cm": ITA_CM,
        "beta": BETA,
    }
    model_params = {
        "upper_limit_ev": UPPER_LIMIT_EV,
        "m_max": M_MAX,
        "initial_occupied": INITIAL_OCCUPIED,
    }
    calc_params = {
        "workflow_name": "ttn_junction_zt",
        "dt_fs": DT_FS,
        "nsteps": NSTEPS,
        "method": METHOD,
    }

    omega_c = Quantity(OMEGA_C_CM, "cm-1").as_au()
    ita = Quantity(ITA_CM, "cm-1").as_au() / 2
    upper_limit = Quantity(1.0, "eV").as_au() * UPPER_LIMIT_EV

    sdf = ColeDavidsonSDF(ita, omega_c, BETA, upper_limit)
    w, c2 = sdf.Wang1(N_PH_MODE)
    c = numpy.sqrt(c2)
    reno = sdf.reno(w[-1])

    beta_e = Quantity(1.0, "eV").as_au() * reno
    alpha_e = Quantity(0.2, "eV").as_au() * reno
    bias = 0.1 * reno
    mu_l = Quantity(bias / 2, "eV").as_au()
    mu_r = Quantity(-bias / 2, "eV").as_au()

    e_k = numpy.arange(1, N_E_MODE + 1) / (N_E_MODE + 1) * 4 * beta_e - 2 * beta_e
    rho_e = 1 / (e_k[1] - e_k[0])
    e_k_l = e_k - mu_l
    e_k_r = e_k - mu_r
    mode_with_energy = [(f"L{i}", float(e)) for i, e in enumerate(e_k_l)] + [
        (f"R{i}", float(e)) for i, e in enumerate(e_k_r)
    ]
    mode_with_energy.sort(key=lambda item: item[1])

    ordered_electrode_dofs: list[str] = []
    first_positive = True
    for mode, energy in mode_with_energy:
        if energy > 0 and first_positive:
            first_positive = False
            ordered_electrode_dofs.append("s")
        ordered_electrode_dofs.append(mode)

    # Build model: full Hamiltonian term list.
    s_idx = ordered_electrode_dofs.index("s")
    hamiltonian_terms_py: list[list[object]] = []

    for mode, energy in mode_with_energy:
        mu = mu_l if mode.startswith("L") else mu_r
        hamiltonian_terms_py.append(["+ -", mode, float(energy + mu), 0])

        v2 = alpha_e**2 / beta_e**2 * numpy.sqrt(4 * beta_e**2 - (energy + mu) ** 2) / 2 / numpy.pi / rho_e
        coupling = float(numpy.sqrt(v2))

        idx = ordered_electrode_dofs.index(mode)
        if idx < s_idx:
            z_dofs = ordered_electrode_dofs[idx + 1 : s_idx]
        else:
            z_dofs = ordered_electrode_dofs[s_idx + 1 : idx]

        hamiltonian_terms_py.extend(
            [
                ["+ " + "Z " * len(z_dofs) + "-", [mode] + z_dofs + ["s"], coupling, 0],
                ["- " + "Z " * len(z_dofs) + "+", [mode] + z_dofs + ["s"], coupling, 0],
            ]
        )

    if INITIAL_OCCUPIED:
        hamiltonian_terms_py.append(["+ -", "s", float(-4 * (c**2 / w**2).sum()), [0, 0]])

    for imode, omega in enumerate(w):
        hamiltonian_terms_py.extend(
            [
                ["p^2", f"v_{imode}", 0.5, 0],
                ["x^2", f"v_{imode}", 0.5 * float(omega) ** 2, 0],
            ]
        )

    for imode, coupling in enumerate(c):
        sys_op = Op("+ -", "s", qn=[0, 0])
        if INITIAL_OCCUPIED:
            sys_op = sys_op - Op.identity("s")
        product = sys_op * Op("x", f"v_{imode}", factor=2 * float(coupling), qn=[0])
        terms = [product] if isinstance(product, Op) else list(product)
        for op in terms:
            symbol, dofs, factor, qn = op.to_tuple()
            factor_c = complex(factor)
            if abs(factor_c.imag) > 1e-12:
                raise ValueError(f"junction_zt expects real primitive terms, got {factor}")
            hamiltonian_terms_py.append([symbol, to_native(dofs), float(factor_c.real), to_native(qn)])

    basis_py: list[list[object]] = []
    for dof in ordered_electrode_dofs:
        basis_py.append(["half_spin", dof])
    nbas = numpy.maximum(16 * c2 / w**3, numpy.ones(len(w)) * 4)
    nbas = numpy.round(nbas).astype(int)
    for imode, omega in enumerate(w):
        basis_py.append(["sho", f"v_{imode}", float(omega), int(nbas[imode])])

    topology_py = {
        "schema": "topology_v1",
        "subtrees": [
            {
                "subtree_id": "left",
                "builder": "binary_mctdh",
                "basis_dofs": ordered_electrode_dofs[:s_idx],
                "dummy_label": "EL-dummy",
            },
            {
                "subtree_id": "right",
                "builder": "binary_mctdh",
                "basis_dofs": ordered_electrode_dofs[s_idx + 1 :],
                "dummy_label": "ER-dummy",
            },
            {
                "subtree_id": "phonon",
                "builder": "binary_mctdh",
                "basis_dofs": [f"v_{imode}" for imode in range(N_PH_MODE)],
                "dummy_label": "phonon-dummy",
            },
        ],
        "assembly": [
            {
                "node_id": "electrodes",
                "basis_items": [{"kind": "dummy", "label": "dummy"}],
                "children": ["left", "right"],
            },
            {
                "node_id": "root",
                "basis_items": [{"kind": "dof", "value": "s"}],
                "children": ["electrodes", "phonon"],
            },
        ],
        "root": "root",
    }

    hamiltonian_terms, hamiltonian_terms_node = run_process(
        define_hamiltonian_terms,
        op_spec=hamiltonian_terms_py,
    )
    basis, basis_node = run_process(define_basis, basis_spec=basis_py)
    topology, topology_node = run_process(build_topology, topology=topology_py)

    script_payload, script_node = run_process(
        build_ttn_script,
        input_params=input_params,
        model_params=model_params,
        calc_params=calc_params,
        op_spec=hamiltonian_terms,
        basis_spec=basis,
        topology=topology,
        real_run=REAL_RUN,
    )

    script_name = script_payload.get_dict()["script_name"]
    script_text = script_payload.get_dict()["script_text"]
    manifest, manifest_node = run_process(
        build_bundle_manifest,
        script_name=script_name,
        script_text=script_text,
        work_dir=WORK_DIR,
    )

    if not REAL_RUN:
        out = materialize_python_script_bundle_preview(
            example_file=__file__,
            work_dir=WORK_DIR,
            script_name=script_name,
            script_text=script_text,
            manifest=manifest,
        )
        if DEBUG_PROVENANCE:
            for label, node in [
                ("define_hamiltonian_terms", hamiltonian_terms_node),
                ("define_basis", basis_node),
                ("build_topology", topology_node),
                ("build_ttn_script", script_node),
                ("build_bundle_manifest", manifest_node),
            ]:
                if node is not None:
                    print(f"[{label}] pk={node.pk}")
        print(f"[preview] wrote 4 scripts to {out}")
        print(f"work_dir={WORK_DIR}")
        return

    outputs, node = run_process(
        BundleRunnerWorkChain,
        code=orm.load_code(CODE),
        manifest=manifest,
        fail_fast=FAIL_FAST,
        max_retries=MAX_RETRIES,
        resume_from_stage=RESUME_FROM_STAGE,
    )
    if DEBUG_PROVENANCE and node is not None:
        print(f"[BundleRunnerWorkChain] pk={node.pk}")
    print(outputs["output_parameters"].get_dict())


if __name__ == "__main__":
    main()
