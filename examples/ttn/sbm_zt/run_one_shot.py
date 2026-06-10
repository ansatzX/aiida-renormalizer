#!/usr/bin/env python
"""TTN zero-temperature script-generation example."""

from __future__ import annotations

import os

from aiida import load_profile, orm

from aiida_renormalizer.calcfunction.calcfunction_ttn_sbm_zt import (
    ColeDavidsonSDF_setup,
    build_bundle_manifest,
    build_time_evolution_section,
    build_ttn_model,
)
from aiida_renormalizer.example_support import materialize_python_script_bundle_preview
from aiida_renormalizer.utils import run_process
from aiida_renormalizer.workchains.bundle_runner import BundleRunnerWorkChain

load_profile()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


CODE = "reno-script-clean@localhost"
WORK_DIR = "generated_scripts"
REAL_RUN = _env_bool("AIIDA_RENO_SBM_ZT_REAL_RUN", True)
DEBUG_PROVENANCE = False
FAIL_FAST = True
MAX_RETRIES = 0
RESUME_FROM_STAGE = 1

# INPUT: spectral density information and mode count.
ITA = 1.0
OMEGA_C = 0.1
BETA = 0.5
RAW_DELTA = 1.0
N_MODES = 1000

# INPUT: system operator definitions.
EPSILON = 0.0
SPIN_DOF = "spin"
SPIN_SIGMAQN = [0, 0]
MODE_DOF_PREFIX = "v_"
SYSTEM_TERMS = [
    ["sigma_z", SPIN_DOF, EPSILON, 0],
    ["sigma_x", SPIN_DOF, "delta_eff", 0],
]

# BUILD MODEL: tensor-network construction choices.
TREE_TYPE = "binary"
M_MAX = 20
UPPER_LIMIT = 30.0

# CALC: dynamics settings.
DT = 0.2
NSTEPS = 200
METHOD = "tdvp-ps"
OBSERVATIONS = [
    {"label": "sigma_z", "symbol": "sigma_z", "dofs": SPIN_DOF, "qn": 0},
    {"label": "sigma_x", "symbol": "sigma_x", "dofs": SPIN_DOF, "qn": 0},
]


# Workflow wiring below this line.
def main() -> None:
    # Build environment information from the spectral density.
    env, env_node = run_process(
        ColeDavidsonSDF_setup,
        ita=ITA,
        omega_c=OMEGA_C,
        beta=BETA,
        upper_limit=UPPER_LIMIT,
        raw_delta=RAW_DELTA,
        n_modes=N_MODES,
    )

    # Build model locally: make the symbolic Hamiltonian and basis explicit.
    omega_k = env.get_array("omega_k").tolist()
    c_j2 = env.get_array("c_j2").tolist()

    hamiltonian_terms_py: list[list[object]] = list(SYSTEM_TERMS)
    for imode, omega in enumerate(omega_k):
        mode_dof = f"{MODE_DOF_PREFIX}{imode}"
        hamiltonian_terms_py.extend(
            [
                ["p^2", mode_dof, 0.5, 0],
                ["x^2", mode_dof, 0.5 * omega**2, 0],
            ]
        )

    for imode, coupling in enumerate(c_j2):
        mode_dof = f"{MODE_DOF_PREFIX}{imode}"
        hamiltonian_terms_py.append(
            ["sigma_z x", [SPIN_DOF, mode_dof], coupling**0.5, [0, 0]]
        )

    basis_py: list[list[object]] = [["half_spin", SPIN_DOF, SPIN_SIGMAQN]]
    for imode, omega in enumerate(omega_k):
        mode_dof = f"{MODE_DOF_PREFIX}{imode}"
        safe_omega = max(float(omega), 1e-12)
        nbas = int(round(max(16 * float(c_j2[imode]) / safe_omega**3, 4.0)))
        basis_py.append(["sho", mode_dof, omega, nbas])

    model_section, model_section_node = run_process(
        build_ttn_model,
        hamiltonian_terms=hamiltonian_terms_py,
        basis=basis_py,
        tree_type=TREE_TYPE,
        m_max=M_MAX,
    )

    # Render the concrete time-evolution block; observations stay visible above.
    calculation_section, calculation_section_node = run_process(
        build_time_evolution_section,
        dt=DT,
        nsteps=NSTEPS,
        method=METHOD,
        observations=OBSERVATIONS,
    )

    # Render the final TTN script and package it into one execution bundle.
    bundle_outputs, bundle_node = run_process(
        build_bundle_manifest,
        environment=env,
        model_section=model_section,
        calculation_section=calculation_section,
        real_run=REAL_RUN,
        work_dir=WORK_DIR,
    )
    script_payload = bundle_outputs["script_payload"].get_dict()
    script_name = script_payload["script_name"]
    script_text = script_payload["script_text"]
    manifest = bundle_outputs["manifest"]

    out = materialize_python_script_bundle_preview(
        example_file=__file__,
        work_dir=WORK_DIR,
        script_name=script_name,
        script_text=script_text,
        manifest=manifest,
    )
    if DEBUG_PROVENANCE:
        for label, node in [
            ("ColeDavidsonSDF_setup", env_node),
            ("build_ttn_model", model_section_node),
            ("build_time_evolution_section", calculation_section_node),
            ("build_bundle_manifest", bundle_node),
        ]:
            if node is not None:
                print(f"[{label}] pk={node.pk}")
    print(f"[preview] wrote bundle scripts to {out}")
    print(f"work_dir={WORK_DIR}")
    if not REAL_RUN:
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
