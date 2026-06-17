#!/usr/bin/env python
"""TTN zero-temperature script-generation example."""

from __future__ import annotations

import os

from aiida import load_profile

from aiida_renormalizer.calcfunction.calcfunction_ttn_sbm_zt import (
    ColeDavidsonSDF_setup,
    build_bundle_manifest,
    build_time_evolution_section,
    build_ttn_model,
)
from aiida_renormalizer.example_support import materialize_python_script_bundle_preview
from aiida_renormalizer.utils import run_process

load_profile()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


WORK_DIR = "generated_scripts"
REAL_RUN = _env_bool("AIIDA_RENO_SBM_ZT_REAL_RUN", False)
DEBUG_PROVENANCE = False

# INPUT: spectral density information and mode count.
ITA = 1.0
OMEGA_C = 0.1
BETA = 0.5
N_MODES = 1000

# INPUT: system operator definitions.
EPSILON = 0.0
DELTA = 1.0
SPIN_DOF = "spin"
SPIN_SIGMAQN = [0, 0]
MODE_DOF_PREFIX = "v_"
SYSTEM_TERMS = [
    ["sigma_z", SPIN_DOF, EPSILON, 0],
    ["sigma_x", SPIN_DOF, DELTA, 0],
]

# BUILD MODEL: tensor-network construction choices.
TREE_TYPE = "binary"
M_MAX = 20
UPPER_LIMIT = 30.0

# CALC: dynamics settings.
DT = 0.2
NSTEPS = 200
METHOD = "tdvp-ps"
OBSERVATION_TERMS = [
    ["sigma_z", SPIN_DOF, 1.0, 0],
    ["sigma_x", SPIN_DOF, 1.0, 0],
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
        n_modes=N_MODES,
    )

    # Build model locally: make the symbolic Hamiltonian and basis explicit.
    omega_k = env.get_array("omega_k").tolist()
    c_j2 = env.get_array("c_j2").tolist()
    renormalization_constant = float(env.base.attributes.get("renormalization_constant"))
    delta_eff = DELTA * renormalization_constant

    hamiltonian_terms_py: list[list[object]] = [
        ["sigma_z", SPIN_DOF, EPSILON, 0],
        ["sigma_x", SPIN_DOF, delta_eff, 0],
    ]
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

    # Record deterministic rendering steps through calcfunctions.
    model_section, model_section_node = run_process(
        build_ttn_model,
        hamiltonian_terms=hamiltonian_terms_py,
        basis=basis_py,
        tree_type=TREE_TYPE,
        m_max=M_MAX,
    )
    calculation_section, calculation_section_node = run_process(
        build_time_evolution_section,
        dt=DT,
        nsteps=NSTEPS,
        method=METHOD,
        observations=OBSERVATION_TERMS,
    )
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
    print(f"[preview] wrote generated script bundle to {out}")
    print(f"work_dir={WORK_DIR}")
    if REAL_RUN:
        print("[calcfunction-only] generated script materialized; execute it directly for calculation.")


if __name__ == "__main__":
    main()
