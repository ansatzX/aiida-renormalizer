#!/usr/bin/env python
"""TTN finite-temperature script-generation example."""

from __future__ import annotations

import numpy
from aiida import load_profile, orm

from aiida_renormalizer.calcfunction.calcfunction_ttn_sbm_ft import (
    build_bundle_manifest,
    build_environment_modes,
    build_ttn_script,
    define_basis,
    define_hamiltonian_terms,
    extract_spectral_density_parameters,
    gather_known_parameters,
)
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

# INPUT: spectral density information and mode count.
ITA = 1.0
OMEGA_C = 1.0
BETA = 0.25
TEMPERATURE = 2.0
RAW_DELTA = 1.0
N_MODES = 1000
UPPER_LIMIT = 30.0

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
M_MAX = 20

# CALC: dynamics settings.
DT = 0.1
NSTEPS = 400
METHOD = "tdvp_ps"


# Workflow wiring below this line.
def main() -> None:
    input_params = {
        "ita": ITA,
        "omega_c": OMEGA_C,
        "beta": BETA,
        "temperature": TEMPERATURE,
    }
    model_params = {
        "epsilon": EPSILON,
        "raw_delta": RAW_DELTA,
        "n_modes": N_MODES,
        "m_max": M_MAX,
        "upper_limit": UPPER_LIMIT,
    }
    calc_params = {
        "workflow_name": "ttn_sbm_ft",
        "dt": DT,
        "nsteps": NSTEPS,
        "method": METHOD,
    }

    known_parameters, known_node = run_process(
        gather_known_parameters,
        input_params=input_params,
        model_params=model_params,
        calc_params=calc_params,
        system_hamiltonian_terms=SYSTEM_TERMS,
    )
    spectral_density_parameters, spectral_node = run_process(
        extract_spectral_density_parameters,
        known_parameters=known_parameters,
    )
    renormalized_discretized_modes, renorm_node = run_process(
        build_environment_modes,
        known_parameters=known_parameters,
        spectral_density_parameters=spectral_density_parameters,
    )

    mode_data = renormalized_discretized_modes.get_dict()
    omega_k = numpy.asarray(mode_data["omega_k"], dtype=float)
    c_j2 = numpy.asarray(mode_data["c_j2"], dtype=float)
    c = numpy.sqrt(c_j2)
    theta_array = numpy.arctanh(numpy.exp(-omega_k / TEMPERATURE / 2))

    # Build model: merge system, environment, and coupling terms into one Hamiltonian.
    hamiltonian_terms_py: list[list[object]] = list(SYSTEM_TERMS)
    for imode, omega in enumerate(omega_k):
        hamiltonian_terms_py.extend(
            [
                ["p^2", f"{MODE_DOF_PREFIX}{imode}_p", 0.5, 0],
                ["x^2", f"{MODE_DOF_PREFIX}{imode}_p", 0.5 * float(omega) ** 2, 0],
                ["p^2", f"{MODE_DOF_PREFIX}{imode}_q", -0.5, 0],
                ["x^2", f"{MODE_DOF_PREFIX}{imode}_q", -0.5 * float(omega) ** 2, 0],
                [
                    "sigma_z x",
                    [SPIN_DOF, f"{MODE_DOF_PREFIX}{imode}_p"],
                    float(numpy.cosh(theta_array[imode]) * c[imode]),
                    [0, 0],
                ],
                [
                    "sigma_z x",
                    [SPIN_DOF, f"{MODE_DOF_PREFIX}{imode}_q"],
                    float(numpy.sinh(theta_array[imode]) * c[imode]),
                    [0, 0],
                ],
            ]
        )

    basis_py: list[list[object]] = [["half_spin", SPIN_DOF, SPIN_SIGMAQN]]
    nbas = numpy.maximum(16 * c**2 / omega_k**3 * numpy.cosh(theta_array) ** 2, numpy.ones(len(omega_k)) * 4)
    nbas = numpy.minimum(nbas, numpy.ones(len(omega_k)) * 512)
    nbas = numpy.round(nbas).astype(int) * 2
    for imode, omega in enumerate(omega_k):
        basis_py.append(["sho", f"{MODE_DOF_PREFIX}{imode}_p", float(omega), int(nbas[imode])])
        basis_py.append(["sho", f"{MODE_DOF_PREFIX}{imode}_q", float(omega), int(nbas[imode])])

    hamiltonian_terms, hamiltonian_terms_node = run_process(
        define_hamiltonian_terms,
        op_spec=hamiltonian_terms_py,
    )
    basis, basis_node = run_process(define_basis, basis_spec=basis_py)

    script_payload, script_node = run_process(
        build_ttn_script,
        input_params=input_params,
        model_params=model_params,
        calc_params=calc_params,
        op_spec=hamiltonian_terms,
        basis_spec=basis,
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
                ("gather_known_parameters", known_node),
                ("extract_spectral_density_parameters", spectral_node),
                ("build_environment_modes", renorm_node),
                ("define_hamiltonian_terms", hamiltonian_terms_node),
                ("define_basis", basis_node),
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
