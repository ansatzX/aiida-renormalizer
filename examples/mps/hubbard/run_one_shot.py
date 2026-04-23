#!/usr/bin/env python
"""MPS Hubbard script-generation example."""

from __future__ import annotations

from aiida import load_profile, orm

from aiida_renormalizer.calcfunction.calcfunction_mps_hubbard import (
    build_bundle_manifest,
    build_mps_script,
    define_basis,
    define_hamiltonian_terms,
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

# INPUT
NSITES = 10
T = -1.0
U = 4.0

# MODEL
NELEC = [5, 5]
M_MAX = 100

# CALC
WORKFLOW_NAME = "hubbard_ground_state"
METHOD = "2site"


def main() -> None:
    input_params = {"nsites": NSITES, "t": T, "U": U}
    model_params = {"nelec": NELEC, "m_max": M_MAX}
    calc_params = {"workflow_name": WORKFLOW_NAME, "method": METHOD}

    qn_up = {"+": [-1, 0], "-": [1, 0], "Z": [0, 0]}
    qn_do = {"+": [0, -1], "-": [0, 1], "Z": [0, 0]}

    hamiltonian_terms_py: list[list[object]] = []
    for i in range(2 * (NSITES - 1)):
        if i % 2 == 0:
            qn1 = [qn_up["Z"], qn_up["+"], qn_do["Z"], qn_up["-"]]
            qn2 = [qn_up["Z"], qn_up["-"], qn_do["Z"], qn_up["+"]]
        else:
            qn1 = [qn_do["Z"], qn_do["+"], qn_up["Z"], qn_do["-"]]
            qn2 = [qn_do["Z"], qn_do["-"], qn_up["Z"], qn_do["+"]]
        hamiltonian_terms_py.extend(
            [
                ["Z + Z -", [i, i, i + 1, i + 2], T, qn1],
                ["Z - Z +", [i, i, i + 1, i + 2], -T, qn2],
            ]
        )

    for i in range(0, 2 * NSITES, 2):
        qn = [qn_up["-"], qn_up["+"], qn_do["-"], qn_do["+"]]
        hamiltonian_terms_py.append(["- + - +", [i, i, i + 1, i + 1], U, qn])

    basis_py: list[list[object]] = []
    for i in range(2 * NSITES):
        if i % 2 == 0:
            sigmaqn = [[0, 0], [1, 0]]
        else:
            sigmaqn = [[0, 0], [0, 1]]
        basis_py.append(["half_spin", i, sigmaqn])

    hamiltonian_terms, hamiltonian_terms_node = run_process(
        define_hamiltonian_terms,
        op_spec=hamiltonian_terms_py,
    )
    basis, basis_node = run_process(define_basis, basis_spec=basis_py)

    script_payload, script_node = run_process(
        build_mps_script,
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
                ("define_hamiltonian_terms", hamiltonian_terms_node),
                ("define_basis", basis_node),
                ("build_mps_script", script_node),
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
