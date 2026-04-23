#!/usr/bin/env python
"""MPS SSH script-generation example."""

from __future__ import annotations

from aiida import load_profile, orm
from renormalizer.model.model import construct_j_matrix
from renormalizer.model.op import Op
from renormalizer.utils import Quantity

from aiida_renormalizer.calcfunction.calcfunction_mps_ssh import (
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
NSITES = 2
G = 0.7
W0 = 0.5
T = -1.0

# MODEL
NBOSON_MAX = 4
BOND_DIM = 16
NSWEEPS = 10
PERIODIC = True

# CALC
WORKFLOW_NAME = "ssh_ground_state"
METHOD = "2site"


def main() -> None:
    input_params = {"nsites": NSITES, "g": G, "w0": W0, "t": T}
    model_params = {
        "nboson_max": NBOSON_MAX,
        "bond_dim": BOND_DIM,
        "nsweeps": NSWEEPS,
        "periodic": PERIODIC,
    }
    calc_params = {"workflow_name": WORKFLOW_NAME, "method": METHOD}

    j_matrix = construct_j_matrix(NSITES, Quantity(T), PERIODIC)
    ops = []

    for imol in range(NSITES):
        for jmol in range(NSITES):
            if j_matrix[imol, jmol] != 0:
                ops.append(Op(r"a^\\dagger a", [imol, jmol], j_matrix[imol, jmol]))
        ops.append(Op(r"b^\\dagger b", (imol, 0), W0))

    for imol in range(NSITES - 1):
        ops.append(Op(r"a^\\dagger a", [imol, imol + 1], G) * Op(r"b^\\dagger+b", (imol + 1, 0)))
        ops.append(Op(r"a^\\dagger a", [imol, imol + 1], -G) * Op(r"b^\\dagger+b", (imol, 0)))
        ops.append(Op(r"a^\\dagger a", [imol + 1, imol], G) * Op(r"b^\\dagger+b", (imol + 1, 0)))
        ops.append(Op(r"a^\\dagger a", [imol + 1, imol], -G) * Op(r"b^\\dagger+b", (imol, 0)))

    if PERIODIC:
        last = NSITES - 1
        ops.append(Op(r"a^\\dagger a", [last, 0], G) * Op(r"b^\\dagger+b", (0, 0)))
        ops.append(Op(r"a^\\dagger a", [last, 0], -G) * Op(r"b^\\dagger+b", (last, 0)))
        ops.append(Op(r"a^\\dagger a", [0, last], G) * Op(r"b^\\dagger+b", (0, 0)))
        ops.append(Op(r"a^\\dagger a", [0, last], -G) * Op(r"b^\\dagger+b", (last, 0)))

    serialized_opsum = [list(op.to_tuple()) for op in ops]

    basis_py: list[list[object]] = []
    for imol in range(NSITES):
        basis_py.append(["simple_electron", imol])
        basis_py.append(["sho", (imol, 0), W0, NBOSON_MAX])

    hamiltonian_terms, hamiltonian_terms_node = run_process(
        define_hamiltonian_terms,
        serialized_opsum=serialized_opsum,
    )
    basis, basis_node = run_process(define_basis, basis_spec=basis_py)

    script_payload, script_node = run_process(
        build_mps_script,
        input_params=input_params,
        model_params=model_params,
        calc_params=calc_params,
        op_data=hamiltonian_terms,
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
