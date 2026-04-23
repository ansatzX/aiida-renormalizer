#!/usr/bin/env python
"""MPS SBM script-generation example."""

from __future__ import annotations

from aiida import load_profile, orm

from aiida_renormalizer.calcfunction.calcfunction_mps_sbm import (
    build_bundle_manifest,
    build_mps_script,
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
ALPHA = 0.05
RAW_DELTA = 1.0
RAW_OMEGA_C = 20.0
N_PHONONS = 300

# MODEL
RENORMALIZATION_P = 1.0

# CALC
WORKFLOW_NAME = "sbm_dynamics"
EVOLVE_DT = 0.1
EVOLVE_TIME = 20


def main() -> None:
    input_params = {
        "alpha": ALPHA,
        "raw_delta": RAW_DELTA,
        "raw_omega_c": RAW_OMEGA_C,
        "n_phonons": N_PHONONS,
    }
    model_params = {"renormalization_p": RENORMALIZATION_P}
    calc_params = {
        "workflow_name": WORKFLOW_NAME,
        "evolve_dt": EVOLVE_DT,
        "evolve_time": EVOLVE_TIME,
    }

    script_payload, script_node = run_process(
        build_mps_script,
        input_params=input_params,
        model_params=model_params,
        calc_params=calc_params,
        real_run=REAL_RUN,
    )
    script_dict = script_payload.get_dict()
    manifest, manifest_node = run_process(
        build_bundle_manifest,
        script_name=script_dict["script_name"],
        script_text=script_dict["script_text"],
        work_dir=WORK_DIR,
    )

    if not REAL_RUN:
        out = materialize_python_script_bundle_preview(
            example_file=__file__,
            work_dir=WORK_DIR,
            script_name=script_dict["script_name"],
            script_text=script_dict["script_text"],
            manifest=manifest,
        )
        if DEBUG_PROVENANCE:
            for label, node in [
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
