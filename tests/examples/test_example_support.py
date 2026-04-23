from __future__ import annotations

from pathlib import Path

import numpy as np
from aiida import orm

from aiida_renormalizer.calcfunction.calcfunction_ttn_sbm_zt import (
    ColeDavidsonSDF_setup,
    bundle_manifest_for_python_script,
)
from aiida_renormalizer.example_support import materialize_python_script_bundle_preview
from aiida_renormalizer.utils import run_process


def test_materialize_python_script_bundle_preview_writes_clean_preview_directory(aiida_profile, tmp_path):
    example_file = tmp_path / "run_one_shot.py"
    example_file.write_text("# preview target\n")

    preview_dir = example_file.with_name("generated_scripts")
    preview_dir.mkdir()
    (preview_dir / "stale.txt").write_text("old\n")

    manifest = bundle_manifest_for_python_script(
        script_name=orm.Str("demo.py"),
        script_text=orm.Str("print('ok')\n"),
        work_dir=orm.Str("generated_scripts"),
    )

    out = materialize_python_script_bundle_preview(
        example_file=example_file,
        work_dir="generated_scripts",
        script_name="demo.py",
        script_text="print('ok')\n",
        manifest=manifest,
    )

    assert out == preview_dir
    assert not (out / "stale.txt").exists()
    assert (out / "demo.py").read_text() == "print('ok')\n"
    assert (out / "01_write_generated_script.py").exists()
    assert (out / "02_compile_generated_script.py").exists()
    assert (out / "bundle_runner_driver.py").exists()


def test_run_process_coerces_python_and_numpy_inputs(aiida_profile):
    manifest, _ = run_process(
        bundle_manifest_for_python_script,
        script_name="demo.py",
        script_text="print('ok')\n",
        work_dir=np.str_("generated_scripts"),
    )
    assert manifest.get_dict()["stage_count"] == 2

    environment, _ = run_process(
        ColeDavidsonSDF_setup,
        ita=np.float64(1.0),
        omega_c=0.1,
        beta=np.float32(0.5),
        upper_limit=30.0,
        raw_delta=np.float64(1.0),
        n_modes=np.int64(4),
    )
    assert environment.base.attributes.get("n_modes") == 4
