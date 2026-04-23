from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from aiida import orm

from aiida_renormalizer.calculations.manifest_ops import bundle_manifest_for_python_script
from aiida_renormalizer.calculations.basic.bundle_runner import BundleRunnerCalcJob
from aiida_renormalizer.utils import run_process
from aiida_renormalizer.workchains.bundle_runner import BundleRunnerWorkChain

def materialize_python_script_bundle_preview(
    *,
    example_file: str | Path,
    work_dir: str,
    script_name: str,
    script_text: str,
    manifest: orm.Dict | dict[str, Any],
) -> Path:
    """Write preview artifacts for a generated single-file Python bundle."""
    manifest_payload = manifest.get_dict() if isinstance(manifest, orm.Dict) else dict(manifest)
    out = Path(example_file).with_name(work_dir)
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)

    (out / script_name).write_text(script_text)
    for index, stage in enumerate(manifest_payload["stages"], start=1):
        (out / f"{index:02d}_{stage['name']}.py").write_text(stage["script"])
    (out / "bundle_runner_driver.py").write_text(BundleRunnerCalcJob.render_driver_template_preview())
    return out


def run_script_bundle_example(
    *,
    example_file: str | Path,
    script_process,
    input_params: dict[str, Any],
    model_params: dict[str, Any],
    calc_params: dict[str, Any],
    work_dir: str,
    code_label: str,
    real_run: bool,
    debug_provenance: bool = False,
):
    """Run a case-specific script calcfunction and either preview or submit the bundle."""
    script_payload, script_node = run_process(
        script_process,
        debug_provenance=debug_provenance,
        input_params=orm.Dict(dict=input_params),
        model_params=orm.Dict(dict=model_params),
        calc_params=orm.Dict(dict=calc_params),
        real_run=orm.Bool(real_run),
    )
    script_dict = script_payload.get_dict()
    manifest, manifest_node = run_process(
        bundle_manifest_for_python_script,
        debug_provenance=debug_provenance,
        script_name=orm.Str(script_dict["script_name"]),
        script_text=orm.Str(script_dict["script_text"]),
        work_dir=orm.Str(work_dir),
    )

    if not real_run:
        out = materialize_python_script_bundle_preview(
            example_file=example_file,
            work_dir=work_dir,
            script_name=script_dict["script_name"],
            script_text=script_dict["script_text"],
            manifest=manifest,
        )
        if debug_provenance:
            for label, node in [
                (getattr(script_process, "__name__", "script_process"), script_node),
                ("bundle_manifest_for_python_script", manifest_node),
            ]:
                if node is not None:
                    print(f"[{label}] pk={node.pk}")
        print(f"[preview] wrote 4 scripts to {out}")
        print(f"work_dir={work_dir}")
        return out

    outputs, node = run_process(
        BundleRunnerWorkChain,
        debug_provenance=debug_provenance,
        code=orm.load_code(code_label),
        manifest=manifest,
        fail_fast=orm.Bool(True),
        max_retries=orm.Int(0),
        resume_from_stage=orm.Int(1),
    )
    if debug_provenance and node is not None:
        print(f"[BundleRunnerWorkChain] pk={node.pk}")
    print(outputs["output_parameters"].get_dict())
    return outputs
