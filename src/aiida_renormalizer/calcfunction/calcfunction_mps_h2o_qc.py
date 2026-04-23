"""MPS H2O QC case-scoped calcfunctions."""

from __future__ import annotations

import base64
from pathlib import Path
from pprint import pformat
from typing import Any

from aiida import orm
from aiida.engine import calcfunction, workfunction
from jinja2 import Environment, FileSystemLoader

_TEMPLATE_ENV = Environment(loader=FileSystemLoader(str(Path(__file__).resolve().parents[1] / "templates")))


def _render_stage_script(template_name: str, context: dict[str, Any]) -> str:
    return _TEMPLATE_ENV.get_template(template_name).render(**context).rstrip() + "\n"


def _validate_stage_payload(raw: list) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("stages must be a non-empty list")
    stages: list[dict] = []
    seen: set[str] = set()
    for i, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"stages[{i - 1}] must be a dict")
        name = str(item.get("name", "")).strip() or f"stage_{i:02d}"
        script = item.get("script")
        if name in seen:
            raise ValueError(f"duplicate stage name: {name}")
        if not isinstance(script, str) or not script.strip():
            raise ValueError(f"stages[{i - 1}].script must be a non-empty string")
        seen.add(name)
        stages.append({"name": name, "script": script})
    return stages


def build_bundle_manifest_payload(stages: list[dict]) -> dict:
    payload_stages = _validate_stage_payload(stages)
    return {"schema": "bundle_manifest_v1", "stage_count": len(payload_stages), "stages": payload_stages}


def render_python_script_bundle_manifest_payload(script_name: str, script_text: str, output_directory: str = "../generated_scripts") -> dict:
    if not script_name.strip() or not script_text.strip() or not output_directory.strip():
        raise ValueError("script_name/script_text/output_directory must be non-empty")
    script_text_b64 = base64.b64encode(script_text.encode("utf-8")).decode("ascii")
    output_path = f"{output_directory.rstrip('/')}/{script_name}"
    return build_bundle_manifest_payload(
        [
            {
                "name": "write_generated_script",
                "script": _render_stage_script(
                    "mps_h2o_qc_write_generated_script_stage.py.jinja",
                    {"output_path": output_path, "script_text_b64": script_text_b64},
                ),
            },
            {
                "name": "compile_generated_script",
                "script": _render_stage_script("mps_h2o_qc_compile_generated_script_stage.py.jinja", {"output_path": output_path}),
            },
        ]
    )


def _render_case_script_payload(*, template_name: str, script_name: str, input_params: orm.Dict, model_params: orm.Dict, calc_params: orm.Dict, real_run: orm.Bool) -> dict[str, str]:
    script_text = _render_stage_script(
        template_name,
        {
            "input_literal": pformat(dict(input_params.get_dict()), sort_dicts=False),
            "model_literal": pformat(dict(model_params.get_dict()), sort_dicts=False),
            "calc_literal": pformat(dict(calc_params.get_dict()), sort_dicts=False),
            "real_run_literal": "True" if bool(real_run.value) else "False",
        },
    )
    return {"script_name": script_name, "script_text": script_text}


@calcfunction
def _materialize_bundle_manifest(stages: orm.List) -> orm.Dict:
    return orm.Dict(dict=build_bundle_manifest_payload(stages.get_list()))


@workfunction
def bundle_manifest_for_python_script(script_name: orm.Str, script_text: orm.Str, work_dir: orm.Str) -> orm.Dict:
    payload = render_python_script_bundle_manifest_payload(script_name.value, script_text.value, work_dir.value)
    return _materialize_bundle_manifest(orm.List(list=payload["stages"]))


@calcfunction
def mps_h2o_qc_script(input_params: orm.Dict, model_params: orm.Dict, calc_params: orm.Dict, real_run: orm.Bool) -> orm.Dict:
    return orm.Dict(
        dict=_render_case_script_payload(
            template_name="example_mps_h2o_qc_single_file.py.jinja",
            script_name="mps_h2o_qc_generated.py",
            input_params=input_params,
            model_params=model_params,
            calc_params=calc_params,
            real_run=real_run,
        )
    )


build_mps_script = mps_h2o_qc_script
build_bundle_manifest = bundle_manifest_for_python_script
