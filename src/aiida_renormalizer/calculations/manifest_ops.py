"""Minimal shared calcfunctions kept for production call paths.

Case-specific calcfunctions live under ``src/aiida_renormalizer/calcfunction/``.
This module intentionally only keeps utilities still imported by production code.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

from aiida import orm
from aiida.engine import calcfunction, workfunction
from jinja2 import Environment, FileSystemLoader

_TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(str(Path(__file__).resolve().parents[1] / "templates")),
)


def _render_stage_script(template_name: str, context: dict[str, Any]) -> str:
    template = _TEMPLATE_ENV.get_template(template_name)
    return template.render(**context).rstrip() + "\n"


def _validate_stage_payload(raw: list) -> list[dict]:
    if not isinstance(raw, list) or not raw:
        raise ValueError("stages must be a non-empty list")

    manifest_stages: list[dict] = []
    seen: set[str] = set()
    for index, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"stages[{index - 1}] must be a dict")

        name = str(item.get("name", "")).strip() or f"stage_{index:02d}"
        script = item.get("script")

        if name in seen:
            raise ValueError(f"duplicate stage name: {name}")
        if not isinstance(script, str) or not script.strip():
            raise ValueError(f"stages[{index - 1}].script must be a non-empty string")

        seen.add(name)
        manifest_stages.append({"name": name, "script": script})

    return manifest_stages


def build_bundle_manifest_payload(stages: list[dict]) -> dict:
    manifest_stages = _validate_stage_payload(stages)
    return {
        "schema": "bundle_manifest_v1",
        "stage_count": len(manifest_stages),
        "stages": manifest_stages,
    }


def render_python_script_bundle_manifest_payload(
    script_name: str,
    script_text: str,
    output_directory: str = "../generated_scripts",
) -> dict:
    if not script_name.strip():
        raise ValueError("script_name must be non-empty")
    if not script_text.strip():
        raise ValueError("script_text must be non-empty")
    if not output_directory.strip():
        raise ValueError("output_directory must be non-empty")

    script_text_b64 = base64.b64encode(script_text.encode("utf-8")).decode("ascii")
    output_path = f"{output_directory.rstrip('/')}/{script_name}"
    stages = [
        {
            "name": "write_generated_script",
            "script": _render_stage_script(
                "write_generated_script_stage.py.jinja",
                {"output_path": output_path, "script_text_b64": script_text_b64},
            ),
        },
        {
            "name": "compile_generated_script",
            "script": _render_stage_script(
                "compile_generated_script_stage.py.jinja",
                {"output_path": output_path},
            ),
        },
    ]
    return build_bundle_manifest_payload(stages)


@calcfunction
def _materialize_bundle_manifest(stages: orm.List) -> orm.Dict:
    return orm.Dict(dict=build_bundle_manifest_payload(stages.get_list()))


@workfunction
def generate_bundle_manifest(stages: orm.List) -> orm.Dict:
    """Build a manifest from explicit stage scripts."""
    _validate_stage_payload(stages.get_list())
    return _materialize_bundle_manifest(stages)


@workfunction
def bundle_manifest_for_python_script(
    script_name: orm.Str,
    script_text: orm.Str,
    work_dir: orm.Str,
) -> orm.Dict:
    """Build a two-stage manifest for a generated python script."""
    payload = render_python_script_bundle_manifest_payload(
        script_name=script_name.value,
        script_text=script_text.value,
        output_directory=work_dir.value,
    )
    return _materialize_bundle_manifest(orm.List(list=payload["stages"]))
