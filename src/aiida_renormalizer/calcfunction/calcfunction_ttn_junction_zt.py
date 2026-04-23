"""TTN junction ZT case-scoped calcfunctions."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from pprint import pformat
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
                "ttn_junction_zt_write_generated_script_stage.py.jinja",
                {"output_path": output_path, "script_text_b64": script_text_b64},
            ),
        },
        {
            "name": "compile_generated_script",
            "script": _render_stage_script(
                "ttn_junction_zt_compile_generated_script_stage.py.jinja",
                {"output_path": output_path},
            ),
        },
    ]
    return build_bundle_manifest_payload(stages)


@calcfunction
def _materialize_bundle_manifest(stages: orm.List) -> orm.Dict:
    return orm.Dict(dict=build_bundle_manifest_payload(stages.get_list()))


@workfunction
def bundle_manifest_for_python_script(
    script_name: orm.Str,
    script_text: orm.Str,
    work_dir: orm.Str,
) -> orm.Dict:
    payload = render_python_script_bundle_manifest_payload(
        script_name=script_name.value,
        script_text=script_text.value,
        output_directory=work_dir.value,
    )
    return _materialize_bundle_manifest(orm.List(list=payload["stages"]))


def _is_supported_dof_atom(obj: Any) -> bool:
    return isinstance(obj, (str, int)) or (
        isinstance(obj, tuple) and all(_is_supported_dof_atom(item) for item in obj)
    )


def _normalize_op_specs(raw_terms: Any) -> list[dict[str, Any]]:
    if not isinstance(raw_terms, list) or not raw_terms:
        raise ValueError("operator specs must be a non-empty list")

    normalized_terms: list[dict[str, Any]] = []
    for raw_term in raw_terms:
        if isinstance(raw_term, dict):
            symbol = str(raw_term["symbol"])
            dofs = raw_term["dofs"]
            factor = raw_term["factor"]
            qn = raw_term.get("qn", 0)
        else:
            if not isinstance(raw_term, list) or len(raw_term) not in (3, 4):
                raise ValueError(
                    "each operator spec item must be [symbol, dofs, factor] or [symbol, dofs, factor, qn]"
                )
            symbol, dofs, factor = raw_term[:3]
            qn = raw_term[3] if len(raw_term) == 4 else 0
            symbol = str(symbol)

        if not _is_supported_dof_atom(dofs) and not (
            isinstance(dofs, list) and all(_is_supported_dof_atom(item) for item in dofs)
        ):
            raise ValueError("operator_specs[].dofs must be a supported dof atom or list of dof atoms")

        normalized_terms.append(
            {
                "symbol": symbol,
                "dofs": dofs,
                "factor": factor,
                "qn": qn,
            }
        )
    return normalized_terms


def _render_python_dof_atom_expr(dof: Any, *, spin_dof: Any | None = None) -> str:
    if isinstance(dof, str):
        if spin_dof is not None and dof == spin_dof:
            return "spin_dof"
        return json.dumps(dof)
    if isinstance(dof, int):
        return repr(dof)
    if isinstance(dof, tuple):
        inner = ", ".join(_render_python_dof_atom_expr(item, spin_dof=spin_dof) for item in dof)
        if len(dof) == 1:
            inner += ","
        return "(" + inner + ")"
    raise TypeError(f"unsupported dof atom for rendering: {dof!r}")


def _render_python_dofs_expr(dofs: Any, *, spin_dof: Any | None = None) -> str:
    if _is_supported_dof_atom(dofs):
        return _render_python_dof_atom_expr(dofs, spin_dof=spin_dof)
    return "[" + ", ".join(_render_python_dof_atom_expr(item, spin_dof=spin_dof) for item in dofs) + "]"


def _render_hamiltonian_terms_block(term_specs: list[dict[str, Any]], *, spin_dof: Any | None = None) -> str:
    rendered_terms = []
    for item in term_specs:
        factor = item["factor"]
        factor_literal = factor if isinstance(factor, str) else repr(factor)
        rendered_terms.append(
            "Op("
            f"{json.dumps(item['symbol'])}, "
            f"{_render_python_dofs_expr(item['dofs'], spin_dof=spin_dof)}, "
            f"factor={factor_literal}, "
            f"qn={repr(item.get('qn', 0))}"
            ")"
        )
    return "ham_terms.extend([\n        " + ",\n        ".join(rendered_terms) + "\n    ])"


def _render_basis_spec_block(basis_specs: list[dict[str, Any]], *, spin_dof: Any | None = None) -> str:
    rendered_items = []
    for item in basis_specs:
        kind = str(item["kind"])
        dof = _render_python_dof_atom_expr(item["dof"], spin_dof=spin_dof)
        if kind == "simple_electron":
            rendered_items.append(f"ba.BasisSimpleElectron({dof})")
            continue
        if kind == "half_spin":
            if "sigmaqn" in item:
                rendered_items.append(f"ba.BasisHalfSpin({dof}, {repr(item['sigmaqn'])})")
            else:
                rendered_items.append(f"ba.BasisHalfSpin({dof})")
            continue
        if kind == "sho":
            omega_literal = item["omega"] if isinstance(item["omega"], str) else repr(item["omega"])
            rendered_items.append(
                f"ba.BasisSHO({dof}, {omega_literal}, {int(item['nbas'])})"
            )
            continue
        raise ValueError(f"unsupported basis kind in renderer: {kind}")
    return "basis = [\n        " + ",\n        ".join(rendered_items) + "\n    ]"


@calcfunction
def define_op_spec(op_spec: orm.List):
    from aiida_renormalizer.data import OpSpecData

    normalized_terms = _normalize_op_specs(op_spec.get_list())
    return OpSpecData.from_list(normalized_terms)


@calcfunction
def define_basis_spec(basis_spec: orm.List):
    from aiida_renormalizer.data import BasisSpecData

    return BasisSpecData.from_list(basis_spec.get_list())


@calcfunction
def define_topology(topology: orm.Dict):
    from aiida_renormalizer.data import TopologyData

    return TopologyData.from_dict(topology.get_dict())


def _render_case_script_from_spec_topology_payload(
    *,
    template_name: str,
    script_name: str,
    input_params: orm.Dict,
    model_params: orm.Dict,
    calc_params: orm.Dict,
    op_spec,
    basis_spec,
    topology,
    real_run: orm.Bool,
    spin_dof: Any | None = None,
    extra_context: dict[str, Any] | None = None,
) -> dict[str, str]:
    context = {
        "input_literal": pformat(dict(input_params.get_dict()), sort_dicts=False),
        "model_literal": pformat(dict(model_params.get_dict()), sort_dicts=False),
        "calc_literal": pformat(dict(calc_params.get_dict()), sort_dicts=False),
        "real_run_literal": "True" if bool(real_run.value) else "False",
        "hamiltonian_terms_block": _render_hamiltonian_terms_block(op_spec.as_list(), spin_dof=spin_dof),
        "basis_spec_block": _render_basis_spec_block(basis_spec.as_list(), spin_dof=spin_dof),
        "topology_literal": pformat(topology.as_dict(), sort_dicts=False),
    }
    if extra_context:
        context.update(extra_context)
    script_text = _render_stage_script(template_name, context)
    return {
        "script_name": script_name,
        "script_text": script_text,
    }


@calcfunction
def ttn_junction_zt_script_from_spec_topology(
    input_params: orm.Dict,
    model_params: orm.Dict,
    calc_params: orm.Dict,
    op_spec,
    basis_spec,
    topology,
    real_run: orm.Bool,
) -> orm.Dict:
    return orm.Dict(
        dict=_render_case_script_from_spec_topology_payload(
            template_name="example_ttn_junction_zt_from_spec_topology_single_file.py.jinja",
            script_name="ttn_junction_zt_generated.py",
            input_params=input_params,
            model_params=model_params,
            calc_params=calc_params,
            op_spec=op_spec,
            basis_spec=basis_spec,
            topology=topology,
            real_run=real_run,
            spin_dof="s",
        )
    )


# Generic component names for example scripts.
build_bundle_manifest = bundle_manifest_for_python_script
define_hamiltonian_terms = define_op_spec
define_basis = define_basis_spec
build_topology = define_topology
build_ttn_script = ttn_junction_zt_script_from_spec_topology
