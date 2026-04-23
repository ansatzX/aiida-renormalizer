"""TTN SBM ZT case-scoped calcfunctions."""

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
                "ttn_sbm_zt_write_generated_script_stage.py.jinja",
                {"output_path": output_path, "script_text_b64": script_text_b64},
            ),
        },
        {
            "name": "compile_generated_script",
            "script": _render_stage_script(
                "ttn_sbm_zt_compile_generated_script_stage.py.jinja",
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
def ColeDavidsonSDF_setup(
    ita: orm.Float,
    omega_c: orm.Float,
    beta: orm.Float,
    upper_limit: orm.Float,
    raw_delta: orm.Float,
    n_modes: orm.Int,
) -> orm.ArrayData:
    """Build renormalized/discretized environment data from a Cole-Davidson spectral density."""
    import numpy as np
    from renormalizer.sbm import ColeDavidsonSDF

    sdf = ColeDavidsonSDF(
        float(ita.value),
        float(omega_c.value),
        float(beta.value),
        float(upper_limit.value),
    )
    omega_k, c_j2 = sdf.Wang1(int(n_modes.value))
    renormalization_constant = float(sdf.reno(float(omega_k[-1])))
    delta_eff = float(float(raw_delta.value) * renormalization_constant)
    node = orm.ArrayData()
    node.set_array("omega_k", np.asarray(omega_k, dtype=float))
    node.set_array("c_j2", np.asarray(c_j2, dtype=float))
    node.base.attributes.set("ita", float(ita.value))
    node.base.attributes.set("omega_c", float(omega_c.value))
    node.base.attributes.set("beta", float(beta.value))
    node.base.attributes.set("upper_limit", float(upper_limit.value))
    node.base.attributes.set("raw_delta", float(raw_delta.value))
    node.base.attributes.set("n_modes", int(n_modes.value))
    node.base.attributes.set("renormalization_constant", renormalization_constant)
    node.base.attributes.set("delta_eff", delta_eff)
    return node


@calcfunction
def define_hamiltonian_terms(hamiltonian_terms: orm.List):
    """Normalize the full user-authored Hamiltonian term list for the TTN case."""
    from aiida_renormalizer.data import OpSpecData

    return OpSpecData.from_list(_normalize_op_specs(hamiltonian_terms.get_list()))


def _first_half_spin_dof(basis_items: list[dict[str, Any]]) -> Any:
    for item in basis_items:
        if str(item.get("kind")) == "half_spin":
            return item["dof"]
    raise ValueError("basis must contain a half_spin item for expectation operators")


@calcfunction
def define_basis(basis: orm.List):
    """Normalize user-authored basis items for the TTN case."""
    from aiida_renormalizer.data import BasisSpecData

    return BasisSpecData.from_list(basis.get_list())


@calcfunction
def build_ttn_model(
    hamiltonian_terms,
    basis,
    tree_type: orm.Str,
    m_max: orm.Int,
) -> orm.Str:
    """Render the model-construction part of the TTN script."""
    basis_items = basis.as_list()
    observable_spin_dof = _first_half_spin_dof(basis_items)
    return orm.Str(
        _render_stage_script(
            "example_ttn_sbm_zt_build_ttn_model.py.jinja",
            {
                "tree_type_literal": json.dumps(tree_type.value),
                "m_max_literal": repr(int(m_max.value)),
                "observable_spin_dof_literal": _render_python_dof_atom_expr(observable_spin_dof),
                "hamiltonian_terms_block": _render_hamiltonian_terms_block(hamiltonian_terms.as_list()),
                "basis_block": _render_basis_spec_block(basis_items),
            },
        )
    )


@calcfunction
def build_dynamcis_calculation(
    dt: orm.Float,
    nsteps: orm.Int,
    method: orm.Str,
) -> orm.Str:
    """Render the calculation part of the TTN script."""
    return orm.Str(
        _render_stage_script(
            "example_ttn_sbm_zt_build_calculation.py.jinja",
            {
                "dt_literal": repr(float(dt.value)),
                "nsteps_literal": repr(int(nsteps.value)),
                "method_literal": json.dumps(method.value),
            },
        )
    )


# Optional alias for callers using corrected spelling.
build_calculation = build_dynamcis_calculation

def _render_ttn_script_payload(
    *,
    environment: orm.ArrayData,
    model_section: str,
    calculation_section: str,
    real_run: bool,
) -> dict[str, str]:
    env_dict = {
        "ita": float(environment.base.attributes.get("ita")),
        "omega_c": float(environment.base.attributes.get("omega_c")),
        "beta": float(environment.base.attributes.get("beta")),
        "upper_limit": float(environment.base.attributes.get("upper_limit")),
        "raw_delta": float(environment.base.attributes.get("raw_delta")),
        "n_modes": int(environment.base.attributes.get("n_modes")),
        "renormalization_constant": float(environment.base.attributes.get("renormalization_constant")),
        "delta_eff": float(environment.base.attributes.get("delta_eff")),
        "omega_k": environment.get_array("omega_k").tolist(),
        "c_j2": environment.get_array("c_j2").tolist(),
    }
    script_text = _render_stage_script(
        "example_ttn_sbm_zt_case_single_file.py.jinja",
        {
            "environment_literal": pformat(env_dict, sort_dicts=False),
            "real_run_literal": "True" if bool(real_run) else "False",
            "model_section": model_section,
            "calculation_section": calculation_section,
        },
    )
    return {
        "script_name": "symbolic_ttn_dynamics_generated.py",
        "script_text": script_text,
    }


@calcfunction
def _materialize_ttn_script_payload(
    environment: orm.ArrayData,
    model_section: orm.Str,
    calculation_section: orm.Str,
    real_run: orm.Bool,
) -> orm.Dict:
    return orm.Dict(
        dict=_render_ttn_script_payload(
            environment=environment,
            model_section=model_section.value,
            calculation_section=calculation_section.value,
            real_run=bool(real_run.value),
        )
    )


@workfunction
def build_bundle_manifest(
    environment: orm.ArrayData,
    model_section: orm.Str,
    calculation_section: orm.Str,
    real_run: orm.Bool,
    work_dir: orm.Str,
):
    """Render the TTN script and materialize its execution bundle."""
    script_payload = _materialize_ttn_script_payload(
        environment=environment,
        model_section=model_section,
        calculation_section=calculation_section,
        real_run=real_run,
    )
    payload = script_payload.get_dict()
    manifest = bundle_manifest_for_python_script(
        script_name=orm.Str(payload["script_name"]),
        script_text=orm.Str(payload["script_text"]),
        work_dir=work_dir,
    )
    return {
        "script_payload": script_payload,
        "manifest": manifest,
    }
