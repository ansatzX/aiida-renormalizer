"""TTN SBM ZT case-scoped calcfunctions."""

from __future__ import annotations

import base64
import json
import math
from pathlib import Path, PurePosixPath
from pprint import pformat
from typing import Any

from aiida import orm
from aiida.engine import calcfunction, workfunction
from jinja2 import Environment, FileSystemLoader

_TEMPLATE_ENV = Environment(
    loader=FileSystemLoader(str(Path(__file__).resolve().parents[1] / "templates")),
)
_BUNDLE_RELATIVE_PATH_CHARS = frozenset(
    "abcdefghijklmnopqrstuvwxyz"
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "0123456789"
    "._-/"
)
_EVOLVE_METHOD_ALIASES = {
    "tdvp-ps": "tdvp_ps",
    "tdvp_ps": "tdvp_ps",
    "tdvp-ps2": "tdvp_ps2",
    "tdvp_ps2": "tdvp_ps2",
    "tdvp-vmf": "tdvp_vmf",
    "tdvp_vmf": "tdvp_vmf",
    "tdvp-mu-cmf": "tdvp_mu_cmf",
    "tdvp_mu_cmf": "tdvp_mu_cmf",
    "tdvp-mu-vmf": "tdvp_mu_vmf",
    "tdvp_mu_vmf": "tdvp_mu_vmf",
    "prop-and-compress": "prop_and_compress",
    "prop_and_compress": "prop_and_compress",
}


def _render_stage_script(template_name: str, context: dict[str, Any]) -> str:
    template = _TEMPLATE_ENV.get_template(template_name)
    return template.render(**context).rstrip() + "\n"


def _validate_bundle_relative_path(path: Any, *, label: str) -> str:
    message = f"{label} must be a bundle-relative path without parent traversal"
    if not isinstance(path, str) or not path:
        raise ValueError(message)
    if path.strip() != path or not path.strip():
        raise ValueError(message)
    if any(char not in _BUNDLE_RELATIVE_PATH_CHARS for char in path):
        raise ValueError(message)
    if "//" in path:
        raise ValueError(message)

    candidate = path
    posix_path = PurePosixPath(candidate)
    if posix_path.is_absolute():
        raise ValueError(message)
    parts = candidate.split("/")
    if any(part in ("", ".", "..") for part in parts):
        raise ValueError(message)
    return PurePosixPath(*parts).as_posix()


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
    output_directory: str = "generated_scripts",
    *,
    include_execute_stage: bool = False,
) -> dict:
    if not script_text.strip():
        raise ValueError("script_text must be non-empty")

    safe_script_name = _validate_bundle_relative_path(script_name, label="script_name")
    safe_output_directory = _validate_bundle_relative_path(
        output_directory.rstrip("/"),
        label="output_directory",
    )
    script_text_b64 = base64.b64encode(script_text.encode("utf-8")).decode("ascii")
    output_path = _validate_bundle_relative_path(
        f"{safe_output_directory}/{safe_script_name}",
        label="output_path",
    )
    stages = [
        {
            "name": "write_generated_script",
            "script": _render_stage_script(
                "ttn_sbm_zt_write_generated_script_stage.py.jinja",
                {"output_path_literal": json.dumps(output_path), "script_text_b64": script_text_b64},
            ),
        },
        {
            "name": "compile_generated_script",
            "script": _render_stage_script(
                "ttn_sbm_zt_compile_generated_script_stage.py.jinja",
                {"output_path_literal": json.dumps(output_path)},
            ),
        },
    ]
    if include_execute_stage:
        stages.append(
            {
                "name": "execute_generated_script",
                "script": _render_stage_script(
                    "ttn_sbm_zt_execute_generated_script_stage.py.jinja",
                    {"output_path_literal": json.dumps(output_path)},
                ),
            }
        )
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
def build_ttn_model(
    hamiltonian_terms: orm.List,
    basis: orm.List,
    tree_type: orm.Str,
    m_max: orm.Int,
) -> orm.Str:
    """Render the model-construction part of the TTN script."""
    term_items = _normalize_op_specs(hamiltonian_terms.get_list())
    from aiida_renormalizer.data import BasisSpecData

    basis_items = BasisSpecData.from_list(basis.get_list()).as_list()
    return orm.Str(
        _render_stage_script(
            "example_ttn_sbm_zt_build_ttn_model.py.jinja",
            {
                "tree_type_literal": json.dumps(tree_type.value),
                "m_max_literal": repr(int(m_max.value)),
                "hamiltonian_terms_block": _render_hamiltonian_terms_block(term_items),
                "basis_block": _render_basis_spec_block(basis_items),
            },
        )
    )


def _validate_literal_payload(value: Any, path: str) -> None:
    if value is None or isinstance(value, (str, bool)):
        return
    if isinstance(value, int):
        return
    if isinstance(value, float):
        if math.isfinite(value):
            return
        raise ValueError(f"{path} must not contain non-finite float values")
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_literal_payload(item, f"{path}[{index}]")
        return
    if isinstance(value, tuple):
        for index, item in enumerate(value):
            _validate_literal_payload(item, f"{path}[{index}]")
        return
    if isinstance(value, dict):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"{path} dict keys must be strings")
            _validate_literal_payload(item, f"{path}.{key}")
        return
    raise ValueError(f"{path} must contain only str, int, finite float, bool, None, list, tuple, or dict")


class _DoubleQuotedLiteralString(str):
    def __repr__(self) -> str:
        return json.dumps(str(self))


def _prefer_double_quoted_strings(value: Any) -> Any:
    if isinstance(value, str):
        return _DoubleQuotedLiteralString(value)
    if isinstance(value, list):
        return [_prefer_double_quoted_strings(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_prefer_double_quoted_strings(item) for item in value)
    if isinstance(value, dict):
        return {
            _DoubleQuotedLiteralString(key): _prefer_double_quoted_strings(item)
            for key, item in value.items()
        }
    return value


def _render_python_literal(value: Any) -> str:
    _validate_literal_payload(value, "literal payload")
    return pformat(_prefer_double_quoted_strings(value), sort_dicts=False)


def _is_supported_observation_dof_atom(obj: Any) -> bool:
    return (
        (isinstance(obj, str) and bool(obj.strip()))
        or (isinstance(obj, int) and not isinstance(obj, bool))
        or (
            isinstance(obj, tuple)
            and all(_is_supported_observation_dof_atom(item) for item in obj)
        )
    )


def _is_supported_observation_dofs(obj: Any) -> bool:
    return _is_supported_observation_dof_atom(obj) or (
        isinstance(obj, list) and bool(obj) and all(_is_supported_observation_dof_atom(item) for item in obj)
    )


def _normalize_evolve_method(raw_method: Any) -> tuple[str, str]:
    if not isinstance(raw_method, str) or not raw_method.strip():
        raise ValueError("time evolution method must be non-empty")
    method_token = raw_method.strip()
    method_name = _EVOLVE_METHOD_ALIASES.get(method_token)
    if method_name is None:
        raise ValueError(f"unsupported time evolution method: {method_token}")
    return method_token, method_name


def _normalize_time_evolution(
    *,
    dt: float,
    nsteps: int,
    method: str,
    observations: list[Any],
) -> dict[str, Any]:
    dt_value = float(dt)
    if not math.isfinite(dt_value) or dt_value <= 0:
        raise ValueError("time evolution dt must be a finite positive number")

    if isinstance(nsteps, bool) or not isinstance(nsteps, int) or nsteps <= 0:
        raise ValueError("time evolution nsteps must be a positive integer")

    method_token, method_name = _normalize_evolve_method(method)

    if not isinstance(observations, list) or not observations:
        raise ValueError("time evolution observations must be a non-empty list")

    normalized_observations: list[dict[str, Any]] = []
    seen_labels: set[str] = set()
    for index, raw_observation in enumerate(observations):
        if not isinstance(raw_observation, dict):
            raise ValueError(f"time evolution observations[{index}] must be a dict")
        if not {"label", "symbol", "dofs"}.issubset(raw_observation):
            raise ValueError(f"time evolution observations[{index}] must contain label, symbol, and dofs")

        raw_label = raw_observation["label"]
        raw_symbol = raw_observation["symbol"]
        label = raw_label.strip() if isinstance(raw_label, str) else ""
        symbol = raw_symbol.strip() if isinstance(raw_symbol, str) else ""
        if not label:
            raise ValueError(f"time evolution observations[{index}].label must be non-empty")
        if not symbol:
            raise ValueError(f"time evolution observations[{index}].symbol must be non-empty")
        if label in seen_labels:
            raise ValueError(f"duplicate observation label: {label}")
        if not _is_supported_observation_dofs(raw_observation["dofs"]):
            raise ValueError(
                f"time evolution observations[{index}].dofs must be a supported dof atom or list of dof atoms"
            )

        seen_labels.add(label)
        normalized_item = dict(raw_observation)
        normalized_item["label"] = label
        normalized_item["symbol"] = symbol
        normalized_item["dofs"] = raw_observation["dofs"]
        normalized_item["qn"] = raw_observation.get("qn", 0)
        _validate_literal_payload(normalized_item, f"time evolution observations[{index}]")
        normalized_observations.append(normalized_item)

    return {
        "dt": dt_value,
        "nsteps": int(nsteps),
        "method_token": method_token,
        "method_name": method_name,
        "observations": normalized_observations,
    }


def _render_time_evolution_section(time_evolution: dict[str, Any]) -> str:
    return _render_stage_script(
        "example_ttn_sbm_zt_build_calculation.py.jinja",
        {
            "dt_literal": repr(float(time_evolution["dt"])),
            "nsteps_literal": repr(int(time_evolution["nsteps"])),
            "method_literal": json.dumps(time_evolution["method_token"]),
            "evolve_method_name": time_evolution["method_name"],
            "observation_specs_literal": _render_python_literal(time_evolution["observations"]),
        },
    )


@calcfunction
def build_time_evolution_section(
    dt: orm.Float,
    nsteps: orm.Int,
    method: orm.Str,
    observations: orm.List,
) -> orm.Str:
    """Render the time-evolution and observation part of the TTN script."""
    time_evolution = _normalize_time_evolution(
        dt=float(dt.value),
        nsteps=int(nsteps.value),
        method=method.value,
        observations=observations.get_list(),
    )
    return orm.Str(_render_time_evolution_section(time_evolution))


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


def _ttn_sbm_zt_artifact_records(*, work_dir: str, script_name: str, real_run: bool) -> list[dict[str, Any]]:
    base = _validate_bundle_relative_path(work_dir.rstrip("/"), label="work_dir")
    safe_script_name = _validate_bundle_relative_path(script_name, label="script_name")
    records = [
        {
            "role": "generated_script",
            "path": _validate_bundle_relative_path(
                f"{base}/{safe_script_name}",
                label="generated_script artifact path",
            ),
            "required": True,
        }
    ]
    if real_run:
        records.append(
            {
                "role": "result",
                "path": _validate_bundle_relative_path(
                    f"{base}/sbm_zt_result.json",
                    label="result artifact path",
                ),
                "required": True,
            }
        )
    else:
        records.append(
            {
                "role": "preview",
                "path": _validate_bundle_relative_path(
                    f"{base}/sbm_zt_preview.json",
                    label="preview artifact path",
                ),
                "required": True,
            }
        )
    return records


@calcfunction
def _attach_artifact_records(manifest: orm.Dict, artifacts: orm.List) -> orm.Dict:
    payload = manifest.get_dict()
    payload["artifacts"] = artifacts.get_list()
    return orm.Dict(dict=payload)


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
    manifest_payload = render_python_script_bundle_manifest_payload(
        script_name=payload["script_name"],
        script_text=payload["script_text"],
        output_directory=work_dir.value,
        include_execute_stage=True,
    )
    manifest = _materialize_bundle_manifest(orm.List(list=manifest_payload["stages"]))
    manifest = _attach_artifact_records(
        manifest=manifest,
        artifacts=orm.List(
            list=_ttn_sbm_zt_artifact_records(
                work_dir=work_dir.value,
                script_name=payload["script_name"],
                real_run=bool(real_run.value),
            )
        ),
    )
    return {
        "script_payload": script_payload,
        "manifest": manifest,
    }
