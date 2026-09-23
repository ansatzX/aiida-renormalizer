"""Case-owned deterministic recording and standalone source generation."""

from __future__ import annotations

import ast
import math
from importlib.resources import files
from pprint import pformat

from aiida import orm
from aiida.engine import calcfunction
from jinja2 import Environment, StrictUndefined

from aiida_renormalizer.data.utils import decode_dof_atom, decode_dofs

from .artifacts import validate_standalone_script
from .evolution import EXTRA_FIELDS, validate_evolve_settings


def _render(name, **context):
    text = files(__package__).joinpath("templates", name).read_text()
    return (
        Environment(undefined=StrictUndefined).from_string(text).render(**context).rstrip() + "\n"
    )


def _op_call(item):
    factor = complex(item["factor"]["real"], item["factor"]["imag"])
    if not math.isfinite(factor.real) or not math.isfinite(factor.imag):
        raise ValueError("operator coefficients must be finite")
    if factor.imag == 0:
        factor = factor.real
    qn = item["qn"]
    qn_source = f"{qn[:1]!r} * {len(qn)}" if qn and all(q == qn[0] for q in qn) else repr(qn)
    return (
        f"Op({item['symbol']!r}, {decode_dofs(item['dofs'])!r}, factor={factor!r}, qn={qn_source})"
    )


def _basis_call(item):
    dof = decode_dof_atom(item["dof"])
    if item["kind"] == "half_spin":
        return f"ba.BasisHalfSpin({dof!r}, sigmaqn={item['sigmaqn']!r})"
    if item["kind"] == "sho":
        return f"ba.BasisSHO({dof!r}, omega={item['omega']!r}, nbas={item['nbas']!r})"
    if item["kind"] == "dummy":
        return f"ba.BasisDummy({dof!r}, nbas={item['nbas']!r}, sigmaqn={item['sigmaqn']!r})"
    raise ValueError("unknown recorded basis")


@calcfunction
def record_model_snapshot(snapshot: orm.Dict) -> orm.Str:
    data = snapshot.get_dict()
    tree_lines = []
    for index, node in enumerate(data["tree"]):
        calls = ", ".join(_basis_call(item) for item in node["basis"])
        tree_lines.append(f"    node_{index} = TreeNodeBasis([{calls}])")
    for index, node in enumerate(data["tree"]):
        if node["children"]:
            children = ", ".join(f"node_{child}" for child in node["children"])
            tree_lines.append(f"    node_{index}.add_child([{children}])")
    tree_lines.append(f"    return BasisTree(node_{data['root']})")
    terms = ",\n        ".join(_op_call(item) for item in data["hamiltonian"])
    source = (
        f"INITIAL_STATE = {pformat(data['initial_state'], sort_dicts=False)}\n"
        f"MAX_BOND_DIMENSION = {data['max_bond_dimension']!r}\n"
        f"EXPAND_BONDS = {data['expand_bonds']!r}\n"
        f"EXPANSION_COEFFICIENT = {data['expansion_coefficient']!r}\n\n"
        "def build_tree_topology():\n" + "\n".join(tree_lines) + "\n\n"
        "def build_hamiltonian_terms():\n    return [\n        " + terms + "\n    ]\n\n"
        "def build_network_operators_and_state():\n"
        "    tree = build_tree_topology()\n"
        "    hamiltonian = TTNO(tree, build_hamiltonian_terms())\n"
        "    state = TTNS(tree, condition=INITIAL_STATE)\n"
        "    state.compress_config = CompressConfig(\n"
        "        CompressCriteria.fixed, max_bonddim=MAX_BOND_DIMENSION)\n"
        "    if EXPAND_BONDS:\n"
        "        state = expand_bond_dimension_general(\n"
        "            state, hamiltonian, coef=EXPANSION_COEFFICIENT, ex_mps=None)\n"
        "    return state, hamiltonian, tree\n"
    )
    return orm.Str(source)


@calcfunction
def record_evolution_snapshot(
    dt: orm.Float,
    nsteps: orm.Int,
    evolve_settings: orm.Dict,
    observations: orm.List,
    observe_initial: orm.Bool,
    observe_every: orm.Int,
    start_time: orm.Float,
    start_step: orm.Int,
    record_bond_dimensions: orm.Bool,
) -> orm.Str:
    if not math.isfinite(dt.value) or dt.value <= 0 or nsteps.value < 0:
        raise ValueError("dt must be finite and positive and nsteps non-negative")
    if observe_every.value < 1 or not math.isfinite(start_time.value) or start_step.value < 0:
        raise ValueError("invalid observation cadence or segment offset")
    settings = evolve_settings.get_dict()
    kwargs = validate_evolve_settings(settings)
    normalized, labels = [], set()
    for item in observations.get_list():
        label = item["label"]
        if (
            not isinstance(label, str)
            or not label.strip()
            or label in labels
            or label in {"step", "time", "bond_dims"}
        ):
            raise ValueError("observation labels must be unique, nonempty, and non-reserved")
        labels.add(label)
        normalized.append(
            {"label": repr(label), "terms": [_op_call(term) for term in item["terms"]]}
        )
    return orm.Str(
        _render(
            "evolution.py.jinja",
            evolve_validation_source=files(__package__).joinpath("evolution.py").read_text(),
            dt_literal=repr(dt.value),
            nsteps_literal=repr(nsteps.value),
            evolve_method_name=settings["method"],
            evolve_config_kwargs={name: repr(value) for name, value in kwargs.items()},
            evolve_config_attributes={name: repr(settings[name]) for name in EXTRA_FIELDS},
            observations=normalized,
            observe_initial_literal=repr(observe_initial.value),
            observe_every_literal=repr(observe_every.value),
            start_time_literal=repr(start_time.value),
            start_step_literal=repr(start_step.value),
            record_bond_dimensions_literal=repr(record_bond_dimensions.value),
        )
    )


@calcfunction
def record_analysis_source(source: orm.Str, arguments: orm.Dict) -> orm.Dict:
    tree = ast.parse(source.value)
    if (
        len(tree.body) != 1
        or not isinstance(tree.body[0], ast.FunctionDef)
        or tree.body[0].decorator_list
    ):
        raise ValueError("analysis must be one undecorated ordinary Python function")
    name = tree.body[0].name
    compile(tree, "analysis", "exec")
    return orm.Dict(
        dict={"source": source.value, "call": f"{name}(rows, **{arguments.get_dict()!r})"}
    )


@calcfunction
def assemble_script(
    environment: orm.ArrayData,
    renormalization: orm.Float,
    model_section: orm.Str,
    calculation_section: orm.Str,
    analysis_section: orm.Dict,
) -> orm.Str:
    analysis = analysis_section.get_dict()
    metadata = dict(environment.base.attributes.all)
    metadata["renormalization_constant"] = renormalization.value
    source = _render(
        "runtime.py.jinja",
        environment_literal=pformat(metadata, sort_dicts=False),
        model_section=model_section.value,
        calculation_section=calculation_section.value,
        analysis_source=analysis.get("source", ""),
        analysis_call=analysis.get("call", "rows"),
    )
    validate_standalone_script(source)
    return orm.Str(source)
