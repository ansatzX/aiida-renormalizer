"""Deterministic AiiDA rendering boundaries for this case, with no solver execution."""

from __future__ import annotations

import math
from importlib.resources import files
from pprint import pformat

from aiida import orm
from aiida.engine import calcfunction
from jinja2 import Environment, StrictUndefined

from aiida_renormalizer.data import BasisSpecData

from .artifacts import validate_standalone_script
from .evolution import EXTRA_FIELDS, validate_evolve_settings
from .representation import (
    normalize_operator_terms,
    render_basis,
    render_operator_terms,
)
from .validation import validate_construction

_TEMPLATES = Environment(undefined=StrictUndefined)


def _render(name, **context):
    source = files(__package__).joinpath("templates", name).read_text()
    return _TEMPLATES.from_string(source).render(**context).rstrip() + "\n"


@calcfunction
def build_ttn_model(
    hamiltonian_terms: orm.List,
    basis: orm.List,
    topology: orm.Dict,
    initial_state: orm.Dict,
    compression: orm.Dict,
    expansion: orm.Dict,
) -> orm.Str:
    """Validate and emit the supplied operators, basis, and construction choices."""
    terms = normalize_operator_terms(hamiltonian_terms.get_list())
    basis_items = BasisSpecData.from_list(basis.get_list()).as_list()
    choices = validate_construction(
        basis_items,
        topology.get_dict(),
        initial_state.get_dict(),
        compression.get_dict(),
        expansion.get_dict(),
    )
    dofs = {item["dof"] for item in basis_items}
    for term in terms:
        term_dofs = term["dofs"] if isinstance(term["dofs"], list) else [term["dofs"]]
        if not set(term_dofs) <= dofs:
            raise ValueError("Hamiltonian refers to a dof absent from the basis")
    return orm.Str(
        _render(
            "example_ttn_sbm_zt_build_ttn_model.py.jinja",
            hamiltonian_terms_block=render_operator_terms(terms),
            basis_block=render_basis(basis_items),
            **{
                f"{key}_literal": pformat(value, sort_dicts=False) for key, value in choices.items()
            },
        )
    )


@calcfunction
def build_time_evolution_section(
    dt: orm.Float,
    nsteps: orm.Int,
    evolve_settings: orm.Dict,
    observations: orm.List,
    observe_initial: orm.Bool,
    observe_every: orm.Int,
    start_time: orm.Float,
    start_step: orm.Int,
) -> orm.Str:
    """Record supported TTN evolution and named operator sums at explicit times."""
    if not math.isfinite(dt.value) or dt.value <= 0 or nsteps.value < 0:
        raise ValueError("dt must be finite and positive; nsteps must be non-negative")
    if observe_every.value < 1:
        raise ValueError("observe_every must be positive")
    if not math.isfinite(start_time.value) or start_step.value < 0:
        raise ValueError("start_time must be finite; start_step must be non-negative")
    settings = evolve_settings.get_dict()
    config_kwargs = validate_evolve_settings(settings)
    items, seen = observations.get_list(), set()
    normalized = []
    for item in items:
        if set(item) != {"label", "terms"}:
            raise ValueError("observations require a label and explicit operator terms")
        label = item["label"]
        if (
            not isinstance(label, str)
            or not label.strip()
            or label in seen
            or label in {"step", "time"}
        ):
            raise ValueError(f"invalid, duplicate, or reserved observation label: {label}")
        seen.add(label)
        normalized.append({"label": label, "terms": normalize_operator_terms(item["terms"])})
    return orm.Str(
        _render(
            "example_ttn_sbm_zt_build_calculation.py.jinja",
            evolve_validation_source=files(__package__).joinpath("evolution.py").read_text(),
            dt_literal=repr(dt.value),
            nsteps_literal=repr(nsteps.value),
            evolve_method_name=settings["method"],
            evolve_config_kwargs={name: repr(value) for name, value in config_kwargs.items()},
            evolve_config_attributes={name: repr(settings[name]) for name in EXTRA_FIELDS},
            observation_specs_literal=pformat(normalized, sort_dicts=False),
            observe_initial_literal=repr(observe_initial.value),
            observe_every_literal=repr(observe_every.value),
            start_time_literal=repr(start_time.value),
            start_step_literal=repr(start_step.value),
        )
    )


@calcfunction
def assemble_script(
    environment: orm.ArrayData,
    renormalization: orm.Float,
    model_section: orm.Str,
    calculation_section: orm.Str,
    restart_contract: orm.Dict,
) -> orm.Str:
    """Record a complete runnable Reno script; dry-run means it is not executed here."""
    restart = restart_contract.get_dict()
    if restart.get("mode") != "replay-only" or restart.get("checkpoint_restart") is not False:
        raise ValueError("this case currently supports replay-only restart")
    metadata = dict(environment.base.attributes.all)
    metadata.pop("array|omega_k", None)
    metadata.pop("array|c_j2", None)
    metadata["renormalization_constant"] = renormalization.value
    metadata["renormalization_cutoffs"] = {
        bound: renormalization.base.attributes.get(bound, None)
        for bound in ("lower_cutoff", "upper_cutoff")
    }
    source = _render(
        "example_ttn_sbm_zt_case_single_file.py.jinja",
        environment_literal=pformat(metadata, sort_dicts=False),
        restart_contract_literal=pformat(restart, sort_dicts=False),
        model_section=model_section.value,
        calculation_section=calculation_section.value,
    )
    validate_standalone_script(source)
    return orm.Str(source)
