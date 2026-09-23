"""Case-owned deterministic AiiDA recording and source rendering boundaries."""

from __future__ import annotations

from importlib.resources import files
from pprint import pformat

from aiida import orm
from aiida.engine import calcfunction
from jinja2 import Environment, StrictUndefined

from .artifacts import validate_standalone_script
from .optimization import restore_optimization

_TEMPLATES = Environment(undefined=StrictUndefined)


def _render(name, **context):
    source = files(__package__).joinpath("templates", name).read_text()
    return _TEMPLATES.from_string(source).render(**context).rstrip() + "\n"


@calcfunction
def build_model_section(basis: orm.List, hamiltonian: orm.List,
                        integrals: orm.ArrayData) -> orm.Str:
    """The input integral node connects the rendered model to FCIDUMP provenance."""
    items = basis.get_list()
    dofs = [item["dof"] for item in items]
    if not items or len(set(dofs)) != len(dofs):
        raise ValueError("basis must have unique, nonempty orbital labels")
    terms = hamiltonian.get_list()
    if not terms:
        raise ValueError("Hamiltonian must have at least one term")
    calls = []
    for term in terms:
        if not set(term["dofs"]) <= set(dofs):
            raise ValueError("Hamiltonian contains orbitals absent from the basis")
        factor = complex(term["factor"]["real"], term["factor"]["imag"])
        factor = factor.real if factor.imag == 0 else factor
        calls.append(f"Op({term['symbol']!r}, {term['dofs']!r}, "
                     f"factor={factor!r}, qn={term['qn']!r})")
    basis_calls = [f"BasisHalfSpin({item['dof']!r}, sigmaqn={item['sigmaqn']!r})"
                   for item in items]
    metadata = {key: integrals.base.attributes.get(key)
                for key in ("fcidump_filename", "fcidump_sha256", "spatial_norbs", "spin_norbs")}
    return orm.Str(_render("model.py.jinja", basis_calls=basis_calls, terms=calls,
                           input_metadata=pformat(metadata)))


@calcfunction
def build_state_section(quantum_number: orm.List, bond_dimension: orm.Int,
                        sector_mixing: orm.Float) -> orm.Str:
    return orm.Str(_render("state.py.jinja", quantum_number=repr(quantum_number.get_list()),
                           bond_dimension=bond_dimension.value, sector_mixing=sector_mixing.value))


@calcfunction
def build_optimization_section(settings: orm.Dict, preserve_initial_state: orm.Bool,
                               nuclear_repulsion: orm.Float, reference_energy: orm.Float,
                               check_reference: orm.Bool, reference_rtol: orm.Float,
                               reference_atol: orm.Float) -> orm.Str:
    restore_optimization(settings.get_dict())
    validation = files(__package__).joinpath("optimization.py").read_text()
    validation = validation.replace("from __future__ import annotations\n", "")
    return orm.Str(_render(
        "optimization.py.jinja", validation_source=validation,
        settings=pformat(settings.get_dict(), sort_dicts=False),
        preserve_initial_state=preserve_initial_state.value,
        nuclear_repulsion=repr(nuclear_repulsion.value),
        reference_energy=repr(reference_energy.value), check_reference=check_reference.value,
        reference_rtol=repr(reference_rtol.value), reference_atol=repr(reference_atol.value),
    ))


@calcfunction
def assemble_script(model_section: orm.Str, state_section: orm.Str,
                    optimization_section: orm.Str) -> orm.Str:
    source = _render("runtime.py.jinja", model=model_section.value, state=state_section.value,
                     optimization=optimization_section.value)
    validate_standalone_script(source)
    return orm.Str(source)
