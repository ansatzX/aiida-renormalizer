"""Case-owned native-object recording and standalone source assembly."""

from __future__ import annotations

import math
from enum import Enum
from importlib.resources import files

from aiida import orm
from aiida.engine import calcfunction
from renormalizer.model import Model, Op, Phonon
from renormalizer.model.basis import BasisSHO, BasisSimpleElectron
from renormalizer.utils import CompressConfig, EvolveConfig

from .artifacts import validate_standalone_script


def literal(value):
    """Readable Python literals preserving native enum, complex and array values."""
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, complex):
        return f"complex({value.real!r}, {value.imag!r})"
    if hasattr(value, "tolist"):
        return f"np.array({literal(value.tolist())})"
    if isinstance(value, float) and not math.isfinite(value):
        return f"float({str(value)!r})"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{literal(k)}: {literal(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        body = ", ".join(literal(x) for x in value)
        return "[" + body + "]" if isinstance(value, list) else "(" + body + ",)" if value else "()"
    if value is None or isinstance(value, (str, bool, int, float, complex)):
        return repr(value)
    raise TypeError(f"unsupported native value: {type(value).__name__}")


def operator_source(op):
    if not isinstance(op, Op):
        raise TypeError("expected a native Reno Op")
    return (
        f"Op(symbol={op.symbol!r}, dof={literal(op.dofs)}, factor={literal(op.factor)}, "
        f"qn={literal([q.tolist() for q in op.qn_list])})"
    )


@calcfunction
def render_model(specification: orm.Dict, source_file: orm.SinglefileData = None) -> orm.Str:
    """Record the frozen native basis/Hamiltonian, optionally linked to its input file."""
    spec = specification.get_dict()
    lines = ["def build_model():", "    basis = ["]
    lines += ["        " + item + "," for item in spec["basis"]]
    lines += ["    ]", "    hamiltonian = ["]
    lines += ["        " + item + "," for item in spec["hamiltonian"]]
    lines += ["    ]", "    return Model(basis=basis, ham_terms=hamiltonian)"]
    return orm.Str("\n".join(lines) + "\n")


def record_model(model, *, source_file=None):
    """Accept the explicitly constructed native Model; reject lossy basis conversion."""
    if type(model) is not Model:
        raise TypeError("pass a native Model with explicit basis and Hamiltonian terms")
    if model.output_ordering != model.basis:
        raise ValueError("custom observable output ordering is not supported")
    if model.dipole:
        raise ValueError("dipole operators are outside this case's model recorder")
    basis = []
    for item in model.basis:
        if type(item) is BasisSimpleElectron:
            expected = [[0], [1]]
            expression = f"BasisSimpleElectron(dof={literal(item.dof)})"
        elif type(item) is BasisSHO:
            if item.x0 or item.dvr or item.general_xp_power or getattr(item, "scale_omega", False):
                raise ValueError("only undisplaced, non-DVR SHO primitive bases are supported")
            expected = [[0]] * item.nbas
            expression = (
                f"BasisSHO(dof={literal(item.dof)}, omega={item.omega!r}, nbas={item.nbas})"
            )
        else:
            raise TypeError(f"unsupported basis: {type(item).__name__}")
        if item.sigmaqn.tolist() != expected:
            raise ValueError("custom primitive quantum numbers are not supported")
        basis.append(expression)
    kwargs = {
        "specification": orm.Dict(
            dict={"basis": basis, "hamiltonian": [operator_source(op) for op in model.ham_terms]}
        )
    }
    if source_file is not None:
        kwargs["source_file"] = source_file
    return render_model(**kwargs)


def configuration_source(config, name):
    """Keep every resolved native setting, including nested RK/Taylor coefficients."""
    if type(config) not in (EvolveConfig, CompressConfig):
        raise TypeError("expected EvolveConfig or CompressConfig")
    lines = [f"{name} = {type(config).__name__}()"]
    for key, value in vars(config).items():
        if key in {"rk_config", "taylor_config"}:
            for nested, item in vars(value).items():
                lines.append(f"{name}.{key}.{nested} = {literal(item)}")
        else:
            lines.append(f"{name}.{key} = {literal(value)}")
    return lines


@calcfunction
def render_initial_state(settings: orm.Dict) -> orm.Str:
    """Record explicit state preparation choices, separately from time evolution."""
    options = settings.get_dict()
    lines = ["def prepare_initial_state(model):"]
    lines.extend("    " + line for line in options["statements"])
    return orm.Str("\n".join(lines) + "\n")


@calcfunction
def render_evolution(settings: orm.Dict) -> orm.Str:
    """Record resolved native controls and explicit segment parameters."""
    options = settings.get_dict()
    lines = ["def evolve_prepared_state(prepared):"]
    lines.extend("    " + line for line in options["statements"])
    return orm.Str("\n".join(lines) + "\n")


@calcfunction
def assemble_script(
    model: orm.Str, initial_state: orm.Str, evolution: orm.Str, run_metadata: orm.Dict
) -> orm.Str:
    """Embed all implementation and frozen input; the remote process needs only Reno."""
    runtime = files(__package__).joinpath("runtime.py").read_text()
    source = runtime + "\n\n" + model.value + "\n" + initial_state.value + "\n" + evolution.value
    source += "\n\nMETADATA = " + repr(run_metadata.get_dict()) + "\n"
    source += """
\ndef main(output_dir=None):
    model = build_model()
    prepared = prepare_initial_state(model)
    final_state, result = evolve_prepared_state(prepared)
    result["metadata"] = METADATA
    directory = Path(__file__).resolve().parent if output_dir is None else Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    (directory / RESULT_NAME).write_text(json.dumps(plain(result), indent=2))
    return final_state, result


if __name__ == "__main__":
    main()
"""
    validate_standalone_script(source)
    return orm.Str(source)


def record_initial_state(
    *,
    temperature,
    thermal_steps,
    electron_site,
    relaxed_phonons,
    evolve_config,
    compress_config,
    expand_bonds,
    expansion_coefficient,
    include_ex,
    subtract_initial_energy,
    exact_ground_thermal,
):
    """Freeze explicit native ground/thermal preparation and electron excitation."""
    relaxed_phonons = list(relaxed_phonons)
    for _, phonon in relaxed_phonons:
        if not isinstance(phonon, Phonon):
            raise TypeError("relaxed_phonons requires native Phonon objects")
        if phonon.omega[0] != phonon.omega[1] or phonon.dis[0] != 0:
            raise ValueError("relaxed excitation supports equal-curvature, zero-origin phonons")
    phonons = (
        "["
        + ", ".join(
            f"({literal(dof)}, Phonon.simple_phonon(omega=Quantity({ph.omega[0]!r}), "
            f"displacement=Quantity({ph.dis[1]!r}), n_phys_dim={ph.n_phys_dim}))"
            for dof, ph in relaxed_phonons
        )
        + "]"
    )
    lines = configuration_source(evolve_config, "evolve_config")
    lines += configuration_source(compress_config, "compress_config")
    lines += [
        f"state = make_ground_state(model, temperature=Quantity({temperature.as_au()!r}), "
        f"thermal_steps={thermal_steps!r}, exact_ground_thermal={exact_ground_thermal!r})",
        f"state = excite_electron(state, site={literal(electron_site)}, relaxed_phonons={phonons})",
        "return prepare_dynamics(state, evolve_config=evolve_config, "
        "compress_config=compress_config, "
        f"expand_bonds={expand_bonds!r}, expansion_coefficient={expansion_coefficient!r}, "
        f"include_ex={include_ex!r}, subtract_initial_energy={subtract_initial_energy!r})",
    ]
    return render_initial_state(orm.Dict(dict={"statements": lines}))


def record_evolution(
    *,
    evolve_config,
    compress_config,
    dt,
    nsteps,
    observe_initial,
    rdm,
    momentum,
    normalize,
    edge_site,
    edge_threshold,
):
    if momentum and not rdm:
        raise ValueError("momentum occupations require rdm=True")
    lines = configuration_source(evolve_config, "evolve_config")
    lines += configuration_source(compress_config, "compress_config")
    lines += [
        "state, hamiltonian, initial_energy = prepared",
        "state, rows = evolve_segment(state, hamiltonian, evolve_config=evolve_config, "
        "compress_config=compress_config, "
        f"dt={literal(dt)}, nsteps={nsteps!r}, observe_initial={observe_initial!r}, "
        f"rdm={rdm!r}, momentum={momentum!r}, normalize={normalize!r}, "
        f"edge_site={edge_site!r}, edge_threshold={edge_threshold!r})",
        "return state, summarize_dynamics(state, rows, initial_energy=initial_energy, "
        f"planned_steps={nsteps!r})",
    ]
    return render_evolution(orm.Dict(dict={"statements": lines}))
