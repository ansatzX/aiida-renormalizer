"""Case-owned native-object recording and standalone source assembly."""

from __future__ import annotations

import math
from enum import Enum
from importlib.resources import files

from aiida import orm
from aiida.engine import calcfunction
from renormalizer.model import Model, Op
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
    thermal_sector,
    imaginary_config,
    evolve_config,
    compress_config,
    current_terms,
    subtract_initial_energy,
):
    """Freeze the native one-electron MPDM imaginary-time and real-time preparations."""
    if thermal_sector != "one_electron":
        raise ValueError("this case supports only thermal_sector=one_electron")
    lines = configuration_source(imaginary_config, "imaginary_config")
    lines += configuration_source(evolve_config, "evolve_config")
    lines += configuration_source(compress_config, "compress_config")
    currents = (
        "["
        + ", ".join(
            "[" + ", ".join(operator_source(op) for op in group) + "]" for group in current_terms
        )
        + "]"
    )
    lines += [
        f"state = make_thermal_state(model, temperature=Quantity({temperature.as_au()!r}), "
        f"thermal_steps={thermal_steps!r}, imaginary_config=imaginary_config, "
        f"compress_config=compress_config, thermal_sector={thermal_sector!r})",
        f"return prepare_correlation(state, current_terms={currents}, "
        "evolve_config=evolve_config, compress_config=compress_config, "
        f"subtract_initial_energy={subtract_initial_energy!r})",
    ]
    return render_initial_state(orm.Dict(dict={"statements": lines}))


def record_evolution(
    *,
    evolve_config,
    compress_config,
    temperature,
    dt,
    nsteps,
    observe_initial,
    correlation_prefactor,
    stop_at_tail,
    tail_window,
    tail_relative_tolerance,
    smoke,
):
    lines = configuration_source(evolve_config, "evolve_config")
    lines += configuration_source(compress_config, "compress_config")
    lines += [
        "state, hamiltonian, currents, initial_energy = prepared",
        "state, rows = evolve_segment(state, hamiltonian, currents, evolve_config=evolve_config, "
        "compress_config=compress_config, "
        f"dt={literal(dt)}, nsteps={nsteps!r}, observe_initial={observe_initial!r}, "
        f"correlation_prefactor={literal(correlation_prefactor)}, "
        f"stop_at_tail={stop_at_tail!r}, tail_window={tail_window!r}, "
        f"tail_relative_tolerance={tail_relative_tolerance!r})",
        f"converged = {stop_at_tail!r} and bool(rows) and rows[-1]['step'] < {nsteps!r} "
        f"and tail_converged(rows, window={tail_window!r}, "
        f"relative_tolerance={tail_relative_tolerance!r})",
        f"interpretation = 'smoke_formula_only' if {smoke!r} else "
        "('correlation_tail_converged' if converged else 'finite_window_formula_not_converged')",
        'result = {"initial_energy": initial_energy, "rows": rows, '
        '"interpretation": interpretation, "final_time": rows[-1]["time"] if rows else 0.0}',
        f"result.update(mobility_from_rows(rows, temperature=Quantity({temperature.as_au()!r})))",
        "result.update(time_series=[row['time'] for row in rows], "
        "auto_corr=[row['auto_correlation'] for row in rows], "
        "auto_corr_decomposition=[row['current_components'] for row in rows])",
        "result['mobility'] = {'au': result['mobility_au'], "
        "'cm2_per_vs': result['mobility_cm2_per_Vs'], 'interpretation': interpretation, "
        f"'converged': converged and not {smoke!r}}}",
        f"result['execution_runtime'] = {{'planned_steps': {nsteps!r}, "
        "'completed_steps': rows[-1]['step'] if rows else 0, "
        "'termination_reason': 'correlation_tail_converged' if converged "
        "else 'requested_steps_completed'}",
        "result['restart_plan'] = {'mode': 'replay-only', 'checkpoint_restart': False}",
        "return state, result",
    ]
    return render_evolution(orm.Dict(dict={"statements": lines}))
