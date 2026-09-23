"""Native Reno inputs recorded as explicit executable calculation sections.

This implementation belongs to this case. It deliberately does not share new
rendering/runtime code with the other reconstructed examples.
"""

from __future__ import annotations

import math
from enum import Enum
from importlib.resources import files

from aiida import orm
from aiida.engine import calcfunction
from renormalizer.model import Op
from renormalizer.model import basis as ba
from renormalizer.utils import CompressConfig, EvolveConfig, OptimizeConfig

from .artifacts import validate_standalone_script


def literal(value):
    """Python representation preserving tuple dofs, complex values and Reno enums."""
    if isinstance(value, Enum):
        return f"configs.{type(value).__name__}.{value.name}"
    if hasattr(value, "item") and getattr(value, "ndim", 0) == 0:
        value = value.item()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value is None or isinstance(value, (str, bool, int, complex)):
        return repr(value)
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("NaN cannot be recorded")
        return repr(value) if math.isfinite(value) else f"float({str(value)!r})"
    if isinstance(value, dict):
        return "{" + ", ".join(f"{literal(k)}: {literal(v)}" for k, v in value.items()) + "}"
    if isinstance(value, (list, tuple)):
        values = ", ".join(literal(v) for v in value)
        return (
            "[" + values + "]"
            if isinstance(value, list)
            else "(" + values + (",)" if value else ")")
        )
    if isinstance(value, CompressConfig):
        raise TypeError("use integer bond dimensions in this case's native DMRG procedure")
    raise TypeError(f"unsupported recorded value: {type(value).__name__}")


def operator_source(operators):
    terms = [operators] if isinstance(operators, Op) else list(operators)
    if not terms or any(not isinstance(op, Op) for op in terms):
        raise TypeError("supply a native Op or a nonempty OpSum/list of Op")
    return (
        "[\n        "
        + ",\n        ".join(
            (
                f"Op({op.symbol!r}, {literal(op.dofs)}, factor={literal(op.factor)}, "
                f"qn={literal(op.qn_list)})"
            )
            for op in terms
        )
        + "\n    ]"
    )


def basis_source(item):
    dof = literal(item.dofs[0])
    if isinstance(item, ba.BasisHalfSpin):
        return f"ba.BasisHalfSpin({dof}, sigmaqn={literal(item.sigmaqn.tolist())})"
    if isinstance(item, ba.BasisSimpleElectron):
        return f"ba.BasisSimpleElectron({dof})"
    if isinstance(item, ba.BasisSHO):
        if getattr(item, "scale_omega", False):
            raise ValueError("scaled SHO is not supported by this case")
        return (
            f"ba.BasisSHO({dof}, omega={literal(item.omega)}, nbas={item.nbas}, "
            f"x0={literal(item.x0)}, dvr={item.dvr!r}, general_xp_power={item.general_xp_power!r})"
        )
    raise TypeError(f"unsupported basis: {type(item).__name__}")


def configuration_source(config, name):
    """Preserve every native config field; unsupported payloads fail, never disappear."""
    if not isinstance(config, (CompressConfig, EvolveConfig, OptimizeConfig)):
        raise TypeError("expected native CompressConfig, EvolveConfig or OptimizeConfig")
    if isinstance(config, OptimizeConfig) and config.nroots != 1:
        raise ValueError("this case's state-returning DMRG API requires nroots=1")
    constructor = type(config).__name__
    if isinstance(config, EvolveConfig):
        lines = [
            f"{name} = EvolveConfig(rk_solver={config.rk_config.method!r}, "
            f"taylor_order={config.taylor_config.order!r})"
        ]
    else:
        lines = [f"{name} = {constructor}()"]
    for field, value in vars(config).items():
        if field in ("rk_config", "taylor_config"):
            continue
        # Native site-specific arrays are runtime state, not silently coerced lists.
        if hasattr(value, "ndim") and value.ndim != 0:
            raise ValueError(f"record a fresh config without runtime array field {field}")
        lines.append(f"{name}.{field} = {literal(value)}")
    return "\n".join(lines)


@calcfunction
def _record_section(source: orm.Str) -> orm.Str:
    """Record an explicit deterministic construction or operation, without executing it."""
    compile(source.value, "recorded_section", "exec")
    return orm.Str(source.value)


def record_model(*, hamiltonian, basis):
    basis = list(basis)
    if not basis:
        raise ValueError("basis cannot be empty")
    source = "def build_model():\n    basis = [\n        "
    source += ",\n        ".join(basis_source(item) for item in basis) + "\n    ]\n"
    source += f"    hamiltonian = {operator_source(hamiltonian)}\n"
    source += "    return Model(basis, hamiltonian)\n\nmodel = build_model()\nmpo = Mpo(model)\n"
    return _record_section(orm.Str(source))


def record_product_state(*, local_states):
    return _record_section(
        orm.Str(f"state = Mps.hartree_product_state(model, condition={literal(local_states)})\n")
    )


def record_observations(*, operators, one_site_rdms=(), bond_entropy=False, electronic_rdm=False):
    source = "observations = {\n"
    for label, ops in operators.items():
        source += f"    {label!r}: Mpo(model, {operator_source(ops)}),\n"
    source += "}\n"
    source += f"rdm_sites = {literal(tuple(one_site_rdms))}\n"
    source += (
        f"measure_bond_entropy = {bond_entropy!r}\nmeasure_electronic_rdm = {electronic_rdm!r}\n"
    )
    return _record_section(orm.Str(source))


def record_evolution(
    *,
    compress_config,
    evolve_config,
    dt,
    nsteps,
    normalize,
    expand_bonds,
    expansion_coefficient,
    expansion_include_ex,
    observe_initial,
    observe_every,
    start_time=0.0,
    start_step=0,
):
    if (
        not isinstance(nsteps, int)
        or nsteps < 0
        or not isinstance(observe_every, int)
        or observe_every < 1
    ):
        raise ValueError("nsteps must be nonnegative; observe_every must be positive")
    if not isinstance(dt, (float, int)) or not math.isfinite(dt) or dt == 0:
        raise ValueError("real nonzero finite dt required")
    evolve_config.check_valid_dt(dt)
    source = configuration_source(compress_config, "compress_config") + "\n"
    source += configuration_source(evolve_config, "evolve_config") + "\n"
    source += "state.compress_config = compress_config\n"
    if expand_bonds:
        source += (
            f"state = state.expand_bond_dimension(mpo, coef={expansion_coefficient!r}, "
            f"include_ex={expansion_include_ex!r})\n"
        )
    source += (
        f"state, rows = evolve_segment(state, mpo, evolve_config=evolve_config, dt={dt!r}, "
        f"nsteps={nsteps!r}, normalize={normalize!r}, observations=observations, "
        "one_site_rdms=rdm_sites, bond_entropy=measure_bond_entropy, "
        "electronic_rdm=measure_electronic_rdm, "
        f"observe_initial={observe_initial!r}, observe_every={observe_every!r}, "
        f"start_time={start_time!r}, start_step={start_step!r})\n"
        "result['trajectory'] = rows\n"
    )
    return _record_section(orm.Str(source))


@calcfunction
def _assemble(**sections) -> orm.Str:
    runtime = files(__package__).joinpath("runtime.py").read_text()
    body = "\n".join(sections[key].value for key in sorted(sections))
    body += "\nresult['final_energy'] = real_value(state.expectation(mpo))\n"
    body += (
        "result['observations'] = observe_state(state, observations, "
        "one_site_rdms=rdm_sites, bond_entropy=measure_bond_entropy, "
        "electronic_rdm=measure_electronic_rdm)\n"
    )
    body += "result['final_quantum_number'] = state.qntot.tolist()\n"
    body += "result['final_bond_dims'] = [int(n) for n in state.bond_dims]\n"
    body += "Path(output_file).write_text(json.dumps(result, indent=2))\nreturn state, result\n"
    body = "    result = {}\n" + "\n".join(
        "    " + line if line else "" for line in body.splitlines()
    )
    source = (
        runtime
        + "\n\ndef main(output_file='sbm_result.json'):\n"
        + body
        + "\n\nif __name__ == '__main__':\n    main(Path(__file__).with_name('sbm_result.json'))\n"
    )
    validate_standalone_script(source)
    return orm.Str(source)


def assemble_script(*sections):
    """Keep the caller's phase order and AiiDA links to every recorded section."""
    if not sections:
        raise ValueError("supply model, state, observations and calculation sections")
    return _assemble(**{f"section_{index:04d}": section for index, section in enumerate(sections)})
