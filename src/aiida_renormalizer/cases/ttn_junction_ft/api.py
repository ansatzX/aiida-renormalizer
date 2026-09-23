"""Native-object recording adapters owned by this TTN case."""

from __future__ import annotations

import builtins
import inspect
import math
import symtable
import textwrap

from renormalizer.model import Op
from renormalizer.model.basis import BasisDummy, BasisHalfSpin, BasisSHO
from renormalizer.tn import BasisTree
from renormalizer.utils import CompressCriteria

from aiida_renormalizer.data.op import serialize_op
from aiida_renormalizer.data.utils import encode_dof_atom
from aiida_renormalizer.utils import run_process

from .evolution import snapshot_evolve_config
from .recording import record_analysis_source, record_evolution_snapshot, record_model_snapshot
from .validation import normalize_initial_state


def operator_terms(operators):
    operators = [operators] if isinstance(operators, Op) else list(operators)
    if not operators or any(not isinstance(op, Op) for op in operators):
        raise TypeError("supply a native Op or nonempty OpSum/list of Op objects")
    return [serialize_op(op) for op in operators]


def record_model(
    *,
    hamiltonian,
    basis_tree,
    initial_state,
    compression_criteria,
    max_bond_dimension,
    expand_bonds,
    expansion_coefficient,
):
    """Record the actual supplied tree; no topology or physical policy is invented."""
    if not isinstance(basis_tree, BasisTree):
        raise TypeError("basis_tree must be a native Reno BasisTree")
    if compression_criteria is not CompressCriteria.fixed:
        raise ValueError("this case currently supports fixed compression only")
    if type(max_bond_dimension) is not int or max_bond_dimension < 1:
        raise ValueError("max_bond_dimension must be a positive integer")
    if (
        type(expand_bonds) is not bool
        or not math.isfinite(expansion_coefficient)
        or expansion_coefficient <= 0
    ):
        raise ValueError("supply an expansion bool and a positive finite coefficient")
    state = normalize_initial_state(initial_state)
    nodes = list(basis_tree.node_list)
    indices = {id(node): index for index, node in enumerate(nodes)}
    physical_basis = {}
    records = []
    for node in nodes:
        items = []
        for basis in node.basis_sets:
            entry = {"dof": encode_dof_atom(basis.dofs[0]), "sigmaqn": basis.sigmaqn.tolist()}
            if type(basis) is BasisHalfSpin:
                entry["kind"] = "half_spin"
            elif type(basis) is BasisSHO:
                if (
                    basis.x0 != 0
                    or basis.dvr
                    or basis.general_xp_power
                    or getattr(basis, "scale_omega", False)
                ):
                    raise ValueError("only undisplaced, unscaled, non-DVR SHO bases are supported")
                if basis.sigmaqn.any():
                    raise ValueError("SHO bases must use their native zero quantum numbers")
                entry.update(kind="sho", omega=float(basis.omega), nbas=basis.nbas)
                if not math.isfinite(entry["omega"]) or entry["omega"] <= 0:
                    raise ValueError("SHO frequencies must be finite and positive")
            elif type(basis) is BasisDummy:
                entry.update(kind="dummy", nbas=basis.nbas)
            else:
                raise TypeError(f"unsupported native basis: {type(basis).__name__}")
            if type(basis) is not BasisDummy:
                dof = basis.dofs[0]
                if not isinstance(dof, str) or dof in physical_basis:
                    raise ValueError("physical dofs must be unique strings")
                physical_basis[dof] = basis
            items.append(entry)
        records.append(
            {"basis": items, "children": [indices[id(child)] for child in node.children]}
        )
    if set(state) != set(physical_basis):
        raise ValueError("initial_state must explicitly specify every physical dof")
    for dof, value in state.items():
        basis = physical_basis[dof]
        if isinstance(value, int):
            if not 0 <= value < basis.nbas:
                raise ValueError(f"initial occupation outside basis: {dof}")
        else:
            if len(value) != basis.nbas:
                raise ValueError(f"initial amplitude dimension mismatch: {dof}")
            occupied_qn = [
                basis.sigmaqn[i].tolist() for i, amplitude in enumerate(value) if amplitude != 0
            ]
            if any(qn != occupied_qn[0] for qn in occupied_qn[1:]):
                raise ValueError(f"initial amplitudes mix QN sectors: {dof}")
    terms = [hamiltonian] if isinstance(hamiltonian, Op) else list(hamiltonian)
    if any(not set(op.dofs) <= set(physical_basis) for op in terms):
        raise ValueError("Hamiltonian uses dofs absent from the supplied physical tree")
    section, _ = run_process(
        record_model_snapshot,
        snapshot={
            "tree": records,
            "root": indices[id(basis_tree.root)],
            "hamiltonian": operator_terms(terms),
            "initial_state": state,
            "max_bond_dimension": max_bond_dimension,
            "expand_bonds": expand_bonds,
            "expansion_coefficient": float(expansion_coefficient),
        },
    )
    return section


def record_evolution(
    *,
    dt,
    nsteps,
    evolve_config,
    observations,
    observe_initial,
    observe_every,
    start_time=0.0,
    start_step=0,
):
    """Record an explicit real-time segment; the emitted function returns its final TTNS."""
    section, _ = run_process(
        record_evolution_snapshot,
        dt=dt,
        nsteps=nsteps,
        evolve_settings=snapshot_evolve_config(evolve_config),
        observations=[
            {"label": label, "terms": operator_terms(terms)}
            for label, terms in observations.items()
        ],
        observe_initial=observe_initial,
        observe_every=observe_every,
        start_time=start_time,
        start_step=start_step,
    )
    return section


def record_analysis(function, **arguments):
    """Embed an explicitly supplied self-contained row analysis and its named arguments."""
    captured = inspect.getclosurevars(function)
    if captured.nonlocals or captured.globals:
        raise ValueError(
            "analysis must be self-contained: use explicit arguments or imports inside it"
        )
    inspect.signature(function).bind([], **arguments)
    source = textwrap.dedent(inspect.getsource(function))
    tables = [symtable.symtable(source, "analysis", "exec")]
    unresolved = set()
    while tables:
        table = tables.pop()
        unresolved.update(
            symbol.get_name()
            for symbol in table.get_symbols()
            if symbol.is_global()
            and symbol.is_referenced()
            and symbol.get_name() not in vars(builtins)
        )
        tables.extend(table.get_children())
    if unresolved:
        raise ValueError(f"analysis requires external names: {sorted(unresolved)}")
    result, _ = run_process(record_analysis_source, source=source, arguments=arguments)
    return result
