"""Human-facing adapters: Reno objects in, recorded executable sections out."""

from __future__ import annotations

from renormalizer.model import Op
from renormalizer.model.basis import BasisHalfSpin, BasisSHO

from aiida_renormalizer.data.op import serialize_op
from aiida_renormalizer.data.utils import decode_dofs
from aiida_renormalizer.utils import run_process

from .evolution import snapshot_evolve_config
from .recording import build_time_evolution_section, build_ttn_model
from .validation import normalize_initial_state


def _operator_terms(operators):
    operators = [operators] if isinstance(operators, Op) else list(operators)
    result = []
    for operator in operators:
        if not isinstance(operator, Op):
            raise TypeError("expected Renormalizer Op or a list/OpSum of operators")
        term = serialize_op(operator)
        term["dofs"] = decode_dofs(term["dofs"])
        result.append(term)
    return result


def record_binary_tree_model(
    *,
    hamiltonian,
    basis,
    bath_dofs,
    root_dofs,
    contract_primitive,
    contract_labels,
    dummy_label,
    initial_state,
    compression_criteria,
    max_bond_dimension,
    expand_bonds,
    expansion_coefficient,
):
    """Serialize Reno objects and record the explicitly chosen binary-tree construction.

    This case uses string dofs and supports HalfSpin and undisplaced, unscaled,
    non-DVR SHO bases, local occupations or real amplitude vectors, fixed compression, and optional
    Hamiltonian-guided expansion.
    Unsupported choices fail here instead of being discarded by the renderer.
    """
    basis_specs = []
    for item in basis:
        if isinstance(item, BasisHalfSpin):
            basis_specs.append(["half_spin", item.dofs[0], item.sigmaqn.tolist()])
        elif isinstance(item, BasisSHO):
            if (
                item.x0 != 0
                or item.dvr
                or item.general_xp_power
                or getattr(item, "scale_omega", False)
            ):
                raise ValueError(
                    "this case renderer supports undisplaced, unscaled, non-DVR SHO bases only"
                )
            basis_specs.append(["sho", item.dofs[0], float(item.omega), item.nbas])
        else:
            raise TypeError(f"unsupported basis type: {type(item).__name__}")
    section, _ = run_process(
        build_ttn_model,
        hamiltonian_terms=_operator_terms(hamiltonian),
        basis=basis_specs,
        topology={
            "kind": "binary_mctdh",
            "bath_dofs": bath_dofs,
            "root_dofs": root_dofs,
            "contract_primitive": contract_primitive,
            "contract_labels": contract_labels,
            "dummy_label": dummy_label,
        },
        initial_state=normalize_initial_state(initial_state),
        compression={"criteria": compression_criteria, "max_bonddim": max_bond_dimension},
        expansion={
            "enabled": expand_bonds,
            "hint": "hamiltonian",
            "coefficient": expansion_coefficient,
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
    """Record one evolution segment and its explicit clock; do not execute it."""
    section, _ = run_process(
        build_time_evolution_section,
        dt=dt,
        nsteps=nsteps,
        evolve_settings=snapshot_evolve_config(evolve_config),
        observations=[
            {"label": label, "terms": _operator_terms(terms)}
            for label, terms in observations.items()
        ],
        observe_initial=observe_initial,
        observe_every=observe_every,
        start_time=start_time,
        start_step=start_step,
    )
    return section
