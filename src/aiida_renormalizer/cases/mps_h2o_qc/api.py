"""Human-facing native Reno inputs for separately recorded model, state and DMRG."""

from __future__ import annotations

import math

from renormalizer.model import Op
from renormalizer.model.basis import BasisHalfSpin

from aiida_renormalizer.data.op import serialize_op
from aiida_renormalizer.data.utils import decode_dofs
from aiida_renormalizer.utils import run_process

from .optimization import snapshot_optimization
from .recording import build_model_section, build_optimization_section, build_state_section


def record_mps_model(*, basis, hamiltonian, integrals):
    """Record the supplied orbital order, basis QNs and native operators.

    The integral node records origin, not a hidden replacement Hamiltonian.
    Additional terms or an intentionally selected subset of integrals are allowed.
    """
    basis_items = []
    for item in basis:
        if type(item) is not BasisHalfSpin or not isinstance(item.dofs[0], int):
            raise TypeError("this QC case supports integer-labelled BasisHalfSpin orbitals")
        basis_items.append({"dof": item.dofs[0], "sigmaqn": item.sigmaqn.tolist()})
    terms = [hamiltonian] if isinstance(hamiltonian, Op) else list(hamiltonian)
    serialized = []
    for term in terms:
        if not isinstance(term, Op):
            raise TypeError("hamiltonian must contain native Reno Op objects")
        spec = serialize_op(term)
        spec["dofs"] = decode_dofs(spec["dofs"])
        if not all(math.isfinite(part) for part in spec["factor"].values()):
            raise ValueError("Hamiltonian coefficients must be finite")
        serialized.append(spec)
    section, _ = run_process(build_model_section, basis=basis_items,
                             hamiltonian=serialized, integrals=integrals)
    return section


def record_random_state(*, quantum_number, bond_dimension, sector_mixing):
    """Record Mps.random; sector_mixing is Reno's percent for selecting symmetry blocks."""
    if (not isinstance(quantum_number, (list, tuple)) or not quantum_number
            or any(type(q) is not int or q < 0 for q in quantum_number)):
        raise ValueError("quantum_number must be a nonempty sequence of nonnegative integers")
    if type(bond_dimension) is not int or bond_dimension < 1:
        raise ValueError("bond_dimension must be a positive integer")
    if not math.isfinite(sector_mixing) or not 0 <= sector_mixing <= 1:
        raise ValueError("sector_mixing must be finite and in [0, 1]")
    section, _ = run_process(build_state_section, quantum_number=list(quantum_number),
                             bond_dimension=bond_dimension, sector_mixing=sector_mixing)
    return section


def record_optimization(*, optimize_config, preserve_initial_state, nuclear_repulsion,
                        reference_energy, check_reference, reference_rtol, reference_atol):
    """Record native DMRG plus explicit electronic-to-total-energy reporting choices."""
    if type(preserve_initial_state) is not bool or type(check_reference) is not bool:
        raise TypeError("state preservation and reference-check choices must be booleans")
    if not all(math.isfinite(x) for x in
               (nuclear_repulsion, reference_energy, reference_rtol, reference_atol)):
        raise ValueError("energy shift, reference and tolerances must be finite")
    if reference_rtol < 0 or reference_atol < 0:
        raise ValueError("reference tolerances must be nonnegative")
    section, _ = run_process(
        build_optimization_section,
        settings=snapshot_optimization(optimize_config),
        preserve_initial_state=preserve_initial_state,
        nuclear_repulsion=float(nuclear_repulsion), reference_energy=float(reference_energy),
        check_reference=check_reference,
        reference_rtol=float(reference_rtol), reference_atol=float(reference_atol),
    )
    return section
