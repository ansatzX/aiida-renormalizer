#!/usr/bin/env python
"""Optical SSH case via direct CalcJob chain."""

from __future__ import annotations

from aiida import load_profile, orm
from aiida.engine import run_get_node

from aiida_renormalizer.calculations.basic.build_mpo import BuildMPOCalcJob
from aiida_renormalizer.calculations.composite.dmrg import DMRGCalcJob
from aiida_renormalizer.data.model import ModelData

CODE = "reno-script-clean@localhost"
BUILD = {
    "nsites": 2,
    "hopping_t": -1.0,
    "eph_g": 0.7,
    "phonon_omega": 0.5,
    "nbas": 4,
    "periodic": True,
}


def _optical_ssh_symbolic_spec(
    *,
    nsites: int,
    hopping_t: float,
    eph_g: float,
    phonon_omega: float,
    nbas: int,
    periodic: bool,
):
    basis = [
        item
        for site in range(nsites)
        for item in (
            {"kind": "half_spin", "dof": f"e_{site}", "sigmaqn": [0, 0]},
            {"kind": "sho", "dof": f"ph_{site}", "omega": phonon_omega, "nbas": nbas},
        )
    ]
    terms = []
    for i in range(nsites):
        terms.append({"symbol": r"b^\dagger b", "dofs": f"ph_{i}", "factor": phonon_omega})
    links = [(i, i + 1) for i in range(nsites - 1)]
    if periodic:
        links.append((nsites - 1, 0))
    for i, j in links:
        ei, ej = f"e_{i}", f"e_{j}"
        pi, pj = f"ph_{i}", f"ph_{j}"
        terms.extend(
            [
                {"symbol": r"a^\dagger a", "dofs": [ei, ej], "factor": hopping_t},
                {"symbol": r"a^\dagger a", "dofs": [ej, ei], "factor": hopping_t},
                {"symbol": r"a^\dagger a b^\dagger+b", "dofs": [ei, ej, pj], "factor": eph_g},
                {"symbol": r"a^\dagger a b^\dagger+b", "dofs": [ei, ej, pi], "factor": -eph_g},
                {"symbol": r"a^\dagger a b^\dagger+b", "dofs": [ej, ei, pj], "factor": eph_g},
                {"symbol": r"a^\dagger a b^\dagger+b", "dofs": [ej, ei, pi], "factor": -eph_g},
            ]
        )
    return basis, terms


def main() -> None:
    load_profile()
    if CODE.startswith("@"):
        raise RuntimeError("Please set CODE = 'your_code_label@your_computer_label'")

    basis, hamiltonian = _optical_ssh_symbolic_spec(**BUILD)
    model = ModelData.from_symbolic_spec(basis=basis, hamiltonian=hamiltonian)
    code = orm.load_code(CODE)

    mpo_out, mpo_node = run_get_node(BuildMPOCalcJob, code=code, model=model)
    gs_out, gs_node = run_get_node(DMRGCalcJob, code=code, model=model, mpo=mpo_out["output_mpo"])

    print(f"[BuildMPOCalcJob] pk={mpo_node.pk}")
    print(f"[DMRGCalcJob] pk={gs_node.pk}")
    print(f"ground_energy={gs_out['output_parameters'].get_dict().get('e')}")


if __name__ == "__main__":
    main()
