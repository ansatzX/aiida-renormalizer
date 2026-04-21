#!/usr/bin/env python
"""Hubbard case via direct CalcJob chain."""

from __future__ import annotations

from aiida import load_profile, orm
from aiida.engine import run_get_node

from aiida_renormalizer.calculations.basic.build_mpo import BuildMPOCalcJob
from aiida_renormalizer.calculations.composite.dmrg import DMRGCalcJob
from aiida_renormalizer.data.model import ModelData

CODE = "reno-script-clean@localhost"
BUILD = {"nsites": 10, "hopping_t": -1.0, "coulomb_u": 4.0}


def _hubbard_symbolic_spec(*, nsites: int, hopping_t: float, coulomb_u: float):
    basis = [
        {"kind": "half_spin", "dof": f"orb_{site}_{spin}", "sigmaqn": [0, 0]}
        for site in range(nsites)
        for spin in ("up", "dn")
    ]
    terms = []
    for site in range(nsites - 1):
        up_i = f"orb_{site}_up"
        up_j = f"orb_{site + 1}_up"
        dn_i = f"orb_{site}_dn"
        dn_j = f"orb_{site + 1}_dn"
        terms.extend(
            [
                {"symbol": "+ -", "dofs": [up_i, up_j], "factor": hopping_t},
                {"symbol": "- +", "dofs": [up_i, up_j], "factor": hopping_t},
                {"symbol": "+ -", "dofs": [dn_i, dn_j], "factor": hopping_t},
                {"symbol": "- +", "dofs": [dn_i, dn_j], "factor": hopping_t},
            ]
        )
    for site in range(nsites):
        terms.append(
            {
                "symbol": "+ - + -",
                "dofs": [f"orb_{site}_up", f"orb_{site}_up", f"orb_{site}_dn", f"orb_{site}_dn"],
                "factor": coulomb_u,
            }
        )
    return basis, terms


def main() -> None:
    load_profile()
    if CODE.startswith("@"):
        raise RuntimeError("Please set CODE = 'your_code_label@your_computer_label'")

    basis, hamiltonian = _hubbard_symbolic_spec(**BUILD)
    model = ModelData.from_symbolic_spec(basis=basis, hamiltonian=hamiltonian)
    code = orm.load_code(CODE)

    mpo_out, mpo_node = run_get_node(BuildMPOCalcJob, code=code, model=model)
    gs_out, gs_node = run_get_node(DMRGCalcJob, code=code, model=model, mpo=mpo_out["output_mpo"])

    print(f"[BuildMPOCalcJob] pk={mpo_node.pk}")
    print(f"[DMRGCalcJob] pk={gs_node.pk}")
    print(f"ground_energy={gs_out['output_parameters'].get_dict().get('e')}")


if __name__ == "__main__":
    main()
