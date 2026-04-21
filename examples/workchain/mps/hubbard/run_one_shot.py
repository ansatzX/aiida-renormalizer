#!/usr/bin/env python
"""Hubbard case migrated from ori_examples/hubbard.py."""
from __future__ import annotations

from aiida import load_profile, orm
from aiida.engine import run_get_node

from aiida_renormalizer.data import ModelData, TensorNetworkLayoutData
from aiida_renormalizer.workchains.ground_state import GroundStateWorkChain
from aiida_renormalizer.workchains.model_to_mpo import ModelToMPOWorkChain

CODE = "reno-script-clean@localhost"
BUILD = {"nsites": 10, "hopping_t": -1.0, "coulomb_u": 4.0}
RUN = {"strategy": "dmrg"}


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


def _run(wc, **inputs):
    outputs, node = run_get_node(wc, **inputs)
    print(f"[{wc.__name__}] pk={node.pk}")
    return outputs


def main() -> None:
    load_profile()
    if CODE.startswith("@"):
        raise RuntimeError("Please set CODE = 'your_code_label@your_computer_label'")

    basis, hamiltonian = _hubbard_symbolic_spec(**BUILD)
    model = ModelData.from_symbolic_spec(basis=basis, hamiltonian=hamiltonian)
    tn_layout = TensorNetworkLayoutData.from_chain(model.base.attributes.get("dof_list"))
    code = orm.load_code(CODE)
    mpo_out = _run(ModelToMPOWorkChain, model=model, code=code, tn_layout=tn_layout)
    mpo = mpo_out["mpo"]
    tn_layout = mpo_out.get("output_tn_layout", tn_layout)
    out = _run(
        GroundStateWorkChain,
        model=model,
        mpo=mpo,
        tn_layout=tn_layout,
        strategy=orm.Str(RUN["strategy"]),
        code=code,
    )
    print(f"ground_energy={out['energy'].value}")


if __name__ == "__main__":
    main()
