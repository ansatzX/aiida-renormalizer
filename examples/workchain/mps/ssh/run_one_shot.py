#!/usr/bin/env python
"""Optical SSH case migrated from ori_examples/ssh.py."""
from __future__ import annotations

from aiida import load_profile, orm
from aiida.engine import run_get_node

from aiida_renormalizer.data import ModelData, TensorNetworkLayoutData
from aiida_renormalizer.workchains.ground_state import GroundStateWorkChain
from aiida_renormalizer.workchains.model_to_mpo import ModelToMPOWorkChain

CODE = "reno-script-clean@localhost"
BUILD = {
    "nsites": 2,
    "hopping_t": -1.0,
    "eph_g": 0.7,
    "phonon_omega": 0.5,
    "nbas": 4,
    "periodic": True,
}
RUN = {"strategy": "dmrg"}


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


def _run(wc, **inputs):
    outputs, node = run_get_node(wc, **inputs)
    print(f"[{wc.__name__}] pk={node.pk}")
    return outputs


def main() -> None:
    load_profile()
    if CODE.startswith("@"):
        raise RuntimeError("Please set CODE = 'your_code_label@your_computer_label'")

    basis, hamiltonian = _optical_ssh_symbolic_spec(**BUILD)
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
