#!/usr/bin/env python
"""TTN SBM zero-temperature case migrated from ori_examples/ttns/sbm_zt.py."""

from __future__ import annotations

import numpy as np
from aiida import load_profile, orm
from aiida.engine import run_get_node

from aiida_renormalizer.data import BasisTreeData, TensorNetworkLayoutData
from aiida_renormalizer.workchains.ohmic_renorm_modes import OhmicRenormModesWorkChain
from aiida_renormalizer.workchains.ttn_symbolic_dynamics import TTNSymbolicDynamicsWorkChain

CODE = "reno-script-clean@localhost"
INPUT = {
    "alpha": 1.0,
    "s_exponent": 0.5,
    "omega_c": 1.0,
    "raw_delta": 1.0,
    "renormalization_p": 1.0,
    "n_modes": 128,
}
BUILD = {"epsilon": 0.0, "tree_type": "binary", "m_max": 20}
RUN = {"dt": 0.2, "nsteps": 200, "method": "tdvp_ps"}


def _compose_sbm_zt_payload(
    omega_k,
    c_j2,
    *,
    delta_eff: float,
    epsilon: float,
    m_max: int,
    tree_type: str = "binary",
):
    omega = np.asarray(omega_k, dtype=float)
    c2 = np.asarray(c_j2, dtype=float)
    c = np.sqrt(c2)
    nbas = np.maximum(16 * c2 / np.maximum(omega, 1e-12) ** 3, 4.0)
    nbas = np.round(nbas).astype(int)

    basis = [{"kind": "half_spin", "dof": "spin", "sigmaqn": [0, 0]}]
    basis.extend(
        {"kind": "sho", "dof": f"v_{i}", "omega": float(w), "nbas": int(n)}
        for i, (w, n) in enumerate(zip(omega, nbas))
    )
    hamiltonian = [
        {"symbol": "sigma_z", "dofs": "spin", "factor": float(epsilon)},
        {"symbol": "sigma_x", "dofs": "spin", "factor": float(delta_eff)},
    ]
    for i, (w, ci) in enumerate(zip(omega, c)):
        dof = f"v_{i}"
        hamiltonian.extend(
            [
                {"symbol": "p^2", "dofs": dof, "factor": 0.5},
                {"symbol": "x^2", "dofs": dof, "factor": 0.5 * float(w) ** 2},
                {"symbol": "sigma_z x", "dofs": ["spin", dof], "factor": float(ci)},
            ]
        )
    return {"basis": basis, "hamiltonian": hamiltonian, "tree_type": tree_type, "m_max": int(m_max)}


def _run(wc, **inputs):
    outputs, node = run_get_node(wc, **inputs)
    print(f"[{wc.__name__}] pk={node.pk}")
    return outputs


def _tree_layout_from_symbolic(symbolic: dict):
    from renormalizer.model import basis as ba
    from renormalizer.tn import BasisTree

    basis = []
    for item in symbolic["basis"]:
        if item["kind"] == "half_spin":
            basis.append(ba.BasisHalfSpin(item["dof"], item.get("sigmaqn", [0, 0])))
        else:
            basis.append(ba.BasisSHO(item["dof"], omega=float(item["omega"]), nbas=int(item["nbas"])))
    tree = BasisTree.binary(basis) if symbolic.get("tree_type", "binary") == "binary" else BasisTree.linear(basis)
    return TensorNetworkLayoutData.from_basis_tree_data(BasisTreeData.from_basis_tree(tree))


def main() -> None:
    load_profile()
    if CODE.startswith("@"):
        raise RuntimeError("Please set CODE = 'your_code_label@your_computer_label'")
    code = orm.load_code(CODE)

    modes = _run(
        OhmicRenormModesWorkChain,
        code=code,
        alpha=orm.Float(INPUT["alpha"]),
        s_exponent=orm.Float(INPUT["s_exponent"]),
        omega_c=orm.Float(INPUT["omega_c"]),
        raw_delta=orm.Float(INPUT["raw_delta"]),
        renormalization_p=orm.Float(INPUT["renormalization_p"]),
        n_modes=orm.Int(INPUT["n_modes"]),
    )
    omega_k = modes["bath_modes"].get_array("omega_k")
    c_j2 = modes["bath_modes"].get_array("c_j2")
    delta_eff = modes["output_parameters"].get_dict()["delta_eff"]
    symbolic = _compose_sbm_zt_payload(
        omega_k,
        c_j2,
        delta_eff=delta_eff,
        epsilon=BUILD["epsilon"],
        tree_type=BUILD["tree_type"],
        m_max=BUILD["m_max"],
    )
    tn_layout = _tree_layout_from_symbolic(symbolic)
    out = _run(
        TTNSymbolicDynamicsWorkChain,
        code=code,
        symbolic_inputs=orm.Dict(dict=symbolic),
        tn_layout=tn_layout,
        dt=orm.Float(RUN["dt"]),
        nsteps=orm.Int(RUN["nsteps"]),
        method=orm.Str(RUN["method"]),
    )
    print(f"outputs={sorted(out.keys())}")


if __name__ == "__main__":
    main()
