#!/usr/bin/env python
"""SBM MPS: system handwritten, env terms from spectral coefficients."""
from __future__ import annotations

from aiida import load_profile, orm
from aiida.engine import run_get_node

from aiida_renormalizer.data import ModelData, TensorNetworkLayoutData
from aiida_renormalizer.workchains.ohmic_renorm_modes import OhmicRenormModesWorkChain
from aiida_renormalizer.workchains.model_to_mpo import ModelToMPOWorkChain
from aiida_renormalizer.workchains.mpo_to_initial_mps import MPOToInitialMPSWorkChain
from aiida_renormalizer.workchains.mps_dynamics import MPSDynamicsWorkChain

CODE = "reno-script-clean@localhost"  # format: "code_label@computer_label"
INPUT = {
    "alpha": 0.05,
    "s_exponent": 0.7,
    "omega_c": 5.0,
    "raw_delta": 0.4,
    "renormalization_p": 2.0,
    "n_modes": 64,
}
BUILD = {"epsilon": 0.0}
RUN = {"total_time": 20.0, "dt": 0.1, "trajectory_interval": 1}


def _run(wc, **inputs):
    outputs, node = run_get_node(wc, **inputs)
    print(f"[{wc.__name__}] pk={node.pk}")
    return outputs


def _sbm_symbolic_spec(omega_k, c_j2, delta_eff, epsilon):
    basis = [{"kind": "half_spin", "dof": "spin", "sigmaqn": [0, 0]}]
    basis.extend(
        {
            "kind": "sho",
            "dof": f"v_{idx}",
            "omega": float(omega),
            "nbas": 4,
        }
        for idx, omega in enumerate(omega_k)
    )
    hamiltonian = [
        {"symbol": "sigma_z", "dofs": "spin", "factor": float(epsilon)},
        {"symbol": "sigma_x", "dofs": "spin", "factor": float(delta_eff)},
    ]
    for idx, coupling in enumerate(c_j2):
        hamiltonian.append(
            {
                "symbol": "p^2",
                "dofs": f"v_{idx}",
                "factor": 0.5,
            }
        )
        hamiltonian.append(
            {
                "symbol": "x^2",
                "dofs": f"v_{idx}",
                "factor": 0.5 * float(omega_k[idx]) ** 2,
            }
        )
        hamiltonian.append(
            {
                "symbol": "sigma_z x",
                "dofs": ["spin", f"v_{idx}"],
                "factor": float(coupling),
            }
        )
    return basis, hamiltonian


def main() -> None:
    load_profile()
    if CODE.startswith("@"):
        raise RuntimeError("Please set CODE = 'your_code_label@your_computer_label'")
    code = orm.load_code(CODE)

    spectral_outputs = _run(
        OhmicRenormModesWorkChain,
        code=code,
        alpha=orm.Float(INPUT["alpha"]),
        s_exponent=orm.Float(INPUT["s_exponent"]),
        omega_c=orm.Float(INPUT["omega_c"]),
        raw_delta=orm.Float(INPUT["raw_delta"]),
        renormalization_p=orm.Float(INPUT["renormalization_p"]),
        n_modes=orm.Int(INPUT["n_modes"]),
    )
    omega_k = spectral_outputs["bath_modes"].get_array("omega_k")
    c_j2 = spectral_outputs["bath_modes"].get_array("c_j2")
    delta_eff = spectral_outputs["output_parameters"].get_dict()["delta_eff"]

    basis, hamiltonian = _sbm_symbolic_spec(omega_k, c_j2, delta_eff, BUILD["epsilon"])
    model = ModelData.from_symbolic_spec(basis=basis, hamiltonian=hamiltonian)
    tn_layout = TensorNetworkLayoutData.from_chain(model.base.attributes.get("dof_list"))
    mpo_out = _run(ModelToMPOWorkChain, model=model, code=code, tn_layout=tn_layout)
    mpo = mpo_out["mpo"]
    tn_layout = mpo_out.get("output_tn_layout", tn_layout)
    init_out = _run(
        MPOToInitialMPSWorkChain,
        model=model,
        mpo=mpo,
        code=code,
        tn_layout=tn_layout,
    )
    init_mps = init_out["initial_mps"]
    tn_layout = init_out.get("output_tn_layout", tn_layout)
    out = _run(
        MPSDynamicsWorkChain,
        model=model,
        mpo=mpo,
        initial_mps=init_mps,
        tn_layout=tn_layout,
        code=code,
        total_time=orm.Float(RUN["total_time"]),
        dt=orm.Float(RUN["dt"]),
        trajectory_interval=orm.Int(RUN["trajectory_interval"]),
    )
    print(f"final_mps_pk={out['final_mps'].pk}")


if __name__ == "__main__":
    main()
