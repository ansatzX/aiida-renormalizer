#!/usr/bin/env python
"""SBM MPS via direct CalcJob chain."""

from __future__ import annotations

from aiida import load_profile, orm
from aiida.engine import run_get_node

from aiida_renormalizer.calculations.basic.build_mpo import BuildMPOCalcJob
from aiida_renormalizer.calculations.bath.ohmic_renorm_modes import OhmicRenormModesCalcJob
from aiida_renormalizer.calculations.composite.dmrg import DMRGCalcJob
from aiida_renormalizer.calculations.composite.tdvp import TDVPCalcJob
from aiida_renormalizer.data.model import ModelData

CODE = "reno-script-clean@localhost"
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

    spectral_out, spectral_node = run_get_node(
        OhmicRenormModesCalcJob,
        code=code,
        alpha=orm.Float(INPUT["alpha"]),
        s_exponent=orm.Float(INPUT["s_exponent"]),
        omega_c=orm.Float(INPUT["omega_c"]),
        raw_delta=orm.Float(INPUT["raw_delta"]),
        renormalization_p=orm.Float(INPUT["renormalization_p"]),
        n_modes=orm.Int(INPUT["n_modes"]),
    )
    params = spectral_out["output_parameters"].get_dict()
    basis, hamiltonian = _sbm_symbolic_spec(params["omega_k"], params["c_j2"], params["delta_eff"], BUILD["epsilon"])
    model = ModelData.from_symbolic_spec(basis=basis, hamiltonian=hamiltonian)

    mpo_out, mpo_node = run_get_node(BuildMPOCalcJob, code=code, model=model)
    dmrg_out, dmrg_node = run_get_node(
        DMRGCalcJob, code=code, model=model, mpo=mpo_out["output_mpo"]
    )
    tdvp_out, tdvp_node = run_get_node(
        TDVPCalcJob,
        code=code,
        model=model,
        mpo=mpo_out["output_mpo"],
        initial_mps=dmrg_out["output_mps"],
        total_time=orm.Float(RUN["total_time"]),
        dt=orm.Float(RUN["dt"]),
        trajectory_interval=orm.Int(RUN["trajectory_interval"]),
    )

    print(f"[OhmicRenormModesCalcJob] pk={spectral_node.pk}")
    print(f"[BuildMPOCalcJob] pk={mpo_node.pk}")
    print(f"[DMRGCalcJob] pk={dmrg_node.pk}")
    print(f"[TDVPCalcJob] pk={tdvp_node.pk}")
    print(f"final_mps_pk={tdvp_out['output_mps'].pk}")


if __name__ == "__main__":
    main()
