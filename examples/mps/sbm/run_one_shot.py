#!/usr/bin/env python
"""SBM MPS: system terms handwritten; env terms from spectral discretization."""
from __future__ import annotations

import numpy as np
from aiida import load_profile, orm
from aiida.engine import run_get_node

from aiida_renormalizer.data.model import ModelData
from aiida_renormalizer.workchains.model_to_mpo import ModelToMPOWorkChain
from aiida_renormalizer.workchains.mpo_to_initial_mps import MPOToInitialMPSWorkChain
from aiida_renormalizer.workchains.mps_dynamics import MPSDynamicsWorkChain
from aiida_renormalizer.workchains.sbm_spectral_modes import SbmSpectralModesWorkChain

CODE = "@localhost"  # format: "code_label@computer_label"
SPECTRAL = {
    "alpha": 0.05,
    "s_exponent": 0.7,
    "omega_c": 5.0,
    "raw_delta": 0.4,
    "renormalization_p": 2.0,
    "n_modes": 64,
    "discretization": "trapz",
}
SYSTEM = {"epsilon": 0.0}
DYNAMICS = {"total_time": 20.0, "dt": 0.1, "trajectory_interval": 1}

def _run(wc, **inputs):
    outputs, node = run_get_node(wc, **inputs)
    print(f"[{wc.__name__}] pk={node.pk}")
    return outputs
def _build_model_data(omega_k, c_j2, delta_eff, epsilon):
    c_k = np.sqrt(c_j2)
    nbas = np.maximum(16.0 * c_j2 / np.maximum(omega_k, 1e-12) ** 3, 4.0)
    nbas = np.round(nbas).astype(int)
    basis = [{"kind": "half_spin", "dof": "spin", "sigmaqn": [0, 0]}]
    basis += [
        {"kind": "sho", "dof": f"v_{i}", "omega": float(w), "nbas": int(n)}
        for i, (w, n) in enumerate(zip(omega_k, nbas))
    ]
    terms = [
        {"symbol": "sigma_z", "dofs": "spin", "factor": float(epsilon)},
        {"symbol": "sigma_x", "dofs": "spin", "factor": float(delta_eff)},
    ]
    for i, w in enumerate(omega_k):
        terms.extend(
            [
                {"symbol": "p^2", "dofs": f"v_{i}", "factor": 0.5},
                {"symbol": "x^2", "dofs": f"v_{i}", "factor": 0.5 * float(w) ** 2},
            ]
        )
        terms.append(
            {
                "symbol": "sigma_z x",
                "dofs": ["spin", f"v_{i}"],
                "factor": 0.5 * float(c_k[i]),
            }
        )
    return ModelData.from_symbolic_spec(basis=basis, hamiltonian=terms)
def main() -> None:
    load_profile()
    if CODE.startswith("@"):
        raise RuntimeError("Please set CODE = 'your_code_label@your_computer_label'")
    spectral_outputs = _run(
        SbmSpectralModesWorkChain,
        alpha=orm.Float(SPECTRAL["alpha"]),
        s_exponent=orm.Float(SPECTRAL["s_exponent"]),
        omega_c=orm.Float(SPECTRAL["omega_c"]),
        raw_delta=orm.Float(SPECTRAL["raw_delta"]),
        renormalization_p=orm.Float(SPECTRAL["renormalization_p"]),
        n_modes=orm.Int(SPECTRAL["n_modes"]),
        discretization=orm.Str(SPECTRAL["discretization"]),
    )
    omega_k = spectral_outputs["bath_modes"].get_array("omega_k")
    c_j2 = spectral_outputs["bath_modes"].get_array("c_j2")
    delta_eff = spectral_outputs["output_parameters"].get_dict()["delta_eff"]
    code = orm.load_code(CODE)
    model = _build_model_data(omega_k, c_j2, delta_eff, SYSTEM["epsilon"])
    mpo = _run(ModelToMPOWorkChain, model=model, code=code)["mpo"]
    init_mps = _run(MPOToInitialMPSWorkChain, model=model, mpo=mpo, code=code)["initial_mps"]
    out = _run(
        MPSDynamicsWorkChain,
        model=model,
        mpo=mpo,
        initial_mps=init_mps,
        code=code,
        total_time=orm.Float(DYNAMICS["total_time"]),
        dt=orm.Float(DYNAMICS["dt"]),
        trajectory_interval=orm.Int(DYNAMICS["trajectory_interval"]),
    )
    print(f"final_mps_pk={out['final_mps'].pk}")
if __name__ == "__main__":
    main()
