#!/usr/bin/env python
"""TTN SBM finite-temperature case via direct CalcJob chain."""

from __future__ import annotations

import numpy as np
from aiida import load_profile, orm
from aiida.engine import run_get_node

from aiida_renormalizer.calculations.bath.ohmic_renorm_modes import OhmicRenormModesCalcJob
from aiida_renormalizer.calculations.ttn.ttns_symbolic_evolve import TTNSSymbolicEvolveCalcJob

CODE = "reno-script-clean@localhost"
INPUT = {
    "alpha": 1.0,
    "s_exponent": 0.5,
    "omega_c": 1.0,
    "raw_delta": 1.0,
    "renormalization_p": 1.0,
    "n_modes": 128,
}
BUILD = {"epsilon": 0.0, "temperature": 2.0, "tree_type": "binary", "m_max": 20}
RUN = {"dt": 0.1, "nsteps": 200, "method": "tdvp_ps"}


def _compose_sbm_ft_payload(
    omega_k,
    c_j2,
    *,
    delta_eff: float,
    epsilon: float,
    temperature: float,
    m_max: int,
    tree_type: str = "binary",
):
    omega = np.asarray(omega_k, dtype=float)
    c2 = np.asarray(c_j2, dtype=float)
    c = np.sqrt(c2)
    theta = np.arctanh(np.exp(-omega / max(float(temperature), 1e-12) / 2.0))
    nbas = np.maximum(16 * c2 / np.maximum(omega, 1e-12) ** 3 * np.cosh(theta) ** 2, 4.0)
    nbas = np.minimum(np.round(nbas).astype(int), 512)

    basis = [{"kind": "half_spin", "dof": "spin", "sigmaqn": [0, 0]}]
    for i, (w, n) in enumerate(zip(omega, nbas)):
        basis.extend(
            [
                {"kind": "sho", "dof": f"v_{i}_p", "omega": float(w), "nbas": int(n)},
                {"kind": "sho", "dof": f"v_{i}_q", "omega": float(w), "nbas": int(n)},
            ]
        )
    hamiltonian = [
        {"symbol": "sigma_z", "dofs": "spin", "factor": float(epsilon)},
        {"symbol": "sigma_x", "dofs": "spin", "factor": float(delta_eff)},
    ]
    for i, (w, ci, th) in enumerate(zip(omega, c, theta)):
        vp, vq = f"v_{i}_p", f"v_{i}_q"
        hamiltonian.extend(
            [
                {"symbol": "p^2", "dofs": vp, "factor": 0.5},
                {"symbol": "x^2", "dofs": vp, "factor": 0.5 * float(w) ** 2},
                {"symbol": "p^2", "dofs": vq, "factor": -0.5},
                {"symbol": "x^2", "dofs": vq, "factor": -0.5 * float(w) ** 2},
                {"symbol": "sigma_z x", "dofs": ["spin", vp], "factor": float(np.cosh(th) * ci)},
                {"symbol": "sigma_z x", "dofs": ["spin", vq], "factor": float(np.sinh(th) * ci)},
            ]
        )
    return {"basis": basis, "hamiltonian": hamiltonian, "tree_type": tree_type, "m_max": int(m_max)}


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
    symbolic = _compose_sbm_ft_payload(
        params["omega_k"],
        params["c_j2"],
        delta_eff=params["delta_eff"],
        epsilon=BUILD["epsilon"],
        temperature=BUILD["temperature"],
        tree_type=BUILD["tree_type"],
        m_max=BUILD["m_max"],
    )

    evolve_out, evolve_node = run_get_node(
        TTNSSymbolicEvolveCalcJob,
        code=code,
        symbolic_inputs=orm.Dict(dict=symbolic),
        dt=orm.Float(RUN["dt"]),
        nsteps=orm.Int(RUN["nsteps"]),
        method=orm.Str(RUN["method"]),
    )
    print(f"[OhmicRenormModesCalcJob] pk={spectral_node.pk}")
    print(f"[TTNSSymbolicEvolveCalcJob] pk={evolve_node.pk}")
    print(f"outputs={sorted(evolve_out.keys())}")


if __name__ == "__main__":
    main()
