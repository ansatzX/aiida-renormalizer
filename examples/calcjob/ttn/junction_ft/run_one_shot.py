#!/usr/bin/env python
"""TTN junction finite-temperature case via direct CalcJob chain."""

from __future__ import annotations

import numpy as np
from aiida import load_profile, orm
from aiida.engine import run_get_node

from aiida_renormalizer.calculations.bath.ohmic_renorm_modes import OhmicRenormModesCalcJob
from aiida_renormalizer.calculations.ttn.ttns_symbolic_evolve import TTNSSymbolicEvolveCalcJob

CODE = "reno-script-clean@localhost"
INPUT = {
    "phonon": {
        "alpha": 0.2,
        "s_exponent": 1.0,
        "omega_c": 1.0,
        "raw_delta": 1.0,
        "renormalization_p": 1.0,
        "n_modes": 128,
    },
    "electrode": {"n_e_mode": 160, "beta_e": 1.0, "alpha_e": 0.2, "bias_v": 0.1},
}
BUILD = {"temperature": 2.0, "initial_occupied": True, "tree_type": "binary", "m_max": 20}
RUN = {"dt": 0.1, "nsteps": 100, "method": "tdvp_ps"}


def _z_string_symbol(length: int, left: str = "+", right: str = "-") -> str:
    return f"{left} " + "Z " * length + right


def _compose_junction_payload(
    *,
    n_e_mode: int,
    e_k,
    mu_l: float,
    mu_r: float,
    alpha_e: float,
    beta_e: float,
    rho_e: float,
    omega_k,
    c_j2,
    temperature: float,
    initial_occupied: bool,
    m_max: int,
    tree_type: str = "binary",
):
    omega = np.asarray(omega_k, dtype=float)
    c2 = np.asarray(c_j2, dtype=float)
    c = np.sqrt(c2)
    e_k_l = np.asarray(e_k, dtype=float) - float(mu_l)
    e_k_r = np.asarray(e_k, dtype=float) - float(mu_r)
    mode_with_e = [(f"L{i}", e) for i, e in enumerate(e_k_l)] + [
        (f"R{i}", e) for i, e in enumerate(e_k_r)
    ]
    mode_with_e.sort(key=lambda x: x[1])

    basis, dofs = [], []
    first_positive = True
    finite_t = float(temperature) > 0
    for mode, e in mode_with_e:
        if e > 0 and first_positive:
            first_positive = False
            basis.append({"kind": "half_spin", "dof": "s", "sigmaqn": [0, 0]})
            dofs.append("s")
        if finite_t:
            for branch in ("p", "q"):
                dof = f"{mode}_{branch}"
                basis.append({"kind": "half_spin", "dof": dof, "sigmaqn": [0, 0]})
                dofs.append(dof)
        else:
            basis.append({"kind": "half_spin", "dof": mode, "sigmaqn": [0, 0]})
            dofs.append(mode)
    if "s" not in dofs:
        basis.insert(0, {"kind": "half_spin", "dof": "s", "sigmaqn": [0, 0]})
        dofs.insert(0, "s")
    s_idx = dofs.index("s")
    hamiltonian = []

    def z_path(idx: int):
        if idx < s_idx:
            return dofs[idx + 1 : s_idx]
        return dofs[s_idx + 1 : idx]

    for mode, e in mode_with_e:
        mu = float(mu_l if mode.startswith("L") else mu_r)
        v2 = (
            alpha_e**2
            / beta_e**2
            * np.sqrt(max(4 * beta_e**2 - (e + mu) ** 2, 1e-14))
            / (2 * np.pi * rho_e)
        )
        v = float(np.sqrt(max(v2, 0.0)))
        if finite_t:
            theta = float(np.arctan(np.exp(-float(temperature) * e / 2.0)))
            for branch, sign, amp in (
                ("p", +1.0, v * np.cos(theta)),
                ("q", -1.0, v * np.sin(theta)),
            ):
                dof = f"{mode}_{branch}"
                hamiltonian.append({"symbol": "+ -", "dofs": dof, "factor": sign * (e + mu)})
                z_dofs = z_path(dofs.index(dof))
                if branch == "p":
                    hamiltonian.extend(
                        [
                            {
                                "symbol": _z_string_symbol(len(z_dofs), "+", "-"),
                                "dofs": [dof, *z_dofs, "s"],
                                "factor": amp,
                            },
                            {
                                "symbol": _z_string_symbol(len(z_dofs), "-", "+"),
                                "dofs": [dof, *z_dofs, "s"],
                                "factor": amp,
                            },
                        ]
                    )
                else:
                    hamiltonian.extend(
                        [
                            {
                                "symbol": _z_string_symbol(len(z_dofs), "-", "-"),
                                "dofs": [dof, *z_dofs, "s"],
                                "factor": amp,
                            },
                            {
                                "symbol": _z_string_symbol(len(z_dofs), "+", "+"),
                                "dofs": [dof, *z_dofs, "s"],
                                "factor": amp,
                            },
                        ]
                    )
        else:
            dof = mode
            hamiltonian.append({"symbol": "+ -", "dofs": dof, "factor": e + mu})
            z_dofs = z_path(dofs.index(dof))
            hamiltonian.extend(
                [
                    {
                        "symbol": _z_string_symbol(len(z_dofs), "+", "-"),
                        "dofs": [dof, *z_dofs, "s"],
                        "factor": v,
                    },
                    {
                        "symbol": _z_string_symbol(len(z_dofs), "-", "+"),
                        "dofs": [dof, *z_dofs, "s"],
                        "factor": v,
                    },
                ]
            )

    if initial_occupied:
        hamiltonian.append(
            {"symbol": "+ -", "dofs": "s", "factor": float(-4.0 * (c**2 / omega**2).sum())}
        )
    if finite_t:
        theta_array = np.arctanh(np.exp(-omega * float(temperature) / 2.0))
        nbas = np.maximum(16 * c2 / np.maximum(omega, 1e-12) ** 3 * np.cosh(theta_array) ** 2, 4.0)
        nbas = np.minimum(np.round(nbas).astype(int), 512)
        for i, (w, n, ci, th) in enumerate(zip(omega, nbas, c, theta_array)):
            vp, vq = f"v_{i}_p", f"v_{i}_q"
            basis.extend(
                [
                    {"kind": "sho", "dof": vp, "omega": float(w), "nbas": int(n)},
                    {"kind": "sho", "dof": vq, "omega": float(w), "nbas": int(n)},
                ]
            )
            hamiltonian.extend(
                [
                    {"symbol": "p^2", "dofs": vp, "factor": 0.5},
                    {"symbol": "x^2", "dofs": vp, "factor": 0.5 * float(w) ** 2},
                    {"symbol": "p^2", "dofs": vq, "factor": -0.5},
                    {"symbol": "x^2", "dofs": vq, "factor": -0.5 * float(w) ** 2},
                    {
                        "symbol": "+ - x",
                        "dofs": ["s", "s", vp],
                        "factor": float(2 * ci * np.cosh(th)),
                    },
                    {
                        "symbol": "+ - x",
                        "dofs": ["s", "s", vq],
                        "factor": float(2 * ci * np.sinh(th)),
                    },
                ]
            )
    else:
        nbas = np.maximum(16 * c2 / np.maximum(omega, 1e-12) ** 3, 4.0)
        nbas = np.round(nbas).astype(int)
        for i, (w, n, ci) in enumerate(zip(omega, nbas, c)):
            vib = f"v_{i}"
            basis.append({"kind": "sho", "dof": vib, "omega": float(w), "nbas": int(n)})
            hamiltonian.extend(
                [
                    {"symbol": "p^2", "dofs": vib, "factor": 0.5},
                    {"symbol": "x^2", "dofs": vib, "factor": 0.5 * float(w) ** 2},
                    {"symbol": "+ - x", "dofs": ["s", "s", vib], "factor": float(2 * ci)},
                ]
            )
    return {
        "basis": basis,
        "hamiltonian": hamiltonian,
        "tree_type": tree_type,
        "m_max": int(m_max),
        "n_e_mode": int(n_e_mode),
    }


def main() -> None:
    load_profile()
    if CODE.startswith("@"):
        raise RuntimeError("Please set CODE = 'your_code_label@your_computer_label'")
    code = orm.load_code(CODE)

    spectral_out, spectral_node = run_get_node(
        OhmicRenormModesCalcJob,
        code=code,
        alpha=orm.Float(INPUT["phonon"]["alpha"]),
        s_exponent=orm.Float(INPUT["phonon"]["s_exponent"]),
        omega_c=orm.Float(INPUT["phonon"]["omega_c"]),
        raw_delta=orm.Float(INPUT["phonon"]["raw_delta"]),
        renormalization_p=orm.Float(INPUT["phonon"]["renormalization_p"]),
        n_modes=orm.Int(INPUT["phonon"]["n_modes"]),
    )
    modes = spectral_out["output_parameters"].get_dict()

    n_e_mode = INPUT["electrode"]["n_e_mode"]
    beta_e = INPUT["electrode"]["beta_e"]
    bias_v = INPUT["electrode"]["bias_v"]
    e_k = np.arange(1, n_e_mode + 1) / (n_e_mode + 1) * 4 * beta_e - 2 * beta_e
    rho_e = 1.0 / (e_k[1] - e_k[0])
    symbolic = _compose_junction_payload(
        n_e_mode=n_e_mode,
        e_k=e_k,
        mu_l=bias_v / 2,
        mu_r=-bias_v / 2,
        alpha_e=INPUT["electrode"]["alpha_e"],
        beta_e=beta_e,
        rho_e=rho_e,
        omega_k=modes["omega_k"],
        c_j2=modes["c_j2"],
        temperature=BUILD["temperature"],
        initial_occupied=BUILD["initial_occupied"],
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
