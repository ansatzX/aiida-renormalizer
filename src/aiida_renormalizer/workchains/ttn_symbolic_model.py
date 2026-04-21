"""WorkChain for declarative symbolic TTN/TTNO model construction."""
from __future__ import annotations

from typing import Any

import numpy as np
from aiida import orm
from aiida.engine import WorkChain

KNOWN_PROCESS_STEPS = {
    "build_sdf",
    "adiabatic_renormalization",
    "discretize_bath",
    "build_symbolic_hamiltonian",
    "ttns_tdvp",
}


def _serialize_basis_item(basis_item: Any) -> dict[str, Any]:
    from renormalizer.model.basis import BasisHalfSpin, BasisSHO

    if isinstance(basis_item, BasisHalfSpin):
        return {
            "kind": "half_spin",
            "dof": basis_item.dof,
            "sigmaqn": np.asarray(basis_item.sigmaqn).tolist(),
        }
    if isinstance(basis_item, BasisSHO):
        return {
            "kind": "sho",
            "dof": basis_item.dof,
            "omega": float(basis_item.omega),
            "nbas": int(basis_item.nbas),
        }
    raise TypeError(f"Unsupported basis item type: {type(basis_item).__name__}")


def _serialize_hamiltonian_item(op: Any) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "symbol": op.symbol,
        "dofs": op.dofs if len(op.dofs) > 1 else op.dofs[0],
        "factor": float(op.factor),
    }
    if op.qn is not None:
        payload["qn"] = np.asarray(op.qn).tolist()
    return payload


def _extract_modes_from_model(model: Any) -> tuple[list[float], list[float]]:
    omega_k: list[float] = []
    c_j2: list[float] = []
    for ph in model.ph_list:
        omega = float(ph.omega[0])
        disp = float(ph.dis[1])
        omega_k.append(omega)
        c_j2.append(float((disp * omega**2) ** 2))
    return omega_k, c_j2


def build_symbolic_inputs_and_metadata(
    *,
    alpha: float,
    s_exponent: float,
    omega_c: float,
    n_modes: int,
    raw_delta: float,
    renormalization_p: float,
    tree_type: str,
    m_max: int,
    process: list[str],
    symbol_map: dict[str, str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build symbolic payload from a declared process list."""
    from renormalizer.model import Phonon, SpinBosonModel
    from renormalizer.sbm import SpectralDensityFunction, param2mollist
    from renormalizer.utils import Quantity

    executed_steps: list[str] = []
    is_sub_ohmic = s_exponent < 1.0

    sdf = SpectralDensityFunction(alpha=alpha, omega_c=Quantity(omega_c), s=s_exponent)
    if "build_sdf" in process:
        executed_steps.append("build_sdf")

    if "adiabatic_renormalization" in process:
        delta_eff, omega_l = sdf.adiabatic_renormalization(Quantity(raw_delta), renormalization_p)
        renormalization_applied = True
        renormalization_scheme = "adiabatic"
        executed_steps.append("adiabatic_renormalization")
    else:
        delta_eff = float(raw_delta)
        omega_l = float(omega_c)
        renormalization_applied = False
        renormalization_scheme = "none"

    if "discretize_bath" in process:
        omega_grid, c_j2_arr = sdf.trapz(n_modes, 0.0, omega_l)
        omega_list, displacement_list = sdf.post_process(omega_grid, c_j2_arr)
        phonons = [Phonon.simplest_phonon(w, d) for w, d in zip(omega_list, displacement_list)]
        model = SpinBosonModel(Quantity(0.0), Quantity(delta_eff), phonons)
        omega_k = np.asarray(omega_grid, dtype=float).tolist()
        c_j2 = np.asarray(c_j2_arr, dtype=float).tolist()
        executed_steps.append("discretize_bath")
    else:
        if is_sub_ohmic:
            raise ValueError("Sub-ohmic (s<1) declaration must include 'discretize_bath'.")
        model = param2mollist(
            alpha,
            Quantity(raw_delta),
            Quantity(omega_c),
            renormalization_p,
            n_modes,
        )
        omega_k, c_j2 = _extract_modes_from_model(model)
        delta_eff = float(model.delta)
        renormalization_applied = True
        renormalization_scheme = "param2mollist"

    if "build_symbolic_hamiltonian" not in process:
        raise ValueError("Declaration must include 'build_symbolic_hamiltonian'.")
    executed_steps.append("build_symbolic_hamiltonian")

    metadata = {
        "declared_process": process,
        "executed_process": executed_steps,
        "renormalization_applied": renormalization_applied,
        "renormalization_scheme": renormalization_scheme,
        "alpha": float(alpha),
        "s_exponent": float(s_exponent),
        "omega_c": float(omega_c),
        "n_modes": int(n_modes),
        "raw_delta": float(raw_delta),
        "delta_eff": float(delta_eff),
        "renormalization_factor": float(delta_eff / raw_delta),
        "renormalization_p": float(renormalization_p),
        "high_frequency_cutoff": float(omega_l),
        "omega_k": omega_k,
        "c_j2": c_j2,
    }

    hamiltonian = [_serialize_hamiltonian_item(item) for item in model.ham_terms]
    if symbol_map:
        for term in hamiltonian:
            old_symbol = str(term["symbol"])
            term["symbol"] = symbol_map.get(old_symbol, old_symbol)

    symbolic_inputs = {
        "basis": [_serialize_basis_item(item) for item in model.basis],
        "hamiltonian": hamiltonian,
        "tree_type": tree_type,
        "m_max": int(m_max),
        "sbm_metadata": metadata,
    }
    return symbolic_inputs, metadata


class TTNSymbolicModelWorkChain(WorkChain):
    """Build symbolic TTN/TTNO inputs from declarative SBM process spec."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("alpha", valid_type=orm.Float, help="SBM coupling strength")
        spec.input("s_exponent", valid_type=orm.Float, default=lambda: orm.Float(1.0))
        spec.input("omega_c", valid_type=orm.Float, help="Bath cutoff frequency")
        spec.input("n_modes", valid_type=orm.Int, help="Number of bath modes")
        spec.input("raw_delta", valid_type=orm.Float, help="Bare tunneling parameter")
        spec.input(
            "renormalization_p",
            valid_type=orm.Float,
            help="Adiabatic renormalization parameter p",
        )
        spec.input("tree_type", valid_type=orm.Str, default=lambda: orm.Str("binary"))
        spec.input("m_max", valid_type=orm.Int, default=lambda: orm.Int(16))
        spec.input("symbol_map", valid_type=orm.Dict, default=lambda: orm.Dict(dict={}))
        spec.input(
            "process",
            valid_type=orm.List,
            default=lambda: orm.List(
                list=[
                    "build_sdf",
                    "adiabatic_renormalization",
                    "discretize_bath",
                    "build_symbolic_hamiltonian",
                    "ttns_tdvp",
                ]
            ),
        )

        spec.output("symbolic_inputs", valid_type=orm.Dict)
        spec.output("output_parameters", valid_type=orm.Dict)

        spec.exit_code(530, "ERROR_INVALID_INPUT", message="Invalid symbolic model inputs")

        spec.outline(cls.setup, cls.build_model, cls.finalize)

    def setup(self):
        process = self.inputs.process.get_list()
        unknown = [step for step in process if step not in KNOWN_PROCESS_STEPS]
        if unknown:
            self.report(f"Unknown process steps: {unknown}")
            return self.exit_codes.ERROR_INVALID_INPUT
        if self.inputs.s_exponent.value < 1.0 and "discretize_bath" not in process:
            self.report("Sub-ohmic (s<1) declaration must include 'discretize_bath'.")
            return self.exit_codes.ERROR_INVALID_INPUT
        if self.inputs.tree_type.value not in {"binary", "linear"}:
            self.report("tree_type must be 'binary' or 'linear'")
            return self.exit_codes.ERROR_INVALID_INPUT

    def build_model(self):
        try:
            symbolic_inputs, metadata = build_symbolic_inputs_and_metadata(
                alpha=self.inputs.alpha.value,
                s_exponent=self.inputs.s_exponent.value,
                omega_c=self.inputs.omega_c.value,
                n_modes=self.inputs.n_modes.value,
                raw_delta=self.inputs.raw_delta.value,
                renormalization_p=self.inputs.renormalization_p.value,
                tree_type=self.inputs.tree_type.value,
                m_max=self.inputs.m_max.value,
                process=self.inputs.process.get_list(),
                symbol_map={str(k): str(v) for k, v in self.inputs.symbol_map.get_dict().items()},
            )
        except (ValueError, KeyError) as exc:
            self.report(str(exc))
            return self.exit_codes.ERROR_INVALID_INPUT
        self.ctx.symbolic_inputs = orm.Dict(dict=symbolic_inputs)
        self.ctx.metadata = metadata

    def finalize(self):
        self.out("symbolic_inputs", self.ctx.symbolic_inputs)
        self.out("output_parameters", orm.Dict(dict=self.ctx.metadata))
