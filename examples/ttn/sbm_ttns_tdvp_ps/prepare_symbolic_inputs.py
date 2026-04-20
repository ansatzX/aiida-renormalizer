#!/usr/bin/env python
"""Prepare symbolic input Dict node for TtnsSymbolicEvolveCalcJob."""
from __future__ import annotations

import sys

from aiida import orm


def _resolve_code() -> orm.Code:
    qb = orm.QueryBuilder().append(
        orm.Code,
        filters={"attributes.input_plugin": "reno.script"},
    )
    result = qb.first()
    if result is not None:
        return result[0]

    # Auto-create a local Python code for reno.script if missing.
    computer = orm.load_computer("localhost")
    code = orm.InstalledCode(
        label="reno-script-auto",
        computer=computer,
        filepath_executable=sys.executable,
        default_calc_job_plugin="reno.script",
        description="Auto-created by examples/ttn/sbm_ttns_tdvp_ps/prepare_symbolic_inputs.py",
    ).store()
    return code


def main() -> None:
    symbolic_inputs = orm.Dict(
        dict={
            "basis": [
                {"kind": "half_spin", "dof": "spin", "sigmaqn": [0, 0]},
                {"kind": "sho", "dof": "v_0", "omega": 0.8, "nbas": 6},
                {"kind": "sho", "dof": "v_1", "omega": 1.2, "nbas": 6},
            ],
            "hamiltonian": [
                {"symbol": "sigma_x", "dofs": "spin", "factor": 0.4},
                {"symbol": "sigma_z", "dofs": "spin", "factor": 0.05},
                {"symbol": r"b^\dagger b", "dofs": "v_0", "factor": 0.8},
                {"symbol": r"b^\dagger b", "dofs": "v_1", "factor": 1.2},
                {"symbol": r"sigma_z x", "dofs": ["spin", "v_0"], "factor": 0.08},
                {"symbol": r"sigma_z x", "dofs": ["spin", "v_1"], "factor": 0.06},
            ],
            "tree_type": "binary",
            "m_max": 16,
        }
    ).store()

    code = _resolve_code()
    print(f"SYMBOLIC_INPUTS_PK={symbolic_inputs.pk}")
    print(f"CODE_PK={code.pk}")


if __name__ == "__main__":
    main()
