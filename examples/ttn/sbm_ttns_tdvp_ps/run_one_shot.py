#!/usr/bin/env python
"""Run symbolic TTNS TDVP-PS evolution in one official CalcJob."""
from __future__ import annotations

from pathlib import Path
import sys

from aiida import orm
from aiida import load_profile
from aiida.manage import get_manager

from aiida_renormalizer.calculations.ttn.ttns_symbolic_evolve import TtnsSymbolicEvolveCalcJob


def _resolve_code() -> orm.Code:
    qb = orm.QueryBuilder().append(
        orm.Code,
        filters={"attributes.input_plugin": "reno.script"},
    )
    result = qb.first()
    if result is not None:
        return result[0]

    computer = orm.load_computer("localhost")
    return orm.InstalledCode(
        label="reno-script-auto",
        computer=computer,
        filepath_executable=sys.executable,
        default_calc_job_plugin="reno.script",
        description="Auto-created by examples/ttn/sbm_ttns_tdvp_ps/run_one_shot.py",
    ).store()


def main() -> None:
    load_profile()
    code = _resolve_code()
    repo_root = Path(__file__).resolve().parents[3]
    artifact_base = repo_root / "tmp"
    artifact_base.mkdir(parents=True, exist_ok=True)

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
    )

    builder = TtnsSymbolicEvolveCalcJob.get_builder()
    builder.code = code
    builder.symbolic_inputs = symbolic_inputs
    builder.dt = orm.Float(0.05)
    builder.nsteps = orm.Int(40)
    builder.method = orm.Str("tdvp_ps")
    builder.metadata.label = "example_sbm_ttns_tdvp_ps_one_shot"
    builder.metadata.options.artifact_storage_backend = "posix"
    builder.metadata.options.artifact_storage_base = str(artifact_base)
    builder.metadata.options.resources = {"num_machines": 1, "num_mpiprocs_per_machine": 1}

    runner = get_manager().create_runner(with_persistence=True, communicator=None, broker_submit=False)
    result, node = runner.run_get_node(builder)
    if "output_parameters" not in result:
        raise RuntimeError(
            f"output_parameters missing, exit_status={node.exit_status}, exit_message={node.exit_message}"
        )
    params = result["output_parameters"].get_dict()
    time_points = params["time_points"]
    if abs(time_points[0]) > 1e-14:
        raise RuntimeError(f"Expected first trajectory point at t=0, got {time_points[0]}")

    print(f"CalcJob PK: {node.pk}")
    if "output_ttns" in result:
        print(f"Output TTNS PK: {result['output_ttns'].pk}")
    print(f"First 3 time points: {time_points[:3]}")
    print(f"First 3 energies: {params['energy_trajectory'][:3]}")


if __name__ == "__main__":
    main()
