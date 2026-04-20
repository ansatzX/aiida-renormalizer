#!/usr/bin/env python
"""Launch symbolic TTNS evolve CalcJob from CLI parameters."""
from __future__ import annotations

import argparse
from pathlib import Path

from aiida import orm
from aiida import load_profile
from aiida.manage import get_manager

from aiida_renormalizer.calculations.ttn.ttns_symbolic_evolve import TtnsSymbolicEvolveCalcJob


def main() -> None:
    load_profile()
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-pk", type=int, required=True)
    parser.add_argument("--symbolic-inputs-pk", type=int, required=True)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--nsteps", type=int, default=40)
    parser.add_argument("--method", type=str, default="tdvp_ps")
    parser.add_argument("--artifact-base", type=str, default="")
    args = parser.parse_args()

    artifact_base = args.artifact_base
    if not artifact_base:
        repo_root = Path(__file__).resolve().parents[3]
        artifact_base = str(repo_root / "tmp")

    builder = TtnsSymbolicEvolveCalcJob.get_builder()
    builder.code = orm.load_node(args.code_pk)
    builder.symbolic_inputs = orm.load_node(args.symbolic_inputs_pk)
    builder.dt = orm.Float(args.dt)
    builder.nsteps = orm.Int(args.nsteps)
    builder.method = orm.Str(args.method)
    builder.metadata.label = "example_sbm_ttns_tdvp_ps_via_verdi"
    builder.metadata.options.artifact_storage_backend = "posix"
    builder.metadata.options.artifact_storage_base = artifact_base
    builder.metadata.options.resources = {"num_machines": 1, "num_mpiprocs_per_machine": 1}

    # Use a runner without communicator so the example does not require RabbitMQ.
    runner = get_manager().create_runner(with_persistence=True, communicator=None, broker_submit=False)
    result, node = runner.run_get_node(builder)
    print(f"PROCESS_PK={node.pk}")
    if "output_parameters" not in result:
        print(f"EXIT_STATUS={node.exit_status}")
        print(f"EXIT_MESSAGE={node.exit_message}")
        if "retrieved" in node.outputs:
            names = sorted(node.outputs.retrieved.list_object_names())
            print(f"RETRIEVED={','.join(names)}")
        raise RuntimeError("output_parameters missing")
    params = result["output_parameters"].get_dict()
    print(f"FINAL_ENERGY={params['final_energy']}")


if __name__ == "__main__":
    main()
