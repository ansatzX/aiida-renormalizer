#!/usr/bin/env python
"""QC case via direct CalcJob chain."""

from __future__ import annotations

from pathlib import Path

from aiida import load_profile, orm
from aiida.engine import run_get_node

from aiida_renormalizer.calculations.basic.build_mpo import BuildMPOCalcJob
from aiida_renormalizer.calculations.composite.dmrg import DMRGCalcJob
from aiida_renormalizer.data import ModelData

CODE = "reno-script-clean@localhost"
INPUT = {"fcidump": Path(__file__).with_name("h2o_fcidump.txt"), "spatial_norbs": 7}


def _build_qc_modeldata(fcidump_path: Path, spatial_norbs: int):
    from renormalizer.model import h_qc
    from renormalizer.model.model import Model

    h1e, h2e, nuc = h_qc.read_fcidump(str(fcidump_path), spatial_norbs)
    basis, ham_terms = h_qc.qc_model(h1e, h2e)
    return ModelData.from_model(Model(basis, ham_terms)), float(nuc)


def main() -> None:
    load_profile()
    if CODE.startswith("@"):
        raise RuntimeError("Please set CODE = 'your_code_label@your_computer_label'")

    model, nuclear_repulsion = _build_qc_modeldata(INPUT["fcidump"], INPUT["spatial_norbs"])
    code = orm.load_code(CODE)

    mpo_out, mpo_node = run_get_node(BuildMPOCalcJob, code=code, model=model)
    gs_out, gs_node = run_get_node(DMRGCalcJob, code=code, model=model, mpo=mpo_out["output_mpo"])

    e_elec = gs_out["output_parameters"].get_dict().get("e", 0.0)
    print(f"[BuildMPOCalcJob] pk={mpo_node.pk}")
    print(f"[DMRGCalcJob] pk={gs_node.pk}")
    print(f"electronic_energy={e_elec}")
    print(f"total_energy={e_elec + nuclear_repulsion}")


if __name__ == "__main__":
    main()
