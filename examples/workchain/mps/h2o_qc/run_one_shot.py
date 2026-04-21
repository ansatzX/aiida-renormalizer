#!/usr/bin/env python
"""QC case migrated from ori_examples/h2o_qc.py."""
from __future__ import annotations

from pathlib import Path

from aiida import load_profile, orm
from aiida.engine import run_get_node

from aiida_renormalizer.data import ModelData, TensorNetworkLayoutData
from aiida_renormalizer.workchains.ground_state import GroundStateWorkChain
from aiida_renormalizer.workchains.model_to_mpo import ModelToMPOWorkChain

CODE = "reno-script-clean@localhost"
INPUT = {
    "fcidump": Path(__file__).with_name("h2o_fcidump.txt"),
    "spatial_norbs": 7,
}
RUN = {"strategy": "dmrg"}


def _build_qc_modeldata(fcidump_path: Path, spatial_norbs: int):
    from renormalizer.model import h_qc
    from renormalizer.model.model import Model

    h1e, h2e, nuc = h_qc.read_fcidump(str(fcidump_path), spatial_norbs)
    basis, ham_terms = h_qc.qc_model(h1e, h2e)
    return ModelData.from_model(Model(basis, ham_terms)), float(nuc)


def _run(wc, **inputs):
    outputs, node = run_get_node(wc, **inputs)
    print(f"[{wc.__name__}] pk={node.pk}")
    return outputs


def main() -> None:
    load_profile()
    if CODE.startswith("@"):
        raise RuntimeError("Please set CODE = 'your_code_label@your_computer_label'")

    model, nuclear_repulsion = _build_qc_modeldata(INPUT["fcidump"], INPUT["spatial_norbs"])
    tn_layout = TensorNetworkLayoutData.from_chain(model.base.attributes.get("dof_list"))
    code = orm.load_code(CODE)
    mpo_out = _run(ModelToMPOWorkChain, model=model, code=code, tn_layout=tn_layout)
    mpo = mpo_out["mpo"]
    tn_layout = mpo_out.get("output_tn_layout", tn_layout)
    out = _run(
        GroundStateWorkChain,
        model=model,
        mpo=mpo,
        tn_layout=tn_layout,
        strategy=orm.Str(RUN["strategy"]),
        code=code,
    )
    e_elec = out["energy"].value
    print(f"electronic_energy={e_elec}")
    print(f"total_energy={e_elec + nuclear_repulsion}")


if __name__ == "__main__":
    main()
