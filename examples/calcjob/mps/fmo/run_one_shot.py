#!/usr/bin/env python
"""FMO case via direct CalcJob chain."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from aiida import load_profile, orm
from aiida.engine import run_get_node

from aiida_renormalizer.calculations.spectra.charge_diffusion import ChargeDiffusionCalcJob
from aiida_renormalizer.data import ModelData

CODE = "reno-script-clean@localhost"
INPUT = {"sdf_json": Path(__file__).with_name("fmo_sdf.json")}
RUN = {"temperature": 0.0, "init_electron": "fc", "stop_at_edge": False, "rdm": False}


def _build_fmo_modeldata(path: Path) -> ModelData:
    from renormalizer.model import HolsteinModel, Mol, Phonon
    from renormalizer.utils import Quantity
    from renormalizer.utils.constant import cm2au

    with path.open() as handle:
        sdf_values = np.asarray(json.load(handle), dtype=float)

    j_matrix_cm = np.array(
        [
            [310, -98, 6, -6, 7, -12, -10, 38],
            [-98, 230, 30, 7, 2, 12, 5, 8],
            [6, 30, 0, -59, -2, -10, 5, 2],
            [-6, 7, -59, 180, -65, -17, -65, -2],
            [7, 2, -2, -65, 405, 89, -6, 5],
            [-12, 11, -10, -17, 89, 320, 32, -10],
            [-10, 5, 5, -64, -6, 32, 270, -11],
            [38, 8, 2, -2, 5, -10, -11, 505],
        ],
        dtype=float,
    )

    n_phonons = 35
    total_hr = 0.42
    omegas_cm = np.linspace(2, 300, n_phonons)
    omegas_au = omegas_cm * cm2au
    hr_factors = np.interp(omegas_cm, sdf_values[:, 0], sdf_values[:, 1])
    hr_factors *= total_hr / hr_factors.sum()
    lams = hr_factors * omegas_au
    phonons = [
        Phonon.simplest_phonon(Quantity(omega), Quantity(lam), lam=True)
        for omega, lam in zip(omegas_au, lams)
    ]

    j_matrix_au = j_matrix_cm * cm2au
    mols = [Mol(Quantity(onsite), phonons) for onsite in np.diag(j_matrix_au)]
    arrangement = np.array([7, 5, 3, 1, 2, 4, 6]) - 1
    model = HolsteinModel(
        list(np.array(mols)[arrangement]),
        j_matrix_au[arrangement][:, arrangement],
    )
    return ModelData.from_model(model)


def main() -> None:
    load_profile()
    if CODE.startswith("@"):
        raise RuntimeError("Please set CODE = 'your_code_label@your_computer_label'")

    out, node = run_get_node(
        ChargeDiffusionCalcJob,
        code=orm.load_code(CODE),
        model=_build_fmo_modeldata(INPUT["sdf_json"]),
        temperature=orm.Float(RUN["temperature"]),
        init_electron=orm.Str(RUN["init_electron"]),
        stop_at_edge=orm.Bool(RUN["stop_at_edge"]),
        rdm=orm.Bool(RUN["rdm"]),
    )
    print(f"[ChargeDiffusionCalcJob] pk={node.pk}")
    print(f"output_parameters_pk={out['output_parameters'].pk}")


if __name__ == "__main__":
    main()
