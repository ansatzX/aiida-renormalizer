#!/usr/bin/env python
"""Kubo transport case via direct CalcJob chain."""

from __future__ import annotations

from pathlib import Path

import yaml
from aiida import load_profile, orm
from aiida.engine import run_get_node

from aiida_renormalizer.calculations.spectra.kubo import KuboCalcJob
from aiida_renormalizer.data import ModelData

CODE = "reno-script-clean@localhost"
INPUT = {"param_file": Path(__file__).with_name("std.yaml")}
RUN = {"insteps": 1}


def _model_from_yaml_params(params):
    from renormalizer.model import load_from_dict

    model, temperature = load_from_dict(params, 3, False)
    if hasattr(temperature, "as_au"):
        temperature = float(temperature.as_au())
    return ModelData.from_model(model), float(temperature)


def main() -> None:
    load_profile()
    if CODE.startswith("@"):
        raise RuntimeError("Please set CODE = 'your_code_label@your_computer_label'")

    with INPUT["param_file"].open() as handle:
        params = yaml.safe_load(handle)
    model, temperature = _model_from_yaml_params(params)

    out, node = run_get_node(
        KuboCalcJob,
        code=orm.load_code(CODE),
        model=model,
        temperature=orm.Float(max(temperature, 1e-6)),
        insteps=orm.Int(RUN["insteps"]),
    )
    print(f"[KuboCalcJob] pk={node.pk}")
    print(f"output_parameters_pk={out['output_parameters'].pk}")


if __name__ == "__main__":
    main()
