"""Lossless native OptimizeConfig snapshot for a single-root, fixed-order QC solve.

This file is also embedded in the standalone runtime. It only imports stdlib and Reno.
"""

from __future__ import annotations

import math

from renormalizer.utils import OptimizeConfig


def snapshot_optimization(config):
    """Reject settings this case cannot faithfully reproduce instead of dropping them."""
    if not isinstance(config, OptimizeConfig):
        raise TypeError("optimize_config must be a native OptimizeConfig")
    defaults = OptimizeConfig()
    fields = {"procedure", "method", "algo", "nroots", "e_rtol", "e_atol", "inverse"}
    for name, value in vars(config).items():
        if name not in fields and (name not in vars(defaults) or value != vars(defaults)[name]):
            raise ValueError(f"unsupported non-default OptimizeConfig setting: {name}")
    if config.method not in {"1site", "2site"}:
        raise ValueError("DMRG method must be 1site or 2site")
    if config.algo not in {"davidson", "arpack", "primme"}:
        raise ValueError("unsupported DMRG eigensolver")
    if config.nroots != 1 or config.inverse != 1:
        raise ValueError("this ground-state report requires nroots=1 and inverse=1")
    for name in ("e_rtol", "e_atol"):
        if not math.isfinite(getattr(config, name)) or getattr(config, name) < 0:
            raise ValueError(f"{name} must be finite and nonnegative")
    if not config.procedure:
        raise ValueError("DMRG requires at least one sweep")
    procedure = []
    for bond_dimension, mixing in config.procedure:
        if isinstance(bond_dimension, bool) or not isinstance(bond_dimension, int):
            raise ValueError("this case requires integer fixed bond dimensions in each sweep")
        if bond_dimension < 1 or not math.isfinite(mixing) or not 0 <= mixing <= 1:
            raise ValueError("sweeps require positive bond dimensions and mixing in [0, 1]")
        procedure.append([bond_dimension, float(mixing)])
    return {name: (procedure if name == "procedure" else getattr(config, name))
            for name in sorted(fields)}


def restore_optimization(settings):
    config = OptimizeConfig(procedure=settings["procedure"])
    for name, value in settings.items():
        setattr(config, name, value)
    snapshot_optimization(config)
    return config
