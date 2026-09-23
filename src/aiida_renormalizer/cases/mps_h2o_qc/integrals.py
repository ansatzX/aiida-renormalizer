"""Record the source integrals and Reno's FCIDUMP spin-orbital conversion."""

from __future__ import annotations

import hashlib

import numpy as np
from aiida import orm
from aiida.engine import calcfunction
from renormalizer.model import h_qc


@calcfunction
def prepare_fcidump(fcidump: orm.SinglefileData, spatial_norbs: orm.Int) -> orm.ArrayData:
    """Use Reno's antisymmetrized two-electron convention; retain input provenance."""
    if spatial_norbs.value < 1:
        raise ValueError("spatial_norbs must be positive")
    with fcidump.as_path() as path:
        h1e, h2e, nuclear_repulsion = h_qc.read_fcidump(str(path), spatial_norbs.value)
    if not (np.isfinite(h1e).all() and np.isfinite(h2e).all()
            and np.isfinite(nuclear_repulsion)):
        raise ValueError("FCIDUMP integrals must be finite")
    node = orm.ArrayData()
    node.set_array("h1e", np.asarray(h1e, dtype=float))
    node.set_array("h2e", np.asarray(h2e, dtype=float))
    node.base.attributes.set_many({
        "nuclear_repulsion": float(nuclear_repulsion),
        "spatial_norbs": spatial_norbs.value,
        "spin_norbs": int(h1e.shape[0]),
        "fcidump_filename": fcidump.filename,
        "fcidump_sha256": hashlib.sha256(fcidump.get_content(mode="rb")).hexdigest(),
    })
    return node
