"""CalcJob for computing site occupations."""
from __future__ import annotations

import os
import tempfile

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import ModelData, MpsData


class ComputeOccupationsCalcJob(RenoBaseCalcJob):
    """Compute occupation numbers for each site.

    Corresponds to Reno API: mps.expectation(n_op) per site
    Used in: ChargeDiffusion, Vibronic, Property
    """

    _template_name = 'occupations_driver.py.jinja'

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        # Additional inputs
        spec.input('mps', valid_type=MpsData, help='MPS state')
        spec.input('dof_type', valid_type=orm.Str, default=lambda: orm.Str('all'),
                   help='Which DOFs to compute: "all", "electronic", "phonon"')

        # Outputs
        # output_parameters will contain:
        # {
        #   'dof_names': [...],
        #   'occupations': [...],
        # }

    def _write_input_files(self, folder) -> None:
        """Write initial_mps.npz and dof_type.json."""
        import json
        super()._write_input_files(folder)

        # Write MPS
        mps_data = self.inputs.mps
        model_data = self.inputs.model
        mps = mps_data.load_mps(model_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            mps_path = os.path.join(tmpdir, 'mps')
            mps.dump(mps_path)
            actual = mps_path + '.npz' if os.path.exists(mps_path + '.npz') else mps_path
            with open(actual, 'rb') as src:
                with folder.open('initial_mps.npz', 'wb') as dst:
                    dst.write(src.read())

        # Write dof_type
        with folder.open('input_dof_type.json', 'w') as f:
            json.dump({'dof_type': self.inputs.dof_type.value}, f)
