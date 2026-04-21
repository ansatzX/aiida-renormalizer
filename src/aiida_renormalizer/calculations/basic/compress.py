"""CalcJob for MPS/MpDm compression."""
from __future__ import annotations

import os
import tempfile

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import ModelData, MPSData


class CompressCalcJob(RenoBaseCalcJob):
    """Compress MPS/MpDm to reduce bond dimension.

    Corresponds to Reno API: mps.compress(config)
    """

    _template_name = 'compress_driver.py.jinja'

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        # Additional inputs
        spec.input('mps', valid_type=MPSData, help='MPS/MpDm state to compress')

        # config input is optional (inherited from base)
        # If not provided, will use default CompressConfig

        # Outputs
        spec.output('output_mps', valid_type=MPSData, help='Compressed MPS/MpDm')

    def _write_input_files(self, folder) -> None:
        """Write initial_mps.npz."""
        super()._write_input_files(folder)

        mps_data = self.inputs.mps
        model_data = self.inputs.model
        MPS = mps_data.load_mps(model_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            mps_path = os.path.join(tmpdir, 'mps')
            MPS.dump(mps_path)
            actual = mps_path + '.npz' if os.path.exists(mps_path + '.npz') else mps_path
            with open(actual, 'rb') as src:
                with folder.open('initial_mps.npz', 'wb') as dst:
                    dst.write(src.read())
