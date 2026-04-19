"""CalcJob for computing expectation value."""
from __future__ import annotations

import os
import tempfile

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import ModelData, MpoData, MpsData


class ExpectationCalcJob(RenoBaseCalcJob):
    """Compute expectation value <mps|mpo|mps>.

    Corresponds to Reno API: mps.expectation(mpo)
    """

    _template_name = 'expectation_driver.py.jinja'

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        # Additional inputs
        spec.input('mps', valid_type=MpsData, help='MPS state')
        spec.input('mpo', valid_type=MpoData, help='MPO operator')

        # Outputs (output_parameters already defined in base)
        # expectation value will be in output_parameters

    def _write_input_files(self, folder) -> None:
        """Write initial_mps.npz and initial_mpo.npz."""
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

        # Write MPO
        mpo_data = self.inputs.mpo
        mpo = mpo_data.load_mpo(model_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            mpo_path = os.path.join(tmpdir, 'mpo')
            mpo.dump(mpo_path)
            actual = mpo_path + '.npz' if os.path.exists(mpo_path + '.npz') else mpo_path
            with open(actual, 'rb') as src:
                with folder.open('initial_mpo.npz', 'wb') as dst:
                    dst.write(src.read())
