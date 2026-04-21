"""CalcJob for constructing MPO from symbolic operator."""
from __future__ import annotations

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import ModelData, MPOData, OpData


class BuildMPOCalcJob(RenoBaseCalcJob):
    """Build an MPO from a symbolic operator (Op or OpSum).

    Corresponds to Reno API: Mpo(model, op) from OpData
    """

    _template_name = 'build_mpo_driver.py.jinja'

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        # Additional inputs
        spec.input('op', valid_type=OpData, help='Symbolic operator to convert to MPO')

        # Outputs
        spec.output('output_mpo', valid_type=MPOData, help='Constructed MPO')

    def _write_input_files(self, folder) -> None:
        """Write op.json for driver.py."""
        import json
        super()._write_input_files(folder)

        op_data = self.inputs.op
        with op_data.base.repository.open('op.json', 'rb') as src:
            with folder.open('input_op.json', 'wb') as dst:
                dst.write(src.read())
