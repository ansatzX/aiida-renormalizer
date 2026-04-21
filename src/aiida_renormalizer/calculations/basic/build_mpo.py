"""CalcJob for constructing MPO from symbolic operator."""
from __future__ import annotations

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import ModelData, MPOData, OpData


class BuildMPOCalcJob(RenoBaseCalcJob):
    """Build an MPO from ModelData Hamiltonian or explicit symbolic operator.

    Corresponds to Reno API: `Mpo(model, op)` where `op` defaults to model.ham_terms
    when no explicit OpData is provided.
    """

    _template_name = 'build_mpo_driver.py.jinja'

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        # Optional operator override; default is model.ham_terms
        spec.input('op', valid_type=OpData, required=False, help='Optional symbolic operator override')

        # Outputs
        spec.output('output_mpo', valid_type=MPOData, help='Constructed MPO')

    def _write_input_files(self, folder) -> None:
        """Write op.json for driver.py."""
        super()._write_input_files(folder)

        if 'op' in self.inputs:
            op_data = self.inputs.op
            with op_data.base.repository.open('op.json', 'rb') as src:
                with folder.open('input_op.json', 'wb') as dst:
                    dst.write(src.read())
