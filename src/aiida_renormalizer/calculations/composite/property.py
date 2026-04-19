"""CalcJob for multi-observable property scanning."""
from __future__ import annotations

import os
import tempfile

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import ModelData, MpsData, MpoData, OpData


class PropertyCalcJob(RenoBaseCalcJob):
    """Compute multiple observables from an MPS.

    Corresponds to Reno API: mps.expectation(mpo) for multiple operators

    Inputs:
        model: ModelData - System definition
        mps: MpsData - Quantum state
        observables: Dict[str, Union[MpoData, OpData]] - Named observables

    Outputs:
        output_parameters: Dict - All observable values
    """

    _template_name = "property_driver.py.jinja"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        # Additional inputs
        spec.input(
            "mps",
            valid_type=MpsData,
            help="MPS state to measure",
        )
        spec.input_namespace(
            "observables",
            dynamic=True,
            help="Named observables (MPO or Op)",
        )

        # Outputs
        # output_parameters will contain all observable values

    def _write_input_files(self, folder) -> None:
        """Write input files for property calculation."""
        import json

        super()._write_input_files(folder)

        # Write MPS
        mps_data = self.inputs.mps
        model_data = self.inputs.model
        mps = mps_data.load_mps(model_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            mps_path = os.path.join(tmpdir, "mps")
            mps.dump(mps_path)
            actual = mps_path + ".npz" if os.path.exists(mps_path + ".npz") else mps_path
            with open(actual, "rb") as src:
                with folder.open("initial_mps.npz", "wb") as dst:
                    dst.write(src.read())

        # Write observables manifest
        observables_manifest = {}
        for name, obs_data in self.inputs.observables.items():
            if isinstance(obs_data, MpoData):
                # Write MPO file
                mpo = obs_data.load_mpo(model_data)
                with tempfile.TemporaryDirectory() as tmpdir:
                    mpo_path = os.path.join(tmpdir, f"obs_{name}")
                    mpo.dump(mpo_path)
                    actual = mpo_path + ".npz" if os.path.exists(mpo_path + ".npz") else mpo_path
                    with open(actual, "rb") as src:
                        with folder.open(f"observable_{name}.npz", "wb") as dst:
                            dst.write(src.read())

                observables_manifest[name] = {
                    "type": "mpo",
                    "file": f"observable_{name}.npz",
                }

            elif isinstance(obs_data, OpData):
                # Write Op JSON
                with obs_data.base.repository.open("op.json", "rb") as src:
                    with folder.open(f"observable_{name}.json", "wb") as dst:
                        dst.write(src.read())

                observables_manifest[name] = {
                    "type": "op",
                    "file": f"observable_{name}.json",
                }

        with folder.open("observables_manifest.json", "w") as f:
            json.dump(observables_manifest, f, indent=2)
