"""CalcJob for TDVP real-time evolution."""
from __future__ import annotations

import os
import tempfile

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import ModelData, MpsData, MpoData


class TDVPCalcJob(RenoBaseCalcJob):
    """TDVP real-time evolution with trajectory output.

    Corresponds to Reno API: mps.evolve(mpo, dt) with EvolveConfig(method=TDVP)

    Inputs:
        model: ModelData - System definition
        mpo: MpoData - Hamiltonian operator
        initial_mps: MpsData - Initial state
        config: ConfigData - EvolveConfig with TDVP parameters
        total_time: Float - Total evolution time
        dt: Float - Time step
        trajectory_interval: Int (optional) - Save every N steps

    Outputs:
        output_mps: MpsData - Final state
        trajectory: ArrayData - MPS snapshots at intervals
        output_parameters: Dict - Energy trajectory, observables
    """

    _template_name = "tdvp_driver.py.jinja"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        # Additional inputs
        spec.input(
            "mpo",
            valid_type=MpoData,
            help="Hamiltonian MPO",
        )
        spec.input(
            "initial_mps",
            valid_type=MpsData,
            help="Initial MPS for time evolution",
        )
        spec.input(
            "total_time",
            valid_type=orm.Float,
            help="Total evolution time",
        )
        spec.input(
            "dt",
            valid_type=orm.Float,
            help="Time step",
        )
        spec.input(
            "trajectory_interval",
            valid_type=orm.Int,
            required=False,
            default=lambda: orm.Int(1),
            help="Save trajectory every N steps",
        )
        spec.input(
            "observables",
            valid_type=orm.List,
            required=False,
            help="List of MPO UUIDs to compute observables at each step",
        )

        # Outputs
        spec.output(
            "output_mps",
            valid_type=MpsData,
            help="Final MPS after evolution",
        )
        spec.output(
            "trajectory",
            valid_type=orm.ArrayData,
            required=False,
            help="MPS snapshots at trajectory intervals",
        )

        # Exit codes
        spec.exit_code(
            310,
            "ERROR_PHYSICAL_VALIDATION",
            message="Energy conservation violated",
        )

    def _get_template_context(self) -> dict:
        """Provide context for Jinja2 template rendering."""
        context = super()._get_template_context()
        context["has_observables"] = "observables" in self.inputs
        return context

    def _write_input_files(self, folder) -> None:
        """Write input files for TDVP evolution."""
        import json

        super()._write_input_files(folder)

        # Write MPO
        mpo_data = self.inputs.mpo
        model_data = self.inputs.model
        mpo = mpo_data.load_mpo(model_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            mpo_path = os.path.join(tmpdir, "mpo")
            mpo.dump(mpo_path)
            actual = mpo_path + ".npz" if os.path.exists(mpo_path + ".npz") else mpo_path
            with open(actual, "rb") as src:
                with folder.open("initial_mpo.npz", "wb") as dst:
                    dst.write(src.read())

        # Write initial MPS
        mps_data = self.inputs.initial_mps
        mps = mps_data.load_mps(model_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            mps_path = os.path.join(tmpdir, "mps")
            mps.dump(mps_path)
            actual = mps_path + ".npz" if os.path.exists(mps_path + ".npz") else mps_path
            with open(actual, "rb") as src:
                with folder.open("initial_mps.npz", "wb") as dst:
                    dst.write(src.read())

        # Write evolution parameters
        params = {
            "total_time": self.inputs.total_time.value,
            "dt": self.inputs.dt.value,
            "trajectory_interval": self.inputs.trajectory_interval.value,
        }

        with folder.open("input_evolution_params.json", "w") as f:
            json.dump(params, f, indent=2)

        # Write observable MPOs (if provided)
        if "observables" in self.inputs:
            observable_uuids = self.inputs.observables.get_list()
            with folder.open("input_observables.json", "w") as f:
                json.dump({"mpo_uuids": observable_uuids}, f)

            # Load and write each observable MPO
            for i, uuid in enumerate(observable_uuids):
                # Note: This would need to load the MpoData node by UUID
                # For now, this is a placeholder
                pass
