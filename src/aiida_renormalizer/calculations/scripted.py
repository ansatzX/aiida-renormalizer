"""L3 ScriptedCalcJob - user-defined Python scripts with AiiDA data access."""
from __future__ import annotations

import typing as t

from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import ModelData, MpsData, MpoData, OpData, ConfigData


class RenoScriptCalcJob(RenoBaseCalcJob):
    """L3 ScriptedCalcJob for user-defined Python scripts.

    Allows users to write custom Python scripts that:
    - Access AiiDA data nodes (ModelData, MpsData, MpoData, etc.) as Python objects
    - Use Renormalizer library directly
    - Return outputs that are automatically converted to AiiDA nodes

    Inputs:
        script: orm.Str - Python script content (or SinglefileData)
        model: ModelData (optional) - System definition
        mps: MpsData (optional) - MPS state
        mpo: MpoData (optional) - MPO operator
        op: OpData (optional) - Operator
        config: ConfigData (optional) - Configuration
        inputs: Dict (optional) - Additional scalar parameters

    Outputs:
        output_parameters: orm.Dict - Always present
        output_mps: MpsData (optional) - If script creates MPS
        output_mpo: MpoData (optional) - If script creates MPO
        output_data: orm.Dict (optional) - Additional structured outputs

    Example script:
        ```python
        # User script has access to:
        # - model, mps, mpo, op, config (if provided as inputs)
        # - inputs dict for scalar parameters
        # - Renormalizer library (already imported)

        # Perform custom calculation
        result = mps.expectation(mpo)
        energies = []

        for i in range(inputs['n_steps']):
            mps.evolve(mpo, inputs['dt'])
            e = mps.expectation(mpo)
            energies.append(e)

        # Save outputs (auto-detected by parser)
        save_output_parameters({'energies': energies, 'final_energy': energies[-1]})
        save_mps(mps, 'output_mps.npz', 'Evolved state')
        ```
    """

    _template_name = "scripted_driver.py.jinja"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        """Define inputs/outputs for scripted CalcJob."""
        super().define(spec)

        # Script input (required)
        spec.input(
            "script",
            valid_type=orm.Str,
            help="Python script content to execute",
        )

        # Data inputs (all optional, available in script context)
        spec.input(
            "model",
            valid_type=ModelData,
            required=False,
            help="Model for script operations",
        )
        spec.input(
            "mps",
            valid_type=MpsData,
            required=False,
            help="MPS state for script operations",
        )
        spec.input(
            "mpo",
            valid_type=MpoData,
            required=False,
            help="MPO operator for script operations",
        )
        spec.input(
            "op",
            valid_type=OpData,
            required=False,
            help="Operator for script operations",
        )
        spec.input(
            "config",
            valid_type=ConfigData,
            required=False,
            help="Configuration for script operations",
        )
        spec.input(
            "inputs",
            valid_type=orm.Dict,
            required=False,
            help="Additional scalar parameters for script",
        )

        # Outputs (flexible, detected by parser)
        spec.output(
            "output_parameters",
            valid_type=orm.Dict,
            help="Script output parameters",
        )
        spec.output(
            "output_mps",
            valid_type=MpsData,
            required=False,
            help="Output MPS if created by script",
        )
        spec.output(
            "output_mpo",
            valid_type=MpoData,
            required=False,
            help="Output MPO if created by script",
        )
        spec.output(
            "output_data",
            valid_type=orm.Dict,
            required=False,
            help="Additional structured data from script",
        )

        # Exit codes
        spec.exit_code(
            500,
            "ERROR_SCRIPT_EXECUTION",
            message="Script execution failed",
        )
        spec.exit_code(
            501,
            "ERROR_INVALID_OUTPUT",
            message="Script produced invalid output format",
        )

    def _get_template_context(self) -> dict:
        """Provide context for Jinja2 template rendering."""
        context = super()._get_template_context()
        context["has_model"] = "model" in self.inputs
        context["has_mps"] = "mps" in self.inputs
        context["has_mpo"] = "mpo" in self.inputs
        context["has_op"] = "op" in self.inputs
        context["has_config"] = "config" in self.inputs
        context["has_inputs"] = "inputs" in self.inputs

        # Inject user script
        context["user_script"] = self.inputs.script.value

        return context

    def _write_input_files(self, folder) -> None:
        """Write input files for scripted calculation."""
        import json
        import tempfile
        import os

        # Write model (if provided)
        if "model" in self.inputs:
            super()._write_input_files(folder)

        # Write MPS (if provided)
        if "mps" in self.inputs:
            mps_data = self.inputs.mps
            model_data = self.inputs.model if "model" in self.inputs else None
            if model_data:
                mps = mps_data.load_mps(model_data)
                with tempfile.TemporaryDirectory() as tmpdir:
                    mps_path = os.path.join(tmpdir, "mps")
                    mps.dump(mps_path)
                    actual = mps_path + ".npz" if os.path.exists(mps_path + ".npz") else mps_path
                    with open(actual, "rb") as src:
                        with folder.open("input_mps.npz", "wb") as dst:
                            dst.write(src.read())

        # Write MPO (if provided)
        if "mpo" in self.inputs:
            mpo_data = self.inputs.mpo
            model_data = self.inputs.model if "model" in self.inputs else None
            if model_data:
                mpo = mpo_data.load_mpo(model_data)
                with tempfile.TemporaryDirectory() as tmpdir:
                    mpo_path = os.path.join(tmpdir, "mpo")
                    mpo.dump(mpo_path)
                    actual = mpo_path + ".npz" if os.path.exists(mpo_path + ".npz") else mpo_path
                    with open(actual, "rb") as src:
                        with folder.open("input_mpo.npz", "wb") as dst:
                            dst.write(src.read())

        # Write Op (if provided)
        if "op" in self.inputs:
            op_data = self.inputs.op
            op_dict = {
                "terms": op_data.base.attributes.get("terms"),
            }
            with folder.open("input_op.json", "w") as f:
                json.dump(op_dict, f, indent=2)

        # Write config (if provided)
        if "config" in self.inputs:
            config_data = self.inputs.config
            config_dict = config_data.base.attributes.get("fields")
            with folder.open("input_config.json", "w") as f:
                json.dump({
                    "config_class": config_data.base.attributes.get("config_class"),
                    "fields": config_dict,
                }, f, indent=2)

        # Write inputs dict (if provided)
        if "inputs" in self.inputs:
            with folder.open("input_parameters.json", "w") as f:
                json.dump(self.inputs.inputs.get_dict(), f, indent=2)

    def _get_retrieve_list(self) -> list[str]:
        """Get list of files to retrieve after calculation."""
        return [
            "output_parameters.json",
            "output_mps.npz",
            "output_mpo.npz",
            "output_data.json",
            "trajectory.npz",
            "aiida.out",
            "aiida.err",
        ]
