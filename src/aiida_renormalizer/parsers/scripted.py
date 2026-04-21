"""Parser for ScriptedCalcJob with automatic output type conversion."""
from __future__ import annotations

import json
import tempfile
import typing as t
from pathlib import Path

import numpy as np
from aiida import orm
from aiida.engine import ExitCode
from aiida.parsers import Parser

from aiida_renormalizer.data import ModelData, MPSData, MPOData, TensorNetworkLayoutData


class ScriptedParser(Parser):
    """Parser for ScriptedCalcJob outputs with automatic type conversion.

    Responsibilities:
    1. Parse output_parameters.json → orm.Dict (required)
    2. Parse output_mps.npz → MPSData (if present, requires model input)
    3. Parse output_mpo.npz → MPOData (if present, requires model input)
    4. Parse output_data.json → orm.Dict (if present)
    5. Physical validation (NaN checks)
    6. Map error conditions to exit codes
    """

    def parse(self, **kwargs) -> ExitCode | None:
        """Parse the ScriptedCalcJob outputs.

        Returns:
            ExitCode if error, None if success.
        """
        # 1. Check for execution failure
        if self.node.exit_status == 100:
            return self.exit_codes.ERROR_EXECUTION_FAILED

        # 2. Parse output_parameters.json (required)
        try:
            with self.retrieved.open('output_parameters.json', 'r') as f:
                params = json.load(f)
            output_params = orm.Dict(dict=params)
            self.out('output_parameters', output_params)
        except FileNotFoundError:
            return self.exit_codes.ERROR_OUTPUT_MISSING

        # 3. Parse output_mps.npz (if present)
        chain_layout_data = None
        if 'output_mps.npz' in self.retrieved.list_object_names():
            try:
                # Model is required to parse MPS
                if 'model' not in self.inputs:
                    self.logger.error("Cannot parse output_mps.npz: model input required")
                    return self.exit_codes.ERROR_INVALID_OUTPUT

                model_data = self.inputs.model
                chain_layout_data = self._resolve_chain_layout(model_data)
                mps_data = self._parse_mps_file('output_mps.npz', model_data, chain_layout_data)
                self.out('output_mps', mps_data)
            except Exception as e:
                self.logger.error(f"Failed to parse output_mps.npz: {e}")
                return self.exit_codes.ERROR_OUTPUT_PARSING

        # 4. Parse output_mpo.npz (if present)
        if 'output_mpo.npz' in self.retrieved.list_object_names():
            try:
                # Model is required to parse MPO
                if 'model' not in self.inputs:
                    self.logger.error("Cannot parse output_mpo.npz: model input required")
                    return self.exit_codes.ERROR_INVALID_OUTPUT

                model_data = self.inputs.model
                if chain_layout_data is None:
                    chain_layout_data = self._resolve_chain_layout(model_data)
                mpo_data = self._parse_mpo_file('output_mpo.npz', model_data, chain_layout_data)
                self.out('output_mpo', mpo_data)
            except Exception as e:
                self.logger.error(f"Failed to parse output_mpo.npz: {e}")
                return self.exit_codes.ERROR_OUTPUT_PARSING

        # 5. Parse output_data.json (if present)
        if 'output_data.json' in self.retrieved.list_object_names():
            try:
                with self.retrieved.open('output_data.json', 'r') as f:
                    data = json.load(f)
                output_data = orm.Dict(dict=data)
                self.out('output_data', output_data)
            except Exception as e:
                self.logger.error(f"Failed to parse output_data.json: {e}")
                return self.exit_codes.ERROR_OUTPUT_PARSING

        if chain_layout_data is not None:
            try:
                if "output_tn_layout" in self.node.process_class.spec().outputs:
                    self.out("output_tn_layout", chain_layout_data)
            except Exception:
                pass

        # 6. Physical validation
        validation_result = self._validate_physical_constraints(params)
        if not validation_result['passed']:
            self.logger.error(f"Physical validation failed: {validation_result['reason']}")
            return self.exit_codes.ERROR_PHYSICAL_VALIDATION

        # 7. Check for script-reported errors
        if params.get('error'):
            self.logger.error(f"Script reported error: {params['error']}")
            return self.exit_codes.ERROR_SCRIPT_EXECUTION

        return ExitCode()

    def _parse_mps_file(
        self,
        filename: str,
        model_data: ModelData,
        tn_layout_data: TensorNetworkLayoutData | None = None,
    ) -> MPSData:
        """Parse an MPS .npz file into MPSData node."""
        with self.retrieved.open(filename, 'rb') as f:
            with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

        try:
            model = model_data.load_model()

            # Detect if it's MpDm or Mps from the file
            with np.load(tmp_path, allow_pickle=True) as data:
                is_mpdm = 'dm' in str(data.get('description', ''))

            if is_mpdm:
                from renormalizer.mps import MpDm
                MPS = MpDm.load(model, tmp_path)
            else:
                from renormalizer.mps import Mps
                MPS = Mps.load(model, tmp_path)

            storage_backend, storage_base, relative_path = self._get_artifact_location(filename)
            mps_data = MPSData.from_mps(
                MPS,
                model_data,
                tn_layout_data,
                storage_backend=storage_backend,
                storage_base=storage_base,
                relative_path=relative_path,
            )
            return mps_data
        finally:
            import os
            os.unlink(tmp_path)

    def _parse_mpo_file(
        self,
        filename: str,
        model_data: ModelData,
        tn_layout_data: TensorNetworkLayoutData | None = None,
    ) -> MPOData:
        """Parse an MPO .npz file into MPOData node."""
        import tempfile

        with self.retrieved.open(filename, 'rb') as f:
            with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

        try:
            model = model_data.load_model()
            from renormalizer.mps import Mpo
            mpo = Mpo.load(model, tmp_path)

            storage_backend, storage_base, relative_path = self._get_artifact_location(filename)
            mpo_data = MPOData.from_mpo(
                mpo,
                model_data,
                tn_layout_data,
                storage_backend=storage_backend,
                storage_base=storage_base,
                relative_path=relative_path,
            )
            return mpo_data
        finally:
            import os
            os.unlink(tmp_path)

    def _get_artifact_location(self, filename: str) -> tuple[str, str, str]:
        """Return storage backend/base/path for parsed wavefunction artifacts."""
        storage_backend = self.node.get_option('artifact_storage_backend') or 'posix'
        storage_base = self.node.get_option('artifact_storage_base') or str(
            Path(tempfile.gettempdir()) / 'aiida-renormalizer-artifacts'
        )
        relative_path = self.node.get_option('artifact_relative_path') or (
            f"parsed/{getattr(self.node, 'uuid', 'unstored')}/{filename}"
        )
        return storage_backend, storage_base, relative_path

    def _resolve_chain_layout(self, model_data: ModelData) -> TensorNetworkLayoutData:
        """Reuse provided chain layout, otherwise create from model dof order."""
        if "tn_layout" in self.node.inputs:
            return self.node.inputs.tn_layout
        dof_order = model_data.base.attributes.get("dof_list") or []
        if not dof_order:
            model = model_data.load_model()
            dof_order = [str(dof) for dof in model.dofs]
        return TensorNetworkLayoutData.from_chain([str(dof) for dof in dof_order])

    def _validate_physical_constraints(self, params: dict) -> dict:
        """Validate physical constraints.

        Checks:
        - NaN/Inf in observables

        Returns:
            {'passed': bool, 'reason': str}
        """
        # Check for NaN/Inf
        for key, value in params.items():
            if isinstance(value, (int, float)):
                if np.isnan(value):
                    return {'passed': False, 'reason': f'{key} contains NaN'}
                if np.isinf(value):
                    return {'passed': False, 'reason': f'{key} contains Inf'}

        return {'passed': True, 'reason': ''}

    @classmethod
    def exit_codes(cls) -> t.Any:
        """Define exit codes for ScriptedCalcJob."""
        from aiida.engine import ExitCodesNamespace

        return ExitCodesNamespace({
            'ERROR_EXECUTION_FAILED': ExitCode(100, 'Remote execution failed'),
            'ERROR_OUTPUT_MISSING': ExitCode(200, 'Output files missing'),
            'ERROR_OUTPUT_PARSING': ExitCode(201, 'Failed to parse output files'),
            'ERROR_PHYSICAL_VALIDATION': ExitCode(310, 'Physical validation failed'),
            'ERROR_SCRIPT_EXECUTION': ExitCode(500, 'Script execution failed'),
            'ERROR_INVALID_OUTPUT': ExitCode(501, 'Invalid output format'),
        })
