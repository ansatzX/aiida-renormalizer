"""Base parser for all Reno CalcJobs."""
from __future__ import annotations

import json
import tempfile
import typing as t
from pathlib import Path

import numpy as np
from aiida import orm
from aiida.engine import ExitCode
from aiida.parsers import Parser

from aiida_renormalizer.data import BasisTreeData, ModelData, MPOData, MPSData, TTNSData


class RenoBaseParser(Parser):
    """Parser for Reno CalcJob outputs.

    Responsibilities:
    1. Parse output_parameters.json → orm.Dict
    2. Parse output_mps.npz → MPSData (if present)
    3. Parse output_mpo.npz → MPOData (if present)
    4. Physical validation (NaN, constraints, convergence)
    5. Map error conditions to exit codes
    """

    def parse(self, **kwargs) -> ExitCode | None:
        """Parse the CalcJob outputs.

        Returns:
            ExitCode if error, None if success.
        """
        # 1. Check for execution failure
        if self.node.exit_status is not None and self.node.exit_status != 0:
            return self.exit_codes.ERROR_EXECUTION_FAILED

        # 2. Parse output_parameters.json
        try:
            with self.retrieved.open('output_parameters.json', 'r') as f:
                params = json.load(f)
            output_params = orm.Dict(dict=params)
            self.out('output_parameters', output_params)
        except FileNotFoundError:
            return self.exit_codes.ERROR_OUTPUT_MISSING

        # 3. Parse output_mps.npz (if present)
        if 'output_mps.npz' in self.retrieved.list_object_names():
            try:
                if "model" not in self.node.inputs:
                    self.logger.error("Cannot parse output_mps.npz: model input required")
                    return self.exit_codes.ERROR_OUTPUT_PARSING
                model_data = self.node.inputs.model
                mps_data = self._parse_mps_file('output_mps.npz', model_data)
                self.out('output_mps', mps_data)
            except Exception as e:
                self.logger.error(f"Failed to parse output_mps.npz: {e}")
                return self.exit_codes.ERROR_OUTPUT_PARSING

        # 4. Parse output_mpo.npz (if present)
        if 'output_mpo.npz' in self.retrieved.list_object_names():
            try:
                if "model" not in self.node.inputs:
                    self.logger.error("Cannot parse output_mpo.npz: model input required")
                    return self.exit_codes.ERROR_OUTPUT_PARSING
                model_data = self.node.inputs.model
                mpo_data = self._parse_mpo_file('output_mpo.npz', model_data)
                self.out('output_mpo', mpo_data)
            except Exception as e:
                self.logger.error(f"Failed to parse output_mpo.npz: {e}")
                return self.exit_codes.ERROR_OUTPUT_PARSING

        # 5. Parse output_ttns.npz (if present)
        parsed_basis_tree_data = None
        if 'output_ttns.npz' in self.retrieved.list_object_names():
            try:
                if "basis_tree" in self.node.inputs:
                    basis_tree_data = self.node.inputs.basis_tree
                elif "output_basis_tree.npz" in self.retrieved.list_object_names():
                    basis_tree_data = self._parse_basis_tree_file("output_basis_tree.npz")
                    parsed_basis_tree_data = basis_tree_data
                elif "symbolic_inputs" in self.node.inputs:
                    basis_tree_data = self._basis_tree_from_symbolic_inputs()
                    parsed_basis_tree_data = basis_tree_data
                else:
                    self.logger.error(
                        "Cannot parse output_ttns.npz: basis_tree input or output_basis_tree.npz required"
                    )
                    return self.exit_codes.ERROR_OUTPUT_PARSING
                ttns_data = self._parse_ttns_file('output_ttns.npz', basis_tree_data)
                self.out('output_ttns', ttns_data)
            except Exception as e:
                self.logger.error(f"Failed to parse output_ttns.npz: {e}")
                return self.exit_codes.ERROR_OUTPUT_PARSING

        # Expose parsed basis tree when process spec supports it.
        if parsed_basis_tree_data is not None:
            try:
                if "output_basis_tree" in self.node.process_class.spec().outputs:
                    self.out("output_basis_tree", parsed_basis_tree_data)
            except Exception:
                pass

        # 6. Physical validation
        validation_result = self._validate_physical_constraints(params)
        if not validation_result['passed']:
            self.logger.error(f"Physical validation failed: {validation_result['reason']}")
            return self.exit_codes.ERROR_PHYSICAL_VALIDATION

        # 7. Check convergence
        if params.get('converged') is False:
            return self.exit_codes.ERROR_NOT_CONVERGED

        return None

    def _parse_mps_file(self, filename: str, model_data: ModelData) -> MPSData:
        """Parse an MPS .npz file into MPSData node."""
        with self.retrieved.open(filename, 'rb') as f:
            with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

        try:
            model = model_data.load_model()

            is_mpdm = getattr(self.node.inputs, 'is_mpdm', None)
            if is_mpdm is not None and is_mpdm.value:
                from renormalizer.mps import MpDm
                MPS = MpDm.load(model, tmp_path)
            else:
                from renormalizer.mps import Mps
                MPS = Mps.load(model, tmp_path)

            storage_backend, storage_base, relative_path = self._get_artifact_location(filename)
            mps_data = MPSData.from_mps(
                MPS,
                model_data,
                storage_backend=storage_backend,
                storage_base=storage_base,
                relative_path=relative_path,
            )
            return mps_data
        finally:
            import os
            os.unlink(tmp_path)

    def _parse_mpo_file(self, filename: str, model_data: ModelData) -> MPOData:
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
                storage_backend=storage_backend,
                storage_base=storage_base,
                relative_path=relative_path,
            )
            return mpo_data
        finally:
            import os
            os.unlink(tmp_path)

    def _parse_ttns_file(self, filename: str, basis_tree_data: BasisTreeData) -> TTNSData:
        """Parse a TTNS .npz file into TTNSData node."""
        with self.retrieved.open(filename, 'rb') as f:
            with tempfile.NamedTemporaryFile(suffix='.npz', delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

        try:
            basis_tree = basis_tree_data.load_basis_tree()
            from renormalizer.tn.tree import TTNS

            ttns_loaded = TTNS.load(basis_tree, tmp_path)
            storage_backend, storage_base, relative_path = self._get_artifact_location(filename)
            return TTNSData.from_ttns(
                ttns_loaded,
                basis_tree_data,
                storage_backend=storage_backend,
                storage_base=storage_base,
                relative_path=relative_path,
            )
        finally:
            import os
            os.unlink(tmp_path)

    def _parse_basis_tree_file(self, filename: str) -> BasisTreeData:
        """Parse a BasisTree .npz file into BasisTreeData node."""
        with self.retrieved.open(filename, "rb") as f:
            with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as tmp:
                tmp.write(f.read())
                tmp_path = tmp.name

        try:
            from renormalizer.tn.treebase import BasisTree
            from aiida_renormalizer.data import BasisTreeData as _BasisTreeData

            basis_tree = BasisTree.load(tmp_path)
            return _BasisTreeData.from_basis_tree(basis_tree)
        finally:
            import os
            os.unlink(tmp_path)

    def _basis_tree_from_symbolic_inputs(self) -> BasisTreeData:
        """Build BasisTreeData from symbolic TTNS inputs."""
        symbolic = self.node.inputs.symbolic_inputs.get_dict()
        basis_spec = symbolic["basis"]

        from renormalizer.model import basis as ba
        from renormalizer.tn import BasisTree

        basis = []
        for item in basis_spec:
            kind = item["kind"]
            dof = item["dof"]
            if kind == "half_spin":
                basis.append(ba.BasisHalfSpin(dof, item.get("sigmaqn", [0, 0])))
            elif kind == "sho":
                basis.append(ba.BasisSHO(dof, omega=float(item["omega"]), nbas=int(item["nbas"])))
            else:
                raise ValueError(f"Unsupported basis kind in symbolic_inputs: {kind}")

        tree_type = symbolic.get("tree_type", "binary")
        if tree_type == "binary":
            basis_tree = BasisTree.binary(basis)
        elif tree_type == "linear":
            basis_tree = BasisTree.linear(basis)
        else:
            raise ValueError(f"Unsupported tree_type in symbolic_inputs: {tree_type}")

        return BasisTreeData.from_basis_tree(basis_tree)

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

    def _validate_physical_constraints(self, params: dict) -> dict:
        """Validate physical constraints from llm_reno Validator agent experience.

        Checks:
        - NaN/Inf in observables
        - Energy monotonicity (for ground state optimization)
        - Energy conservation (for real-time evolution)
        - Bond dimension collapse (bond_dim == 1 for all sites)
        - Constraint violations (e.g., |<σ_z>| > 1)

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

        # Check bond dimension collapse
        if 'bond_dims' in params:
            bond_dims = params['bond_dims']
            if all(d == 1 for d in bond_dims):
                return {'passed': False, 'reason': 'All bond dimensions collapsed to 1'}

        # Check spin constraint (if applicable)
        if 'sigma_z' in params:
            sigma_z = params['sigma_z']
            if abs(sigma_z) > 1.0 + 1e-6:
                return {'passed': False, 'reason': f'|<σ_z>| = {abs(sigma_z)} > 1'}

        # Check energy monotonicity (for optimization)
        if params.get('calc_type') == 'optimization':
            if 'energy_trajectory' in params:
                energies = params['energy_trajectory']
                for i in range(1, len(energies)):
                    if energies[i] > energies[i-1] + 1e-8:
                        return {
                            'passed': False,
                            'reason': f'Energy increased at step {i}: {energies[i-1]} → {energies[i]}'
                        }

        # Check energy conservation (for real-time evolution)
        if params.get('calc_type') == 'real_time_evolution':
            if 'energy_trajectory' in params:
                energies = params['energy_trajectory']
                initial_energy = energies[0]
                for i, e in enumerate(energies[1:], 1):
                    if abs(e - initial_energy) > 1e-4:
                        return {
                            'passed': False,
                            'reason': f'Energy drift at step {i}: {initial_energy} → {e}'
                        }

        return {'passed': True, 'reason': ''}
