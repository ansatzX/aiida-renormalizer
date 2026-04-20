"""Base CalcJob class for all Renormalizer calculations."""
from __future__ import annotations

import typing as t

from aiida import orm
from aiida.common.datastructures import CalcInfo, CodeInfo
from aiida.engine import CalcJob, CalcJobProcessSpec

from aiida_renormalizer.data import ConfigData, ModelData


class RenoBaseCalcJob(CalcJob):
    """Abstract base class for all Reno CalcJobs (L1 and L2).

    Provides:
    - Common input ports (model, config, code)
    - Common output ports (output_parameters)
    - Environment control (disable MKL/OMP threading)
    - Template-based driver.py generation
    """

    _template_name: str  # Must be overridden by subclasses

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        """Define inputs/outputs common to all Reno CalcJobs."""
        super().define(spec)

        # Common inputs
        spec.input(
            'model',
            valid_type=ModelData,
            required=False,
            help='Renormalizer Model (required for MPS/MPO-style jobs)',
        )
        spec.input('config', valid_type=ConfigData, required=False,
                   help='Configuration (EvolveConfig/OptimizeConfig/CompressConfig)')
        spec.input('code', valid_type=orm.Code, help='Code pointing to Python with renormalizer')
        spec.input(
            'metadata.options.artifact_storage_backend',
            valid_type=str,
            required=False,
            help='Artifact storage backend for parsed tensor artifacts (e.g. posix).',
        )
        spec.input(
            'metadata.options.artifact_storage_base',
            valid_type=str,
            required=False,
            help='Base path for parsed tensor artifacts.',
        )
        spec.input(
            'metadata.options.artifact_relative_path',
            valid_type=str,
            required=False,
            help='Relative path for parsed tensor artifacts.',
        )
        # Real CalcJobProcessSpec has the metadata parser option.
        # Some unit tests pass a lightweight Mock spec without this structure.
        try:
            spec.inputs['metadata']['options']['parser_name'].default = 'reno.base'
        except (TypeError, KeyError, AttributeError):
            pass
        # Common outputs
        spec.output('output_parameters', valid_type=orm.Dict,
                    help='Convergence info, energies, observables, etc.')

        # Exit codes (shared across all Reno CalcJobs, used by RenoBaseParser)
        spec.exit_code(100, 'ERROR_EXECUTION_FAILED', message='Remote execution failed')
        spec.exit_code(200, 'ERROR_OUTPUT_MISSING', message='Output files missing')
        spec.exit_code(201, 'ERROR_OUTPUT_PARSING', message='Failed to parse output files')
        spec.exit_code(300, 'ERROR_NOT_CONVERGED', message='Calculation did not converge')
        spec.exit_code(310, 'ERROR_PHYSICAL_VALIDATION', message='Physical validation failed')
        spec.exit_code(400, 'ERROR_OOM_WALLTIME', message='Out of memory or walltime exceeded')

    def prepare_for_submission(self, folder) -> CalcInfo:
        """Prepare the calculation for submission.

        This method:
        1. Writes driver.py from Jinja2 template
        2. Writes input files (model.json, config.json, initial_mps.npz, etc.)
        3. Sets environment variables (MKL/OMP thread control)
        4. Returns CalcInfo for AiiDA scheduler
        """
        if not getattr(self, '_template_name', None):
            raise NotImplementedError(
                f"{self.__class__.__name__} must define _template_name"
            )

        # 1. Generate driver.py from template
        driver_content = self._render_driver_template()
        with folder.open('driver.py', 'w') as f:
            f.write(driver_content)

        # 2. Write input files
        self._write_input_files(folder)

        # 3. Prepare code info
        code = self.inputs.code
        code_info = CodeInfo()
        code_info.code_uuid = code.uuid
        code_info.cmdline_params = ['driver.py']
        code_info.stdout_name = 'aiida.out'
        code_info.stderr_name = 'aiida.err'

        # 4. Environment control: disable threading
        # (Conflict with Reno's internal parallelization)
        default_prepend = '\n'.join([
            'export MKL_NUM_THREADS=1',
            'export OMP_NUM_THREADS=1',
            'export OPENBLAS_NUM_THREADS=1',
        ])
        user_prepend = self.inputs.metadata.options.get('prepend_text', '')

        # 5. Prepare calc info
        calc_info = CalcInfo()
        calc_info.codes_info = [code_info]
        calc_info.retrieve_list = self._get_retrieve_list()
        calc_info.prepend_text = f"{default_prepend}\n{user_prepend}"

        return calc_info

    def _get_retrieve_list(self) -> list[str]:
        """Get list of files to retrieve after calculation.

        Subclasses can override to add additional files (e.g., trajectory.npz).
        """
        return [
            'output_parameters.json',
            'output_mps.npz',
            'output_mpo.npz',
            'trajectory.npz',
            'aiida.out',
            'aiida.err',
        ]

    def _render_driver_template(self) -> str:
        """Render driver.py from Jinja2 template.

        Subclasses can override to provide custom context.
        """
        from jinja2 import Environment, FileSystemLoader
        import os

        template_dir = os.path.join(os.path.dirname(__file__), '..', 'templates')
        env = Environment(loader=FileSystemLoader(template_dir))
        template = env.get_template(self._template_name)

        context = self._get_template_context()
        return template.render(**context)

    def _get_template_context(self) -> dict:
        """Provide context for Jinja2 template rendering.

        Subclasses should override to add calc-specific variables.
        """
        return {
            'calcjob_class': self.__class__.__name__,
        }

    def _write_input_files(self, folder) -> None:
        """Write input files (model.json, config.json, initial_mps.npz, etc.).

        Subclasses should override to write calc-specific inputs.
        """
        import json
        from aiida_renormalizer.data.utils import read_json_from_repository

        # Write model.json when present. TTN jobs do not use ModelData.
        if 'model' in self.inputs:
            model_data = self.inputs.model
            model_dict = {
                'basis': read_json_from_repository(model_data, 'basis.json'),
                'ham_opsum': read_json_from_repository(model_data, 'ham_opsum.json'),
            }

            try:
                dipole_data = read_json_from_repository(model_data, 'dipole.json')
                model_dict['dipole'] = dipole_data
            except FileNotFoundError:
                pass

            with folder.open('input_model.json', 'w') as f:
                json.dump(model_dict, f, indent=2)

        # Write config.json (if provided)
        if 'config' in self.inputs:
            config_data = self.inputs.config
            config_dict = config_data.base.attributes.get('fields')
            with folder.open('input_config.json', 'w') as f:
                json.dump({
                    'config_class': config_data.base.attributes.get('config_class'),
                    'fields': config_dict,
                }, f, indent=2)

    @staticmethod
    def _sanitize_compress_config_payload(payload: dict[str, t.Any]) -> dict[str, t.Any]:
        """Strip unsupported upstream-unimplemented compress settings."""
        allowed_fields = {
            "criteria",
            "threshold",
            "max_bonddim",
            "vmethod",
            "vprocedure",
            "vrtol",
            "vguess_m",
            "dump_matrix_size",
            "dump_matrix_dir",
        }
        cleaned = dict(payload)
        fields = cleaned.get("fields")
        if isinstance(fields, dict):
            cleaned["fields"] = {k: v for k, v in fields.items() if k in allowed_fields}
        else:
            cleaned = {k: v for k, v in cleaned.items() if k in allowed_fields}
        return cleaned
