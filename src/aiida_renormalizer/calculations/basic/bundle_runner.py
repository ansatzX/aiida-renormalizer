"""CalcJob for single-submit sequential stage bundle execution."""
from __future__ import annotations

import json

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob


class BundleRunnerCalcJob(RenoBaseCalcJob):
    """Execute a stage manifest sequentially inside one scheduler allocation."""

    _template_name = "bundle_runner_driver.py.jinja"

    @classmethod
    def _validate_manifest(cls, value: orm.Dict, _) -> str | None:
        payload = value.get_dict()
        stages = payload.get("stages")
        if not isinstance(stages, list) or not stages:
            return "manifest.stages must be a non-empty list"
        seen: set[str] = set()
        for index, stage in enumerate(stages):
            if not isinstance(stage, dict):
                return f"manifest.stages[{index}] must be a dict"
            name = stage.get("name")
            script = stage.get("script")
            if not isinstance(name, str) or not name.strip():
                return f"manifest.stages[{index}].name must be a non-empty string"
            if name in seen:
                return f"duplicate stage name: {name}"
            seen.add(name)
            if not isinstance(script, str) or not script.strip():
                return f"manifest.stages[{index}].script must be a non-empty string"
        return None

    @classmethod
    def _validate_resume_from_stage(cls, value: orm.Int, _) -> str | None:
        if value.value < 1:
            return "resume_from_stage must be >= 1"
        return None

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        spec.input(
            "manifest",
            valid_type=orm.Dict,
            validator=cls._validate_manifest,
            help="Bundle stage manifest with ordered stage scripts",
        )
        spec.input(
            "resume_from_stage",
            valid_type=orm.Int,
            default=lambda: orm.Int(1),
            validator=cls._validate_resume_from_stage,
            help="1-based stage index to start/resume from",
        )
        spec.input(
            "fail_fast",
            valid_type=orm.Bool,
            default=lambda: orm.Bool(True),
            help="Fail immediately on first stage error",
        )

        spec.output(
            "output_parameters",
            valid_type=orm.Dict,
            help="Bundle execution summary and stage statuses",
        )

    def _write_input_files(self, folder) -> None:
        with folder.open("input_manifest.json", "w") as handle:
            json.dump(self.inputs.manifest.get_dict(), handle, indent=2)

        with folder.open("input_bundle_control.json", "w") as handle:
            json.dump(
                {
                    "resume_from_stage": self.inputs.resume_from_stage.value,
                    "fail_fast": bool(self.inputs.fail_fast.value),
                },
                handle,
                indent=2,
            )

    def _get_retrieve_list(self) -> list[str]:
        return [
            "output_parameters.json",
            "bundle_state.json",
            "stage_summary.json",
            "bundle.log",
            "aiida.out",
            "aiida.err",
        ]
