"""CalcJob for building maximally entangled MpDm initial state."""
from __future__ import annotations

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import MPSData


class MaxEntangledMpdmCalcJob(RenoBaseCalcJob):
    """Build an infinite-temperature MpDm from model and space declaration."""

    _template_name = "max_entangled_mpdm_driver.py.jinja"

    @classmethod
    def _validate_space(cls, value: orm.Str, _) -> str | None:
        if value.value not in {"GS", "EX"}:
            return "space must be 'GS' or 'EX'."
        return None

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        spec.input(
            "space",
            valid_type=orm.Str,
            default=lambda: orm.Str("GS"),
            validator=cls._validate_space,
            help="Max-entangled state space selector: GS or EX",
        )

        spec.output(
            "output_mps",
            valid_type=MPSData,
            help="Maximally entangled MpDm initial state",
        )

        spec.exit_code(
            347,
            "ERROR_MAX_ENTANGLED_BUILD_FAILED",
            message="Max-entangled MpDm state construction failed",
        )
