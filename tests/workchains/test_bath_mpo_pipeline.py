"""Tests for BathMpoPipelineWorkChain."""
from aiida.engine import WorkChain

from aiida_renormalizer.workchains.bath_mpo_pipeline import BathMpoPipelineWorkChain


class TestBathMpoPipelineWorkChain:
    """Test cases for BathMpoPipelineWorkChain."""

    def test_workchain_class_exists(self):
        from aiida_renormalizer.workchains import BathMpoPipelineWorkChain as BMPWC
        assert BMPWC is BathMpoPipelineWorkChain

    def test_workchain_inherits_from_workchain(self):
        assert issubclass(BathMpoPipelineWorkChain, WorkChain)

    def test_workchain_has_exit_codes(self):
        exit_codes = BathMpoPipelineWorkChain.exit_codes
        assert hasattr(exit_codes, "ERROR_SPECTRAL_DENSITY_FAILED")
        assert hasattr(exit_codes, "ERROR_DISCRETIZATION_FAILED")
        assert hasattr(exit_codes, "ERROR_MAPPING_FAILED")
        assert hasattr(exit_codes, "ERROR_INPUT_VALIDATION")

    def test_exit_codes_have_correct_status(self):
        exit_codes = BathMpoPipelineWorkChain.exit_codes
        assert exit_codes.ERROR_SPECTRAL_DENSITY_FAILED.status == 470
        assert exit_codes.ERROR_DISCRETIZATION_FAILED.status == 471
        assert exit_codes.ERROR_MAPPING_FAILED.status == 472
        assert exit_codes.ERROR_INPUT_VALIDATION.status == 473

    def test_workchain_has_required_methods(self):
        required_methods = [
            "setup",
            "needs_spectral_density",
            "needs_discretization",
            "run_spectral_density",
            "inspect_spectral_density",
            "run_discretization",
            "inspect_discretization",
            "run_mapping",
            "inspect_mapping",
            "finalize",
        ]
        for method in required_methods:
            assert hasattr(BathMpoPipelineWorkChain, method), f"Missing method: {method}"

