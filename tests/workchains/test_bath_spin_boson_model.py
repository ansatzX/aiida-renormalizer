"""Tests for BathSpinBosonModelWorkChain."""
from aiida.engine import WorkChain

from aiida_renormalizer.workchains.bath_spin_boson_model import BathSpinBosonModelWorkChain


class TestBathSpinBosonModelWorkChain:
    def test_workchain_class_exists(self):
        from aiida_renormalizer.workchains import BathSpinBosonModelWorkChain as BSMWC

        assert BSMWC is BathSpinBosonModelWorkChain

    def test_workchain_inherits_from_workchain(self):
        assert issubclass(BathSpinBosonModelWorkChain, WorkChain)

    def test_workchain_has_expected_outputs(self):
        spec = BathSpinBosonModelWorkChain.spec()
        assert "bath_model" in spec.outputs
        assert "bath_modes" in spec.outputs
        assert "output_parameters" in spec.outputs

    def test_workchain_has_required_methods(self):
        required_methods = [
            "setup",
            "build_model",
            "finalize",
        ]
        for method in required_methods:
            assert hasattr(BathSpinBosonModelWorkChain, method), f"Missing method: {method}"

    def test_entry_point(self):
        from aiida.plugins import WorkflowFactory

        wc_class = WorkflowFactory("reno.bath_spin_boson_model")
        assert wc_class is BathSpinBosonModelWorkChain

