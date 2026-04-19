"""Integration test for all L2 Composite CalcJobs."""
from __future__ import annotations

import pytest
from aiida import orm


class TestCompositeCalcJobIntegration:
    """Integration tests for L2 Composite CalcJobs."""

    def test_all_calcjobs_importable(self):
        """All L2 Composite CalcJobs should be importable."""
        from aiida_renormalizer.calculations.composite import (
            DMRGCalcJob,
            ImagTimeCalcJob,
            TDVPCalcJob,
            ThermalPropCalcJob,
            PropertyCalcJob,
        )

        assert DMRGCalcJob is not None
        assert ImagTimeCalcJob is not None
        assert TDVPCalcJob is not None
        assert ThermalPropCalcJob is not None
        assert PropertyCalcJob is not None

    def test_all_calcjobs_registered(self):
        """All L2 Composite CalcJobs should be registered via entry points."""
        from aiida.plugins import CalculationFactory

        # Test loading via entry points
        dmrg = CalculationFactory("reno.dmrg")
        imag_time = CalculationFactory("reno.imag_time")
        tdvp = CalculationFactory("reno.tdvp")
        thermal_prop = CalculationFactory("reno.thermal_prop")
        prop = CalculationFactory("reno.property")

        assert dmrg.__name__ == "DMRGCalcJob"
        assert imag_time.__name__ == "ImagTimeCalcJob"
        assert tdvp.__name__ == "TDVPCalcJob"
        assert thermal_prop.__name__ == "ThermalPropCalcJob"
        assert prop.__name__ == "PropertyCalcJob"

    def test_all_templates_exist(self):
        """All template files should exist."""
        from pathlib import Path
        import aiida_renormalizer

        pkg_dir = Path(aiida_renormalizer.__file__).parent
        template_dir = pkg_dir / "templates"

        templates = [
            "dmrg_driver.py.jinja",
            "imag_time_driver.py.jinja",
            "tdvp_driver.py.jinja",
            "thermal_prop_driver.py.jinja",
            "property_driver.py.jinja",
        ]

        for tmpl in templates:
            path = template_dir / tmpl
            assert path.exists(), f"Template not found: {tmpl}"

    def test_all_templates_renderable(self):
        """All templates should render without errors."""
        from jinja2 import Environment, FileSystemLoader
        from pathlib import Path
        import aiida_renormalizer

        pkg_dir = Path(aiida_renormalizer.__file__).parent
        template_dir = pkg_dir / "templates"

        env = Environment(loader=FileSystemLoader(str(template_dir)))

        context = {
            "calcjob_class": "TestCalcJob",
            "has_initial_mps": False,
            "has_omega": False,
            "has_dt": False,
            "has_observables": False,
            "n_iterations": 10,
        }

        templates = [
            "dmrg_driver.py.jinja",
            "imag_time_driver.py.jinja",
            "tdvp_driver.py.jinja",
            "thermal_prop_driver.py.jinja",
            "property_driver.py.jinja",
        ]

        for tmpl_name in templates:
            template = env.get_template(tmpl_name)
            rendered = template.render(**context)
            assert len(rendered) > 0, f"Template rendered empty: {tmpl_name}"
            assert "import" in rendered, f"Template missing imports: {tmpl_name}"
