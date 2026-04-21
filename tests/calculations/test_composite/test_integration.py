"""Integration test for all L2 Composite CalcJobs."""
from __future__ import annotations

class TestCompositeCalcJobIntegration:
    """Integration tests for L2 Composite CalcJobs."""

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
