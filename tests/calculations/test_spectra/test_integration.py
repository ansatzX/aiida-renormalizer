"""Integration tests for L2 Spectra and Transport CalcJobs."""
import os
import pytest

from aiida.plugins import CalculationFactory


def test_all_entry_points_registered():
    """Test that all L2 Spectra/Transport CalcJobs are registered as entry points."""
    expected_entry_points = [
        "reno.spectra_zero_t",
        "reno.spectra_finite_t",
        "reno.kubo",
        "reno.correction_vector",
        "reno.charge_diffusion",
        "reno.spectral_function",
    ]

    for entry_point in expected_entry_points:
        calc_class = CalculationFactory(entry_point)
        assert calc_class is not None, f"Failed to load {entry_point}"


def test_all_templates_exist():
    """Test that all template files exist."""
    template_dir = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..",
        "src", "aiida_renormalizer", "templates"
    )

    template_mapping = {
        "reno.spectra_zero_t": "spectra_zero_t_driver.py.jinja",
        "reno.spectra_finite_t": "spectra_finite_t_driver.py.jinja",
        "reno.kubo": "kubo_driver.py.jinja",
        "reno.correction_vector": "correction_vector_driver.py.jinja",
        "reno.charge_diffusion": "charge_diffusion_driver.py.jinja",
        "reno.spectral_function": "spectral_function_driver.py.jinja",
    }

    for entry_point, template_name in template_mapping.items():
        calc_class = CalculationFactory(entry_point)
        template_path = os.path.join(template_dir, template_name)

        assert hasattr(calc_class, '_template_name'), \
            f"{calc_class.__name__} missing _template_name"
        assert calc_class._template_name == template_name, \
            f"{calc_class.__name__} has wrong _template_name"
        assert os.path.exists(template_path), \
            f"Template not found: {template_path}"


def test_template_rendering():
    """Test that all templates can be rendered with minimal context."""
    from jinja2 import Environment, FileSystemLoader

    template_dir = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..",
        "src", "aiida_renormalizer", "templates"
    )

    env = Environment(loader=FileSystemLoader(template_dir))

    template_names = [
        "spectra_zero_t_driver.py.jinja",
        "spectra_finite_t_driver.py.jinja",
        "kubo_driver.py.jinja",
        "correction_vector_driver.py.jinja",
        "charge_diffusion_driver.py.jinja",
        "spectral_function_driver.py.jinja",
    ]

    # Minimal context for template rendering
    minimal_context = {
        'calcjob_class': 'TestCalcJob',
        'has_mpo': False,
        'has_initial_mps': False,
    }

    for template_name in template_names:
        template = env.get_template(template_name)
        # This should not raise an error
        content = template.render(**minimal_context)
        assert len(content) > 0, f"Template {template_name} produced empty output"

