"""Integration tests for bath-related CalcJobs."""
import os

from aiida.plugins import CalculationFactory


def test_bath_entry_points_registered():
    """Test bath CalcJob entry points."""
    expected_entry_points = [
        "reno.bath_spectral_density",
        "reno.bath_discretization",
        "reno.bath_to_mpo_coeff",
    ]
    for entry_point in expected_entry_points:
        calc_class = CalculationFactory(entry_point)
        assert calc_class is not None, f"Failed to load {entry_point}"


def test_bath_templates_exist():
    """Test bath template files exist and match class-level template names."""
    template_dir = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..",
        "src", "aiida_renormalizer", "templates",
    )

    template_mapping = {
        "reno.bath_spectral_density": "bath_spectral_density_driver.py.jinja",
        "reno.bath_discretization": "bath_discretization_driver.py.jinja",
        "reno.bath_to_mpo_coeff": "bath_to_mpo_coeff_driver.py.jinja",
    }

    for entry_point, template_name in template_mapping.items():
        calc_class = CalculationFactory(entry_point)
        template_path = os.path.join(template_dir, template_name)
        assert hasattr(calc_class, "_template_name")
        assert calc_class._template_name == template_name
        assert os.path.exists(template_path), f"Template not found: {template_path}"


def test_import_all_bath_calculations():
    """Test that all bath CalcJob classes can be imported."""
    from aiida_renormalizer.calculations.bath import (
        BathSpectralDensityCalcJob,
        BathDiscretizationCalcJob,
        BathToMpoCoeffCalcJob,
    )

    assert BathSpectralDensityCalcJob is not None
    assert BathDiscretizationCalcJob is not None
    assert BathToMpoCoeffCalcJob is not None

