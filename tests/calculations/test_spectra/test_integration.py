"""Integration tests for L2 Spectra and Transport CalcJobs."""
import os

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
