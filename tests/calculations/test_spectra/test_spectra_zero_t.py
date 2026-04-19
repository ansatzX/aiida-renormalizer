"""Tests for SpectraZeroTCalcJob."""
import pytest
from aiida import orm
from aiida.common import AttributeDict

from aiida_renormalizer.calculations.spectra.spectra_zero_t import SpectraZeroTCalcJob


def _make_calcjob(cls, inputs_dict):
    """Create a CalcJob instance without triggering plumpy Process.__init__."""
    from plumpy.utils import AttributesFrozendict
    calcjob = object.__new__(cls)
    calcjob._parsed_inputs = AttributesFrozendict(inputs_dict)
    return calcjob


@pytest.fixture
def generate_inputs(fixture_model_data, fixture_code):
    """Generate minimal inputs for SpectraZeroTCalcJob."""
    model = fixture_model_data
    code = fixture_code

    inputs = AttributeDict()
    inputs.model = model
    inputs.code = code
    inputs.spectratype = orm.Str("abs")
    inputs.propagation = orm.Str("two_way")
    inputs.metadata = AttributeDict()
    inputs.metadata.options = AttributeDict()
    inputs.metadata.options.resources = {"num_machines": 1, "num_mpiprocs_per_machine": 1}
    return inputs


def test_spectra_zero_t_definition(generate_inputs):
    """Test SpectraZeroTCalcJob process definition."""
    from aiida.engine import CalcJobProcessSpec

    # Create a spec and define it
    spec = CalcJobProcessSpec()
    SpectraZeroTCalcJob.define(spec)

    # Check that required ports are defined
    assert "model" in spec.inputs
    assert "code" in spec.inputs
    assert "spectratype" in spec.inputs


def test_spectra_zero_t_template_context(generate_inputs):
    """Test template context generation."""
    calc = _make_calcjob(SpectraZeroTCalcJob, {
        "model": generate_inputs.model,
        "spectratype": orm.Str("abs"),
        "propagation": orm.Str("two_way"),
    })

    context = calc._get_template_context()

    assert context["spectratype"] == "abs"
    assert context["propagation"] == "two_way"
    assert context["has_mpo"] == False
    assert context["has_initial_mps"] == False


def test_spectra_zero_t_write_inputs(generate_inputs, tmp_path):
    """Test writing input files."""
    calc = _make_calcjob(SpectraZeroTCalcJob, {
        "model": generate_inputs.model,
        "spectratype": orm.Str("abs"),
        "propagation": orm.Str("two_way"),
    })

    # Create a mock folder
    from aiida.common.folders import Folder

    folder = Folder(str(tmp_path))

    # Write input files
    calc._write_input_files(folder)

    # Check that required files are created
    assert (tmp_path / "input_model.json").exists()
    assert (tmp_path / "input_spectra_params.json").exists()


def test_spectra_zero_t_template_exists():
    """Test that the template file exists."""
    import os
    from aiida_renormalizer.calculations.spectra.spectra_zero_t import SpectraZeroTCalcJob

    template_dir = os.path.join(
        os.path.dirname(__file__),
        "..", "..", "..",
        "src", "aiida_renormalizer", "templates"
    )
    template_name = SpectraZeroTCalcJob._template_name
    template_path = os.path.join(template_dir, template_name)

    assert os.path.exists(template_path), f"Template not found: {template_path}"


def test_spectra_zero_t_entry_point():
    """Test that the CalcJob is registered as an entry point."""
    from aiida.plugins import CalculationFactory

    calc_class = CalculationFactory("reno.spectra_zero_t")
    assert calc_class is SpectraZeroTCalcJob
