"""Tests for SpectraFiniteTCalcJob."""
import pytest
from aiida import orm
from aiida.common import AttributeDict

from aiida_renormalizer.calculations.spectra.spectra_finite_t import SpectraFiniteTCalcJob


def _make_calcjob(cls, inputs_dict):
    """Create a CalcJob instance without triggering plumpy Process.__init__."""
    from plumpy.utils import AttributesFrozendict
    calcjob = object.__new__(cls)
    calcjob._parsed_inputs = AttributesFrozendict(inputs_dict)
    return calcjob


@pytest.fixture
def generate_inputs(fixture_model_data, fixture_code):
    """Generate minimal inputs for SpectraFiniteTCalcJob."""
    model = fixture_model_data
    code = fixture_code

    inputs = AttributeDict()
    inputs.model = model
    inputs.code = code
    inputs.temperature = orm.Float(300.0)  # Temperature in K (will be converted)
    inputs.spectratype = orm.Str("abs")
    inputs.insteps = orm.Int(10)
    inputs.metadata = AttributeDict()
    inputs.metadata.options = AttributeDict()
    inputs.metadata.options.resources = {"num_machines": 1, "num_mpiprocs_per_machine": 1}
    return inputs


def test_spectra_finite_t_definition(generate_inputs):
    """Test SpectraFiniteTCalcJob process definition."""
    from aiida.engine import CalcJobProcessSpec

    # Create a spec and define it
    spec = CalcJobProcessSpec()
    SpectraFiniteTCalcJob.define(spec)

    # Check that required ports are defined
    assert "model" in spec.inputs
    assert "code" in spec.inputs
    assert "temperature" in spec.inputs
    assert "spectratype" in spec.inputs


def test_spectra_finite_t_template_context(generate_inputs):
    """Test template context generation."""
    calc = _make_calcjob(SpectraFiniteTCalcJob, {
        "model": generate_inputs.model,
        "temperature": orm.Float(300.0),
        "spectratype": orm.Str("abs"),
        "insteps": orm.Int(10),
    })

    context = calc._get_template_context()

    assert context["temperature"] == 300.0
    assert context["spectratype"] == "abs"
    assert context["insteps"] == 10
    assert context["has_mpo"] == False
    assert context["has_initial_mps"] == False


def test_spectra_finite_t_write_inputs(generate_inputs, tmp_path):
    """Test writing input files."""
    calc = _make_calcjob(SpectraFiniteTCalcJob, {
        "model": generate_inputs.model,
        "temperature": orm.Float(300.0),
        "spectratype": orm.Str("abs"),
        "insteps": orm.Int(10),
    })

    from aiida.common.folders import Folder
    folder = Folder(str(tmp_path))

    calc._write_input_files(folder)

    assert (tmp_path / "input_model.json").exists()
    assert (tmp_path / "input_spectra_params.json").exists()

