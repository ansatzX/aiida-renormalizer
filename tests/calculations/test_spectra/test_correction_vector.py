"""Tests for CorrectionVectorCalcJob."""
import pytest
import numpy as np
from aiida import orm
from aiida.common import AttributeDict

from aiida_renormalizer.calculations.spectra.correction_vector import CorrectionVectorCalcJob


def _make_calcjob(cls, inputs_dict):
    """Create a CalcJob instance without triggering plumpy Process.__init__."""
    from plumpy.utils import AttributesFrozendict
    calcjob = object.__new__(cls)
    calcjob._parsed_inputs = AttributesFrozendict(inputs_dict)
    return calcjob


@pytest.fixture
def generate_inputs(fixture_model_data, fixture_code):
    """Generate minimal inputs for CorrectionVectorCalcJob."""
    model = fixture_model_data
    code = fixture_code

    # Create frequency array
    frequencies = orm.ArrayData()
    frequencies.set_array("frequencies", np.linspace(-5, 5, 11))

    inputs = AttributeDict()
    inputs.model = model
    inputs.code = code
    inputs.frequencies = frequencies
    inputs.eta = orm.Float(0.1)
    inputs.m_max = orm.Int(50)
    inputs.metadata = AttributeDict()
    inputs.metadata.options = AttributeDict()
    inputs.metadata.options.resources = {"num_machines": 1, "num_mpiprocs_per_machine": 1}
    return inputs


def test_correction_vector_definition(generate_inputs):
    """Test CorrectionVectorCalcJob process definition."""
    from aiida.engine import CalcJobProcessSpec

    # Create a spec and define it
    spec = CalcJobProcessSpec()
    CorrectionVectorCalcJob.define(spec)

    # Check that required ports are defined
    assert "model" in spec.inputs
    assert "code" in spec.inputs
    assert "frequencies" in spec.inputs
    assert "eta" in spec.inputs


def test_correction_vector_template_context(generate_inputs):
    """Test template context generation."""
    calc = _make_calcjob(CorrectionVectorCalcJob, {
        "model": generate_inputs.model,
        "frequencies": generate_inputs.frequencies,
        "eta": orm.Float(0.1),
        "m_max": orm.Int(50),
        "method": orm.Str("1site"),
        "rtol": orm.Float(1e-5),
        "n_cores": orm.Int(1),
    })

    context = calc._get_template_context()

    assert context["eta"] == 0.1
    assert context["m_max"] == 50
    assert context["method"] == "1site"
    assert context["n_cores"] == 1


def test_correction_vector_write_inputs(generate_inputs, tmp_path):
    """Test writing input files."""
    calc = _make_calcjob(CorrectionVectorCalcJob, {
        "model": generate_inputs.model,
        "frequencies": generate_inputs.frequencies,
        "eta": orm.Float(0.1),
        "m_max": orm.Int(50),
        "method": orm.Str("1site"),
        "rtol": orm.Float(1e-5),
        "n_cores": orm.Int(1),
    })

    from aiida.common.folders import Folder
    folder = Folder(str(tmp_path))

    calc._write_input_files(folder)

    assert (tmp_path / "input_model.json").exists()
    assert (tmp_path / "input_cv_params.json").exists()
    assert (tmp_path / "frequencies.npy").exists()


def test_correction_vector_entry_point():
    """Test that the CalcJob is registered as an entry point."""
    from aiida.plugins import CalculationFactory

    calc_class = CalculationFactory("reno.correction_vector")
    assert calc_class is CorrectionVectorCalcJob
