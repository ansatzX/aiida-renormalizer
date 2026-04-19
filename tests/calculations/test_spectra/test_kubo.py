"""Tests for KuboCalcJob."""
import pytest
import numpy as np
from aiida import orm
from aiida.common import AttributeDict

from aiida_renormalizer.calculations.spectra.kubo import KuboCalcJob


def _make_calcjob(cls, inputs_dict):
    """Create a CalcJob instance without triggering plumpy Process.__init__."""
    from plumpy.utils import AttributesFrozendict
    calcjob = object.__new__(cls)
    calcjob._parsed_inputs = AttributesFrozendict(inputs_dict)
    return calcjob


@pytest.fixture
def generate_inputs(fixture_model_data, fixture_code):
    """Generate minimal inputs for KuboCalcJob."""
    model = fixture_model_data
    code = fixture_code

    inputs = AttributeDict()
    inputs.model = model
    inputs.code = code
    inputs.temperature = orm.Float(300.0)
    inputs.insteps = orm.Int(1)
    inputs.metadata = AttributeDict()
    inputs.metadata.options = AttributeDict()
    inputs.metadata.options.resources = {"num_machines": 1, "num_mpiprocs_per_machine": 1}
    return inputs


def test_kubo_definition(generate_inputs):
    """Test KuboCalcJob process definition."""
    from aiida.engine import CalcJobProcessSpec

    # Create a spec and define it
    spec = CalcJobProcessSpec()
    KuboCalcJob.define(spec)

    # Check that required ports are defined
    assert "model" in spec.inputs
    assert "code" in spec.inputs
    assert "temperature" in spec.inputs


def test_kubo_template_context(generate_inputs):
    """Test template context generation."""
    calc = _make_calcjob(KuboCalcJob, {
        "model": generate_inputs.model,
        "temperature": orm.Float(300.0),
        "insteps": orm.Int(1),
    })

    context = calc._get_template_context()

    assert context["temperature"] == 300.0
    assert context["insteps"] == 1
    assert context["has_distance_matrix"] == False


def test_kubo_with_distance_matrix(generate_inputs, tmp_path):
    """Test KuboCalcJob with distance matrix input."""
    # Create a simple distance matrix
    dist_matrix = orm.ArrayData()
    dist_matrix.set_array("matrix", np.array([[0, 1], [-1, 0]]))

    calc = _make_calcjob(KuboCalcJob, {
        "model": generate_inputs.model,
        "temperature": orm.Float(300.0),
        "insteps": orm.Int(1),
        "distance_matrix": dist_matrix,
    })

    context = calc._get_template_context()

    assert context["has_distance_matrix"] == True


def test_kubo_entry_point():
    """Test that the CalcJob is registered as an entry point."""
    from aiida.plugins import CalculationFactory

    calc_class = CalculationFactory("reno.kubo")
    assert calc_class is KuboCalcJob
