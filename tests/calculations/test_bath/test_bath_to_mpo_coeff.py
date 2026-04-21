"""Tests for BathToMPOCoeffCalcJob."""
import numpy as np
import pytest
from aiida import orm
from aiida.common import AttributeDict

from aiida_renormalizer.calculations.bath.bath_to_mpo_coeff import BathToMPOCoeffCalcJob


def _make_calcjob(cls, inputs_dict):
    """Create a CalcJob instance without triggering plumpy Process.__init__."""
    from plumpy.utils import AttributesFrozendict
    calcjob = object.__new__(cls)
    calcjob._parsed_inputs = AttributesFrozendict(inputs_dict)
    return calcjob


@pytest.fixture
def omega_k():
    arr = orm.ArrayData()
    arr.set_array("omega_k", np.array([0.2, 0.5, 1.0, 2.0], dtype=float))
    return arr


@pytest.fixture
def c_j2():
    arr = orm.ArrayData()
    arr.set_array("c_j2", np.array([0.01, 0.02, 0.03, 0.01], dtype=float))
    return arr


@pytest.fixture
def generate_inputs(fixture_model_data, fixture_code, omega_k, c_j2):
    """Generate minimal inputs for BathToMPOCoeffCalcJob."""
    inputs = AttributeDict()
    inputs.model = fixture_model_data
    inputs.code = fixture_code
    inputs.omega_k = omega_k
    inputs.c_j2 = c_j2
    inputs.frequency_scale = orm.Float(1.0)
    inputs.coupling_scale = orm.Float(1.0)
    inputs.metadata = AttributeDict()
    inputs.metadata.options = AttributeDict()
    inputs.metadata.options.resources = {"num_machines": 1, "num_mpiprocs_per_machine": 1}
    return inputs


def test_bath_to_mpo_coeff_definition():
    """Test BathToMPOCoeffCalcJob process definition."""
    from aiida.engine import CalcJobProcessSpec

    spec = CalcJobProcessSpec()
    BathToMPOCoeffCalcJob.define(spec)

    assert "model" in spec.inputs
    assert "code" in spec.inputs
    assert "omega_k" in spec.inputs
    assert "c_j2" in spec.inputs
    assert "output_parameters" in spec.outputs


def test_bath_to_mpo_coeff_template_context():
    """Test template context generation."""
    calc = _make_calcjob(BathToMPOCoeffCalcJob, {
        "frequency_scale": orm.Float(1.0),
        "coupling_scale": orm.Float(1.0),
    })
    context = calc._get_template_context()
    assert context["frequency_scale"] == 1.0
    assert context["coupling_scale"] == 1.0


def test_bath_to_mpo_coeff_retrieve_list():
    """Test that retrieve list doesn't include output_mps.npz."""
    calc = _make_calcjob(BathToMPOCoeffCalcJob, {})
    retrieve_list = calc._get_retrieve_list()
    assert "output_mps.npz" not in retrieve_list
    assert "output_parameters.json" in retrieve_list


def test_bath_to_mpo_coeff_entry_point():
    """Test that the CalcJob is registered as an entry point."""
    from aiida.plugins import CalculationFactory

    calc_class = CalculationFactory("reno.bath_to_mpo_coeff")
    assert calc_class is BathToMPOCoeffCalcJob

