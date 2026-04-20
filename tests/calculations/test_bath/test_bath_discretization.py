"""Tests for BathDiscretizationCalcJob."""
import numpy as np
import pytest
from aiida import orm
from aiida.common import AttributeDict

from aiida_renormalizer.calculations.bath.bath_discretization import BathDiscretizationCalcJob


def _make_calcjob(cls, inputs_dict):
    """Create a CalcJob instance without triggering plumpy Process.__init__."""
    from plumpy.utils import AttributesFrozendict
    calcjob = object.__new__(cls)
    calcjob._parsed_inputs = AttributesFrozendict(inputs_dict)
    return calcjob


@pytest.fixture
def omega_grid():
    arr = orm.ArrayData()
    arr.set_array("omega_grid", np.linspace(0.0, 5.0, 200))
    return arr


@pytest.fixture
def j_omega():
    omega = np.linspace(0.0, 5.0, 200)
    arr = orm.ArrayData()
    arr.set_array("j_omega", omega * np.exp(-omega))
    return arr


@pytest.fixture
def generate_inputs(fixture_model_data, fixture_code, omega_grid, j_omega):
    """Generate minimal inputs for BathDiscretizationCalcJob."""
    inputs = AttributeDict()
    inputs.model = fixture_model_data
    inputs.code = fixture_code
    inputs.omega_grid = omega_grid
    inputs.j_omega = j_omega
    inputs.n_modes = orm.Int(16)
    inputs.method = orm.Str("trapz")
    inputs.metadata = AttributeDict()
    inputs.metadata.options = AttributeDict()
    inputs.metadata.options.resources = {"num_machines": 1, "num_mpiprocs_per_machine": 1}
    return inputs


def test_bath_discretization_definition():
    """Test BathDiscretizationCalcJob process definition."""
    from aiida.engine import CalcJobProcessSpec

    spec = CalcJobProcessSpec()
    BathDiscretizationCalcJob.define(spec)

    assert "model" in spec.inputs
    assert "code" in spec.inputs
    assert "omega_grid" in spec.inputs
    assert "j_omega" in spec.inputs
    assert "output_parameters" in spec.outputs


def test_bath_discretization_template_context():
    """Test template context generation."""
    calc = _make_calcjob(BathDiscretizationCalcJob, {
        "n_modes": orm.Int(16),
        "method": orm.Str("trapz"),
    })
    context = calc._get_template_context()
    assert context["n_modes"] == 16
    assert context["method"] == "trapz"


def test_bath_discretization_retrieve_list():
    """Test that retrieve list doesn't include output_mps.npz."""
    calc = _make_calcjob(BathDiscretizationCalcJob, {})

    retrieve_list = calc._get_retrieve_list()

    assert "output_mps.npz" not in retrieve_list
    assert "output_parameters.json" in retrieve_list


def test_bath_discretization_entry_point():
    """Test that the CalcJob is registered as an entry point."""
    from aiida.plugins import CalculationFactory

    calc_class = CalculationFactory("reno.bath_discretization")
    assert calc_class is BathDiscretizationCalcJob
