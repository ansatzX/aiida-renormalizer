"""Tests for ChargeDiffusionCalcJob."""
import pytest
from aiida import orm
from aiida.common import AttributeDict

from aiida_renormalizer.calculations.spectra.charge_diffusion import ChargeDiffusionCalcJob


def _make_calcjob(cls, inputs_dict):
    """Create a CalcJob instance without triggering plumpy Process.__init__."""
    from plumpy.utils import AttributesFrozendict
    calcjob = object.__new__(cls)
    calcjob._parsed_inputs = AttributesFrozendict(inputs_dict)
    return calcjob


@pytest.fixture
def generate_inputs(fixture_model_data, fixture_code):
    """Generate minimal inputs for ChargeDiffusionCalcJob."""
    model = fixture_model_data
    code = fixture_code

    inputs = AttributeDict()
    inputs.model = model
    inputs.code = code
    inputs.temperature = orm.Float(0.0)
    inputs.init_electron = orm.Str("relaxed")
    inputs.stop_at_edge = orm.Bool(True)
    inputs.rdm = orm.Bool(False)
    inputs.metadata = AttributeDict()
    inputs.metadata.options = AttributeDict()
    inputs.metadata.options.resources = {"num_machines": 1, "num_mpiprocs_per_machine": 1}
    return inputs


def test_charge_diffusion_definition(generate_inputs):
    """Test ChargeDiffusionCalcJob process definition."""
    from aiida.engine import CalcJobProcessSpec

    # Create a spec and define it
    spec = CalcJobProcessSpec()
    ChargeDiffusionCalcJob.define(spec)

    # Check that required ports are defined
    assert "model" in spec.inputs
    assert "code" in spec.inputs


def test_charge_diffusion_template_context(generate_inputs):
    """Test template context generation."""
    calc = _make_calcjob(ChargeDiffusionCalcJob, {
        "model": generate_inputs.model,
        "temperature": orm.Float(0.0),
        "init_electron": orm.Str("relaxed"),
        "stop_at_edge": orm.Bool(True),
        "rdm": orm.Bool(False),
    })

    context = calc._get_template_context()

    assert context["temperature"] == 0.0
    assert context["init_electron"] == "relaxed"
    assert context["stop_at_edge"] == True
    assert context["rdm"] == False


def test_charge_diffusion_retrieve_list(generate_inputs):
    """Test that retrieve list doesn't include output_mps.npz."""
    calc = _make_calcjob(ChargeDiffusionCalcJob, {})

    retrieve_list = calc._get_retrieve_list()

    # Charge diffusion doesn't output MPS
    assert 'output_mps.npz' not in retrieve_list
    assert 'output_parameters.json' in retrieve_list


def test_charge_diffusion_entry_point():
    """Test that the CalcJob is registered as an entry point."""
    from aiida.plugins import CalculationFactory

    calc_class = CalculationFactory("reno.charge_diffusion")
    assert calc_class is ChargeDiffusionCalcJob
