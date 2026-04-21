"""Tests for SpectralFunctionCalcJob."""
import pytest
from aiida import orm
from aiida.common import AttributeDict

from aiida_renormalizer.calculations.spectra.spectral_function import SpectralFunctionCalcJob


def _make_calcjob(cls, inputs_dict):
    """Create a CalcJob instance without triggering plumpy Process.__init__."""
    from plumpy.utils import AttributesFrozendict
    calcjob = object.__new__(cls)
    calcjob._parsed_inputs = AttributesFrozendict(inputs_dict)
    return calcjob


@pytest.fixture
def generate_inputs(fixture_model_data, fixture_code):
    """Generate minimal inputs for SpectralFunctionCalcJob."""
    model = fixture_model_data
    code = fixture_code

    inputs = AttributeDict()
    inputs.model = model
    inputs.code = code
    inputs.metadata = AttributeDict()
    inputs.metadata.options = AttributeDict()
    inputs.metadata.options.resources = {"num_machines": 1, "num_mpiprocs_per_machine": 1}
    return inputs


def test_spectral_function_definition(generate_inputs):
    """Test SpectralFunctionCalcJob process definition."""
    from aiida.engine import CalcJobProcessSpec

    # Create a spec and define it
    spec = CalcJobProcessSpec()
    SpectralFunctionCalcJob.define(spec)

    # Check that required ports are defined
    assert "model" in spec.inputs
    assert "code" in spec.inputs


def test_spectral_function_template_context(generate_inputs):
    """Test template context generation."""
    calc = _make_calcjob(SpectralFunctionCalcJob, {
        "model": generate_inputs.model,
    })

    context = calc._get_template_context()

    assert context["has_mpo"] == False
    assert context["has_initial_mps"] == False


def test_spectral_function_retrieve_list(generate_inputs):
    """Test that retrieve list doesn't include output_mps.npz."""
    calc = _make_calcjob(SpectralFunctionCalcJob, {})

    retrieve_list = calc._get_retrieve_list()

    # Spectral function doesn't output MPS
    assert 'output_mps.npz' not in retrieve_list
    assert 'output_parameters.json' in retrieve_list

