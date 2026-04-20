"""Tests for BathSpectralDensityCalcJob."""
import pytest
from aiida import orm
from aiida.common import AttributeDict

from aiida_renormalizer.calculations.bath.bath_spectral_density import BathSpectralDensityCalcJob


def _make_calcjob(cls, inputs_dict):
    """Create a CalcJob instance without triggering plumpy Process.__init__."""
    from plumpy.utils import AttributesFrozendict
    calcjob = object.__new__(cls)
    calcjob._parsed_inputs = AttributesFrozendict(inputs_dict)
    return calcjob


@pytest.fixture
def generate_inputs(fixture_model_data, fixture_code):
    """Generate minimal inputs for BathSpectralDensityCalcJob."""
    inputs = AttributeDict()
    inputs.model = fixture_model_data
    inputs.code = fixture_code
    inputs.spectral_density_type = orm.Str("ohmic_exp")
    inputs.omega_min = orm.Float(0.0)
    inputs.omega_max = orm.Float(5.0)
    inputs.num_points = orm.Int(256)
    inputs.alpha = orm.Float(1.0)
    inputs.s_exponent = orm.Float(1.0)
    inputs.cutoff = orm.Float(1.0)
    inputs.lambda_reorg = orm.Float(1.0)
    inputs.metadata = AttributeDict()
    inputs.metadata.options = AttributeDict()
    inputs.metadata.options.resources = {"num_machines": 1, "num_mpiprocs_per_machine": 1}
    return inputs


def test_bath_spectral_density_definition():
    """Test BathSpectralDensityCalcJob process definition."""
    from aiida.engine import CalcJobProcessSpec

    spec = CalcJobProcessSpec()
    BathSpectralDensityCalcJob.define(spec)

    assert "model" in spec.inputs
    assert "code" in spec.inputs
    assert "spectral_density_type" in spec.inputs
    assert "output_parameters" in spec.outputs


def test_bath_spectral_density_template_context(generate_inputs):
    """Test template context generation."""
    calc = _make_calcjob(BathSpectralDensityCalcJob, {
        "spectral_density_type": orm.Str("ohmic_exp"),
        "omega_min": orm.Float(0.0),
        "omega_max": orm.Float(5.0),
        "num_points": orm.Int(256),
        "alpha": orm.Float(1.0),
        "s_exponent": orm.Float(1.0),
        "cutoff": orm.Float(1.0),
        "lambda_reorg": orm.Float(1.0),
        "beta": orm.Float(0.7),
    })

    context = calc._get_template_context()

    assert context["spectral_density_type"] == "ohmic_exp"
    assert context["num_points"] == 256
    assert context["has_custom_spectrum"] is False


def test_bath_spectral_density_retrieve_list():
    """Test that retrieve list doesn't include output_mps.npz."""
    calc = _make_calcjob(BathSpectralDensityCalcJob, {})

    retrieve_list = calc._get_retrieve_list()

    assert "output_mps.npz" not in retrieve_list
    assert "output_parameters.json" in retrieve_list


def test_bath_spectral_density_entry_point():
    """Test that the CalcJob is registered as an entry point."""
    from aiida.plugins import CalculationFactory

    calc_class = CalculationFactory("reno.bath_spectral_density")
    assert calc_class is BathSpectralDensityCalcJob
