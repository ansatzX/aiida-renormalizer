"""Integration tests for ScriptedCalcJob and ScriptedParser."""
import pytest
from aiida import orm

from aiida_renormalizer.calculations.scripted import RenoScriptCalcJob
from aiida_renormalizer.parsers.scripted import ScriptedParser
from aiida_renormalizer.data import ModelData, MpsData, MpoData


@pytest.fixture
def simple_model():
    """Create a simple SHO chain model using conftest fixtures."""
    # Use existing fixture pattern
    from renormalizer.model.basis import BasisSHO
    from renormalizer.model import Model
    from renormalizer.model import OpSum
    from renormalizer.model.op import Op

    basis = [BasisSHO("v0", omega=1.0, nbas=4), BasisSHO("v1", omega=1.5, nbas=4)]
    ham_terms = OpSum([Op("b^\\dagger b", "v0", 1.0), Op("b^\\dagger b", "v1", 1.5)])

    model = Model(basis, ham_terms)
    model_data = ModelData.from_model(model)
    return model_data


@pytest.fixture
def simple_mps(simple_model, artifact_storage_base):
    """Create a simple MPS state."""
    from renormalizer.mps import Mps

    model = simple_model.load_model()
    mps = Mps.random(model, qntot=0, m_max=10)
    mps_data = MpsData.from_mps(
        mps,
        simple_model,
        storage_backend="posix",
        storage_base=str(artifact_storage_base),
        relative_path="scripted/simple_mps.npz",
    )
    return mps_data


@pytest.fixture
def simple_mpo(simple_model):
    """Create a simple MPO."""
    from renormalizer.mps import Mpo

    model = simple_model.load_model()
    mpo = Mpo(model)
    mpo_data = MpoData.from_mpo(mpo, simple_model)
    return mpo_data


def test_entry_points():
    """Test that entry points are properly registered."""
    from aiida.plugins.entry_point import get_entry_point_names

    # Check calculation entry point
    calc_names = get_entry_point_names('aiida.calculations')
    assert 'reno.script' in calc_names

    # Check parser entry point
    parser_names = get_entry_point_names('aiida.parsers')
    assert 'reno.scripted' in parser_names


def test_calcjob_import():
    """Test that RenoScriptCalcJob can be imported."""
    from aiida_renormalizer.calculations.scripted import RenoScriptCalcJob

    assert RenoScriptCalcJob is not None
    assert hasattr(RenoScriptCalcJob, 'define')
    assert hasattr(RenoScriptCalcJob, '_template_name')


def test_parser_import():
    """Test that ScriptedParser can be imported."""
    from aiida_renormalizer.parsers.scripted import ScriptedParser

    assert ScriptedParser is not None
    assert hasattr(ScriptedParser, 'parse')
    assert hasattr(ScriptedParser, '_parse_mps_file')
    assert hasattr(ScriptedParser, '_parse_mpo_file')


def test_template_exists():
    """Test that the Jinja2 template exists."""
    import os
    from aiida_renormalizer.calculations.scripted import RenoScriptCalcJob

    template_dir = os.path.join(
        os.path.dirname(__file__),
        '..',
        'src',
        'aiida_renormalizer',
        'templates'
    )
    template_file = os.path.join(template_dir, RenoScriptCalcJob._template_name)

    assert os.path.exists(template_file), f"Template not found: {template_file}"


def test_template_rendering(simple_model, simple_mps):
    """Test that the template can be rendered."""
    from jinja2 import Environment, FileSystemLoader
    import os
    from aiida_renormalizer.calculations.scripted import RenoScriptCalcJob

    # Mock a CalcJob instance (without MPO to avoid NumPy 2.0 issues)
    class MockCalcJob:
        _template_name = "scripted_driver.py.jinja"

        def __init__(self, model, mps):
            self.inputs = {
                'model': model,
                'mps': mps,
                'script': orm.Str("# Test script\nenergy = 1.0\nsave_output_parameters({'energy': energy})"),
                'metadata': {
                    'options': {
                        'artifact_storage_backend': 'posix',
                        'artifact_storage_base': '/tmp/aiida-renormalizer-artifacts',
                    }
                }
            }

        def _get_template_context(self):
            return {
                'calcjob_class': 'RenoScriptCalcJob',
                'artifact_storage_backend': 'posix',
                'artifact_storage_base': '/tmp/aiida-renormalizer-artifacts',
                'has_model': 'model' in self.inputs,
                'has_mps': 'mps' in self.inputs,
                'has_mpo': False,
                'has_op': False,
                'has_config': False,
                'has_inputs': False,
                'user_script': self.inputs['script'].value,
            }

    mock_job = MockCalcJob(simple_model, simple_mps)

    template_dir = os.path.join(
        os.path.dirname(__file__),
        '..',
        'src',
        'aiida_renormalizer',
        'templates'
    )
    env = Environment(loader=FileSystemLoader(template_dir))
    template = env.get_template(mock_job._template_name)

    context = mock_job._get_template_context()
    driver_content = template.render(**context)

    # Check that driver was rendered
    assert driver_content is not None
    assert len(driver_content) > 0

    # Check that user script is included
    assert "energy = 1.0" in driver_content

    # Check that context variables are used
    assert "model = load_model()" in driver_content
    assert "mps = load_mps" in driver_content
    # MPO not loaded when has_mpo is False
    assert "mpo = None" in driver_content


def test_scripted_calcjob_builder_structure():
    """Test that RenoScriptCalcJob builder has correct structure."""
    # Test that the builder can be accessed (doesn't require actual code)
    builder_class = RenoScriptCalcJob.get_builder()

    # Validate builder class exists
    assert builder_class is not None

    # Test that spec has correct inputs/outputs
    from aiida.engine import CalcJobProcessSpec
    spec = CalcJobProcessSpec()
    RenoScriptCalcJob.define(spec)

    # Check all expected inputs are defined
    assert 'script' in spec.inputs
    assert 'model' in spec.inputs
    assert 'mps' in spec.inputs
    assert 'mpo' in spec.inputs
    assert 'op' in spec.inputs
    assert 'config' in spec.inputs
    assert 'inputs' in spec.inputs

    # Check all expected outputs are defined
    assert 'output_parameters' in spec.outputs
    assert 'output_mps' in spec.outputs
    assert 'output_mpo' in spec.outputs
    assert 'output_data' in spec.outputs


def test_parser_exit_codes():
    """Test that parser has all required exit codes."""
    from aiida_renormalizer.parsers.scripted import ScriptedParser

    exit_codes = ScriptedParser.exit_codes()

    # Check all exit codes exist
    assert exit_codes.ERROR_EXECUTION_FAILED.status == 100
    assert exit_codes.ERROR_OUTPUT_MISSING.status == 200
    assert exit_codes.ERROR_OUTPUT_PARSING.status == 201
    assert exit_codes.ERROR_PHYSICAL_VALIDATION.status == 310
    assert exit_codes.ERROR_SCRIPT_EXECUTION.status == 500
    assert exit_codes.ERROR_INVALID_OUTPUT.status == 501


def test_calcjob_exit_codes():
    """Test that CalcJob has all required exit codes."""
    from aiida_renormalizer.calculations.scripted import RenoScriptCalcJob
    from aiida.engine import CalcJobProcessSpec

    spec = CalcJobProcessSpec()
    RenoScriptCalcJob.define(spec)

    # Check exit codes exist
    exit_codes = spec.exit_codes
    assert 'ERROR_SCRIPT_EXECUTION' in exit_codes
    assert 'ERROR_INVALID_OUTPUT' in exit_codes


def test_calcjob_inputs_outputs():
    """Test that CalcJob has correct inputs and outputs defined."""
    from aiida_renormalizer.calculations.scripted import RenoScriptCalcJob
    from aiida.engine import CalcJobProcessSpec

    spec = CalcJobProcessSpec()
    RenoScriptCalcJob.define(spec)

    # Check inputs
    assert 'script' in spec.inputs
    assert 'model' in spec.inputs
    assert 'mps' in spec.inputs
    assert 'mpo' in spec.inputs
    assert 'op' in spec.inputs
    assert 'config' in spec.inputs
    assert 'inputs' in spec.inputs

    # Check outputs
    assert 'output_parameters' in spec.outputs
    assert 'output_mps' in spec.outputs
    assert 'output_mpo' in spec.outputs
    assert 'output_data' in spec.outputs

    # Check required/optional
    assert spec.inputs['script'].required is True
    assert spec.inputs['model'].required is False
    assert spec.inputs['mps'].required is False
    assert spec.inputs['mpo'].required is False


def test_example_workflow_script():
    """Test an example workflow script structure."""
    # Example script for a multi-step workflow
    script = """
# Multi-step workflow example
# 1. Calculate initial energy
initial_energy = mps.expectation(mpo)

# 2. Evolve the system
from renormalizer.mps import Mps
results = []
for i in range(inputs['n_steps']):
    mps.evolve(mpo, inputs['dt'])
    e = mps.expectation(mpo)
    results.append(e)

# 3. Calculate final observables
final_energy = results[-1]

# 4. Save outputs
save_output_parameters({
    'initial_energy': initial_energy,
    'final_energy': final_energy,
    'energy_trajectory': results,
    'n_steps': len(results),
})

# 5. Save evolved MPS
save_mps(mps, 'output_mps.npz', f'Evolved state, E={final_energy:.6f}')
"""

    # Verify script structure
    assert 'expectation' in script
    assert 'evolve' in script
    assert 'save_output_parameters' in script
    assert 'save_mps' in script
    assert 'inputs[' in script
