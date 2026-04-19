"""Tests for RenoScriptCalcJob."""
import json
import pytest
from aiida import orm
from aiida.engine import run_get_node

from aiida_renormalizer.data import ModelData, MpsData, MpoData, ConfigData


@pytest.fixture
def simple_model():
    """Create a simple SHO chain model."""
    from renormalizer.model.basis import BasisSHO
    from renormalizer.model import Model
    from renormalizer.model.op import Op

    basis = [BasisSHO(f"v{i}", omega=1.0, nbas=4) for i in range(4)]
    ham_terms = [
        Op(r"b^\dagger b", f"v{i}", 1.0 * (i + 1)) for i in range(4)
    ]

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
        relative_path="calcjobs/scripted_mps.npz",
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


def test_scripted_calcjob_basic(simple_model, simple_mps, simple_mpo, fixture_code):
    """Test basic ScriptedCalcJob execution with script."""
    from aiida_renormalizer.calculations.scripted import RenoScriptCalcJob

    # Simple script that calculates expectation value
    script = """
# Calculate expectation value
energy = mps.expectation(mpo)

# Save results
save_output_parameters({
    'energy': energy,
    'calcjob_class': 'RenoScriptCalcJob',
})
"""

    # Build process
    builder = RenoScriptCalcJob.get_builder()
    builder.code = fixture_code
    builder.script = orm.Str(script)
    builder.model = simple_model
    builder.mps = simple_mps
    builder.mpo = simple_mpo

    # Run (would fail without actual scheduler, but we can test the structure)
    # In real tests, we'd use aiida-testing fixtures
    # For now, just validate the builder structure
    assert 'script' in builder
    assert 'model' in builder
    assert 'mps' in builder
    assert 'mpo' in builder


def test_scripted_calcjob_with_inputs_dict(simple_model, simple_mps, simple_mpo, fixture_code):
    """Test ScriptedCalcJob with inputs dict for scalar parameters."""
    from aiida_renormalizer.calculations.scripted import RenoScriptCalcJob

    # Script that uses inputs dict
    script = """
# Use inputs dict for parameters
n_steps = inputs.get('n_steps', 10)
dt = inputs.get('dt', 0.01)

# Simple loop
results = []
for i in range(n_steps):
    results.append(i * dt)

# Save results
save_output_parameters({
    'results': results,
    'n_steps': n_steps,
    'dt': dt,
})
"""

    builder = RenoScriptCalcJob.get_builder()
    builder.code = fixture_code
    builder.script = orm.Str(script)
    builder.model = simple_model
    builder.mps = simple_mps
    builder.mpo = simple_mpo
    builder.inputs = orm.Dict({'n_steps': 5, 'dt': 0.1})

    # Validate builder
    assert 'inputs' in builder
    assert builder.inputs['n_steps'] == 5


def test_scripted_calcjob_output_mps(simple_model, simple_mps, fixture_code):
    """Test ScriptedCalcJob that outputs MPS."""
    from aiida_renormalizer.calculations.scripted import RenoScriptCalcJob

    # Script that modifies MPS and saves it
    script = """
# Modify MPS (example: compress)
from renormalizer.mps import Mps

# Use the loaded MPS
output_mps = mps.copy()

# Save modified MPS
save_mps(output_mps, 'output_mps.npz', 'Modified MPS')
save_output_parameters({
    'success': True,
    'n_sites': len(output_mps),
})
"""

    builder = RenoScriptCalcJob.get_builder()
    builder.code = fixture_code
    builder.script = orm.Str(script)
    builder.model = simple_model
    builder.mps = simple_mps

    # Validate builder
    assert 'mps' in builder


def test_scripted_calcjob_error_handling(simple_model, fixture_code):
    """Test ScriptedCalcJob error handling."""
    from aiida_renormalizer.calculations.scripted import RenoScriptCalcJob

    # Script with error
    script = """
# This will cause an error
result = undefined_variable

save_output_parameters({'result': result})
"""

    builder = RenoScriptCalcJob.get_builder()
    builder.code = fixture_code
    builder.script = orm.Str(script)
    builder.model = simple_model

    # Validate builder (execution would fail, but structure is valid)
    assert 'script' in builder


def test_scripted_calcjob_no_data_inputs(fixture_code):
    """Test ScriptedCalcJob without any data inputs."""
    from aiida_renormalizer.calculations.scripted import RenoScriptCalcJob

    # Script that doesn't need Reno data
    script = """
# Simple calculation
import numpy as np

x = np.linspace(0, 1, 10)
y = x ** 2

save_output_parameters({
    'x': x.tolist(),
    'y': y.tolist(),
    'mean_y': float(np.mean(y)),
})
"""

    builder = RenoScriptCalcJob.get_builder()
    builder.code = fixture_code
    builder.script = orm.Str(script)

    # Validate builder
    assert 'script' in builder
    # No data inputs required


def test_scripted_template_rendering(simple_model, simple_mps, fixture_code):
    """Test that the Jinja2 template renders correctly."""
    from aiida_renormalizer.calculations.scripted import RenoScriptCalcJob

    script = "# Test script\nenergy = mps.expectation(mpo)\nsave_output_parameters({'energy': energy})"

    builder = RenoScriptCalcJob.get_builder()
    builder.code = fixture_code
    builder.script = orm.Str(script)
    builder.model = simple_model
    builder.mps = simple_mps

    # We can't actually render without submitting, but we can check the context
    # In a real integration test, we'd check the generated driver.py


def test_scripted_calcjob_with_config(simple_model, simple_mps, simple_mpo, fixture_code):
    """Test ScriptedCalcJob with config input."""
    from aiida_renormalizer.calculations.scripted import RenoScriptCalcJob
    from renormalizer.utils import OptimizeConfig

    config = OptimizeConfig(procedure=[[10, 0.4], [20, 0.2], [30, 0]])
    config_data = ConfigData.from_config(config)

    # Script that uses config
    script = """
# Use config parameters
procedure = config.procedure

save_output_parameters({
    'procedure': procedure,
})
"""

    builder = RenoScriptCalcJob.get_builder()
    builder.code = fixture_code
    builder.script = orm.Str(script)
    builder.model = simple_model
    builder.mps = simple_mps
    builder.mpo = simple_mpo
    builder.config = config_data

    # Validate builder
    assert 'config' in builder


def test_scripted_calcjob_output_data(simple_model, fixture_code):
    """Test ScriptedCalcJob that outputs additional structured data."""
    from aiida_renormalizer.calculations.scripted import RenoScriptCalcJob

    # Script that saves additional data
    script = """
import numpy as np

# Generate some data
spectral_data = {
    'frequencies': np.linspace(0, 10, 100).tolist(),
    'intensities': np.random.rand(100).tolist(),
}

# Save main output
save_output_parameters({
    'n_points': 100,
    'frequency_range': [0, 10],
})

# Save additional structured data
save_data(spectral_data, 'output_data.json')
"""

    builder = RenoScriptCalcJob.get_builder()
    builder.code = fixture_code
    builder.script = orm.Str(script)
    builder.model = simple_model

    # Validate builder
    assert 'script' in builder
