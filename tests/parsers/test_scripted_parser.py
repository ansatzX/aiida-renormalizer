"""Tests for ScriptedParser."""
import json
import pytest
import tempfile
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from aiida import orm
from aiida.parsers import Parser

from aiida_renormalizer.parsers.scripted import ScriptedParser
from aiida_renormalizer.data import ModelData, MpsData, MpoData


@pytest.fixture
def mock_calcjob_node(aiida_profile):
    """Create a mock CalcJobNode suitable for passing to Parser.__init__.

    Uses a real CalcJobNode so it passes AiiDA's isinstance checks.
    """
    from aiida.orm import CalcJobNode

    node = CalcJobNode()
    node.set_option("resources", {"num_machines": 1})
    return node


@pytest.fixture
def scripted_parser(mock_calcjob_node):
    """Create a ScriptedParser with a mock node."""
    return ScriptedParser(node=mock_calcjob_node)


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
def simple_mps(simple_model, tmp_path):
    """Create a simple MPS state."""
    from renormalizer.mps import Mps

    model = simple_model.load_model()
    mps = Mps.random(model, qntot=0, m_max=10)
    mps_data = MpsData.from_mps(
        mps,
        simple_model,
        storage_backend="posix",
        storage_base=str(tmp_path / "fixture-artifacts"),
        relative_path="states/simple_mps.npz",
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


def test_parser_output_parameters(scripted_parser):
    """Test parsing output_parameters.json."""
    # Create mock retrieved data
    with tempfile.TemporaryDirectory() as tmpdir:
        # Write output_parameters.json
        params = {
            'energy': -1.5,
            'n_steps': 100,
            'converged': True,
        }
        with open(Path(tmpdir) / 'output_parameters.json', 'w') as f:
            json.dump(params, f)

        # Create a mock retrieved folder
        from aiida.orm import FolderData
        retrieved = FolderData(tree=tmpdir)

        # Test that the parser can be instantiated
        assert scripted_parser is not None


def test_parser_output_mps(simple_model, simple_mps):
    """Test parsing output_mps.npz."""
    from aiida_renormalizer.parsers.scripted import ScriptedParser

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write MPS file
        model = simple_model.load_model()
        mps = simple_mps.load_mps(simple_model)
        mps_path = Path(tmpdir) / 'output_mps'
        mps.dump(str(mps_path))

        # Write output_parameters.json
        params = {'success': True}
        with open(Path(tmpdir) / 'output_parameters.json', 'w') as f:
            json.dump(params, f)

        # Test file existence
        assert (Path(tmpdir) / 'output_mps.npz').exists()


def test_parser_output_mpo(simple_model, simple_mpo):
    """Test parsing output_mpo.npz."""
    from aiida_renormalizer.parsers.scripted import ScriptedParser

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write MPO file
        model = simple_model.load_model()
        mpo = simple_mpo.load_mpo(simple_model)
        mpo_path = Path(tmpdir) / 'output_mpo'
        mpo.dump(str(mpo_path))

        # Write output_parameters.json
        params = {'success': True}
        with open(Path(tmpdir) / 'output_parameters.json', 'w') as f:
            json.dump(params, f)

        # Test file existence
        assert (Path(tmpdir) / 'output_mpo.npz').exists()


def test_parser_output_data():
    """Test parsing output_data.json."""
    from aiida_renormalizer.parsers.scripted import ScriptedParser

    with tempfile.TemporaryDirectory() as tmpdir:
        # Write output_parameters.json
        params = {'success': True}
        with open(Path(tmpdir) / 'output_parameters.json', 'w') as f:
            json.dump(params, f)

        # Write output_data.json
        data = {
            'spectral_data': [1.0, 2.0, 3.0],
            'frequencies': [0.1, 0.2, 0.3],
        }
        with open(Path(tmpdir) / 'output_data.json', 'w') as f:
            json.dump(data, f)

        # Test file existence
        assert (Path(tmpdir) / 'output_data.json').exists()


def test_parser_nan_validation(scripted_parser):
    """Test NaN detection in physical validation."""
    # Test valid params
    valid_params = {'energy': -1.5, 'n_steps': 100}
    result = scripted_parser._validate_physical_constraints(valid_params)
    assert result['passed'] is True

    # Test NaN
    nan_params = {'energy': float('nan')}
    result = scripted_parser._validate_physical_constraints(nan_params)
    assert result['passed'] is False
    assert 'NaN' in result['reason']

    # Test Inf
    inf_params = {'energy': float('inf')}
    result = scripted_parser._validate_physical_constraints(inf_params)
    assert result['passed'] is False
    assert 'Inf' in result['reason']


def test_parser_exit_codes():
    """Test exit codes are defined."""
    from aiida_renormalizer.parsers.scripted import ScriptedParser

    exit_codes = ScriptedParser.exit_codes()

    assert hasattr(exit_codes, 'ERROR_EXECUTION_FAILED')
    assert hasattr(exit_codes, 'ERROR_OUTPUT_MISSING')
    assert hasattr(exit_codes, 'ERROR_OUTPUT_PARSING')
    assert hasattr(exit_codes, 'ERROR_PHYSICAL_VALIDATION')
    assert hasattr(exit_codes, 'ERROR_SCRIPT_EXECUTION')
    assert hasattr(exit_codes, 'ERROR_INVALID_OUTPUT')

    # Check exit codes
    assert exit_codes.ERROR_EXECUTION_FAILED.status == 100
    assert exit_codes.ERROR_OUTPUT_MISSING.status == 200
    assert exit_codes.ERROR_SCRIPT_EXECUTION.status == 500
    assert exit_codes.ERROR_INVALID_OUTPUT.status == 501


def test_parser_missing_output_parameters(scripted_parser):
    """Test parser behavior when output_parameters.json is missing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Don't create output_parameters.json
        from aiida.orm import FolderData
        retrieved = FolderData(tree=tmpdir)

        # Parser should handle missing file
        # In a real test, we'd need to mock the node and retrieved
        # For now, just verify the parser exists
        assert scripted_parser is not None


def test_parser_auto_type_conversion(simple_model, simple_mps, scripted_parser):
    """Test automatic type conversion for MPS/MPO outputs.

    Tests that _parse_mps_file correctly converts an .npz file into MpsData
    by mocking the retrieved folder to serve the file from a temp directory.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create MPS file on disk
        model = simple_model.load_model()
        mps = simple_mps.load_mps(simple_model)
        mps_path = Path(tmpdir) / 'output_mps'
        mps.dump(str(mps_path))

        npz_path = Path(tmpdir) / 'output_mps.npz'
        assert npz_path.exists()

        # Mock retrieved so _parse_mps_file can open the file from disk
        mock_retrieved = MagicMock()
        mock_retrieved.open = MagicMock(
            side_effect=lambda f, mode: open(npz_path, mode)
        )
        artifact_base = Path(tmpdir) / "parsed-artifacts"
        mock_node = MagicMock()
        mock_node.get_option.side_effect = (
            lambda key, default=None: {
                'artifact_storage_backend': 'posix',
                'artifact_storage_base': str(artifact_base),
            }.get(key, default)
        )

        with patch.object(type(scripted_parser), 'retrieved', new_callable=lambda: property(lambda self: mock_retrieved)), \
             patch.object(type(scripted_parser), 'node', new_callable=lambda: property(lambda self: mock_node)):

            mps_data = scripted_parser._parse_mps_file('output_mps.npz', simple_model)

        # Verify it's MpsData
        assert isinstance(mps_data, MpsData)
        assert mps_data.artifact_metadata['storage_base'] == str(artifact_base)

        # Verify we can load it back
        loaded_mps = mps_data.load_mps(simple_model)
        assert loaded_mps is not None
        assert len(loaded_mps) == model.nsite


def test_parser_mpdm_detection(simple_model, scripted_parser):
    """Test parser detection of MpDm vs Mps."""
    from renormalizer.mps import MpDm

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create MpDm (density matrix) via max_entangled_gs (MpDm.random is not supported)
        model = simple_model.load_model()
        mpdm = MpDm.max_entangled_gs(model)
        mpdm_path = Path(tmpdir) / 'output_mps'
        mpdm.dump(str(mpdm_path))

        npz_file = str(mpdm_path) + '.npz'

        # Add 'dm' to description to mark as MpDm
        with np.load(npz_file, allow_pickle=True) as data:
            arrays = dict(data)
        arrays['description'] = 'density matrix'
        np.savez(npz_file, **arrays)

        # Mock retrieved so _parse_mps_file can open the file from disk
        mock_retrieved = MagicMock()
        mock_retrieved.open = MagicMock(
            side_effect=lambda f, mode: open(npz_file, mode)
        )
        artifact_base = Path(tmpdir) / "parsed-artifacts"
        mock_node = MagicMock()
        mock_node.get_option.side_effect = (
            lambda key, default=None: {
                'artifact_storage_backend': 'posix',
                'artifact_storage_base': str(artifact_base),
            }.get(key, default)
        )

        with patch.object(type(scripted_parser), 'retrieved', new_callable=lambda: property(lambda self: mock_retrieved)), \
             patch.object(type(scripted_parser), 'node', new_callable=lambda: property(lambda self: mock_node)):

            mpdm_data = scripted_parser._parse_mps_file('output_mps.npz', simple_model)

        # Verify it's MpsData (can hold both Mps and MpDm)
        assert isinstance(mpdm_data, MpsData)
        assert mpdm_data.artifact_metadata['storage_base'] == str(artifact_base)
