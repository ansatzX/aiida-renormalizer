# -*- coding: utf-8 -*-
"""Tests for CLI commands."""
import json

import pytest
from click.testing import CliRunner
from aiida import orm
from aiida.engine import run


@pytest.fixture
def runner():
    """Click CLI runner."""
    return CliRunner()


@pytest.fixture
def sample_model_config(tmp_path):
    """Create a sample model configuration file."""
    import yaml
    config = {
        'model_type': 'spin_boson',
        'n_sites': 4,
        'hopping': 1.0,
        'coupling': 0.5,
        'phonon_freq': 0.1,
    }
    config_file = tmp_path / 'model.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    return config_file


@pytest.fixture
def sample_basis_config(tmp_path):
    """Create a sample basis configuration file."""
    import yaml
    config = {
        'n_phys_dim': 2,
        'n_vib': 4,
    }
    config_file = tmp_path / 'basis.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    return config_file


@pytest.fixture
def sample_tree_config(tmp_path):
    """Create a sample tree topology configuration file."""
    import yaml
    config = {
        'tree_type': 'binary',
        'n_leaves': 8,
        'basis_grouping': 'default',
    }
    config_file = tmp_path / 'tree.yaml'
    with open(config_file, 'w') as f:
        yaml.dump(config, f)
    return config_file


@pytest.fixture
def mock_code(aiida_profile):
    """Create a mock code for testing."""
    from aiida.orm import InstalledCode, Computer
    from aiida.common.exceptions import NotExistent

    # Try to get existing computer
    try:
        computer = Computer.objects.get(label='localhost-test')
    except NotExistent:
        computer = Computer(
            label='localhost-test',
            hostname='localhost',
            description='Local computer for testing',
            transport_type='core.local',
            scheduler_type='core.direct',
            workdir='/tmp/aiida_run'
        )
        computer.store()
        computer.set_default_mpiprocs_per_machine(1)
        computer.configure()

    # Create or get code
    try:
        code = orm.load_code('test-reno@localhost-test')
    except:
        code = InstalledCode(
            label='test-reno',
            computer=computer,
            filepath_executable='/bin/echo',
            default_calc_job_plugin='reno.script',
        )
        code.store()

    return code


@pytest.fixture
def stored_mps_node(aiida_profile, sho_model, sho_mps, tmp_path):
    """Create a stored MpsData node backed by an external artifact."""
    from aiida_renormalizer.data import ModelData, MpsData

    model_node = ModelData.from_model(sho_model)
    model_node.store()

    mps_node = MpsData.from_mps(
        sho_mps,
        model_node,
        storage_backend="posix",
        storage_base=str(tmp_path / "source-artifacts"),
        relative_path="states/source_mps.npz",
    )
    mps_node.store()
    return mps_node


def test_reno_group(runner):
    """Test main reno command group."""
    from aiida_renormalizer.cli.cmd_reno import reno

    result = runner.invoke(reno, ['--help'])
    assert result.exit_code == 0
    assert 'Commands for aiida-renormalizer' in result.output
    assert 'bundle' in result.output


def test_bundle_help(runner):
    """Test bundle command help."""
    from aiida_renormalizer.cli.cmd_bundle import bundle

    result = runner.invoke(bundle, ['--help'])
    assert result.exit_code == 0
    assert 'Export stored artifact nodes' in result.output


def test_bundle_exports_artifact_and_manifest(runner, stored_mps_node, tmp_path):
    """Bundle command should write a publication-ready bundle layout."""
    from aiida_renormalizer.cli.cmd_bundle import bundle

    output_dir = tmp_path / "bundle"
    result = runner.invoke(bundle, ['-n', str(stored_mps_node.pk), '-o', str(output_dir)])

    assert result.exit_code == 0
    assert (output_dir / "manifest.json").exists()
    assert (output_dir / "README.md").exists()
    assert (output_dir / "metadata" / "summary.json").exists()

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["node_uuid"] == str(stored_mps_node.uuid)
    assert manifest["bundle_relative_path"].startswith("artifacts/mps-")
    assert manifest["bundle_relative_path"].endswith(".npz")
    assert manifest["exported_at"]
    assert manifest["summary"]["n_sites"] == stored_mps_node.base.attributes.get("n_sites")
    bundled_path = output_dir / manifest["bundle_relative_path"]
    assert bundled_path.exists()
    assert bundled_path.name == f"mps-{str(stored_mps_node.uuid)[:8]}.npz"

    summary = json.loads((output_dir / "metadata" / "summary.json").read_text())
    assert summary["n_sites"] == stored_mps_node.base.attributes.get("n_sites")

    bundle_readme = (output_dir / "README.md").read_text()
    assert "publication bundle" in bundle_readme.lower()
    assert str(stored_mps_node.uuid) in bundle_readme


def test_bundle_relinks_node_when_requested(runner, stored_mps_node, tmp_path):
    """Bundle command should optionally relink the node to the bundle artifact."""
    from aiida_renormalizer.cli.cmd_bundle import bundle

    output_dir = tmp_path / "bundle-relink"
    result = runner.invoke(
        bundle,
        ['-n', str(stored_mps_node.pk), '-o', str(output_dir), '--relink'],
    )

    assert result.exit_code == 0
    assert stored_mps_node.artifact_metadata["storage_base"] == str(output_dir)
    assert stored_mps_node.artifact_metadata["relative_path"].startswith("artifacts/mps-")


def test_ground_state_help(runner):
    """Test ground-state command help."""
    from aiida_renormalizer.cli.cmd_ground_state import ground_state

    result = runner.invoke(ground_state, ['--help'])
    assert result.exit_code == 0
    assert 'Run ground state calculation' in result.output
    assert '--artifact-storage-base' in result.output


def test_evolve_help(runner):
    """Test evolve command help."""
    from aiida_renormalizer.cli.cmd_evolve import evolve

    result = runner.invoke(evolve, ['--help'])
    assert result.exit_code == 0
    assert 'Run time evolution calculation' in result.output
    assert '--artifact-storage-base' in result.output


def test_spectrum_help(runner):
    """Test spectrum command help."""
    from aiida_renormalizer.cli.cmd_spectrum import spectrum

    result = runner.invoke(spectrum, ['--help'])
    assert result.exit_code == 0
    assert 'Calculate spectral properties' in result.output
    assert '--publication-bundle' in result.output


def test_convergence_help(runner):
    """Test convergence command help."""
    from aiida_renormalizer.cli.cmd_convergence import convergence

    result = runner.invoke(convergence, ['--help'])
    assert result.exit_code == 0
    assert 'Run convergence study' in result.output


def test_status_help(runner):
    """Test status command help."""
    from aiida_renormalizer.cli.cmd_status import status

    result = runner.invoke(status, ['--help'])
    assert result.exit_code == 0
    assert 'Show status of aiida-renormalizer processes' in result.output


def test_status_list(runner, aiida_profile):
    """Test status command listing processes."""
    from aiida_renormalizer.cli.cmd_status import status

    result = runner.invoke(status)
    assert result.exit_code == 0


def test_status_show_process(runner, aiida_profile, mock_code):
    """Test status command showing specific process."""
    from aiida_renormalizer.cli.cmd_status import status

    result = runner.invoke(status, ['-p', 999])
    # Should handle gracefully
    assert result.exit_code in [0, 1]


def test_ground_state_missing_model(runner, aiida_profile):
    """Test ground-state command with missing model file."""
    from aiida_renormalizer.cli.cmd_ground_state import ground_state

    result = runner.invoke(ground_state, ['-m', 'nonexistent.yaml', '-b', 'nonexistent.yaml'])
    assert result.exit_code != 0


def test_evolve_missing_state(runner, aiida_profile):
    """Test evolve command with missing state."""
    from aiida_renormalizer.cli.cmd_evolve import evolve

    result = runner.invoke(evolve, ['-s', 999, '-H', 999, '-t', 100.0])
    assert result.exit_code != 0


def test_spectrum_missing_state(runner, aiida_profile):
    """Test spectrum command with missing state."""
    from aiida_renormalizer.cli.cmd_spectrum import spectrum

    result = runner.invoke(spectrum, ['-s', 999, '-H', 999, '-o', 'a'])
    assert result.exit_code != 0


def test_convergence_missing_model(runner, aiida_profile):
    """Test convergence command with missing model file."""
    from aiida_renormalizer.cli.cmd_convergence import convergence

    result = runner.invoke(
        convergence,
        ['-m', 'nonexistent.yaml', '-b', 'nonexistent.yaml', '-r', '10,20']
    )
    assert result.exit_code != 0


def test_convergence_invalid_range(runner, aiida_profile, sample_model_config,
                                   sample_basis_config, mock_code):
    """Test convergence command with invalid range format."""
    from aiida_renormalizer.cli.cmd_convergence import convergence

    result = runner.invoke(
        convergence,
        [
            '-m', str(sample_model_config),
            '-b', str(sample_basis_config),
            '-t', 'bond_dimension',
            '-r', 'invalid'  # Invalid format
        ]
    )
    # Should fail with invalid range
    assert result.exit_code != 0 or 'Invalid' in result.output


def test_status_all_flag(runner, aiida_profile):
    """Test status command with --all flag."""
    from aiida_renormalizer.cli.cmd_status import status

    result = runner.invoke(status, ['--all'])
    assert result.exit_code == 0


def test_status_limit(runner, aiida_profile):
    """Test status command with limit option."""
    from aiida_renormalizer.cli.cmd_status import status

    result = runner.invoke(status, ['-l', '5'])
    assert result.exit_code == 0


def test_status_type_filter(runner, aiida_profile):
    """Test status command with type filter."""
    from aiida_renormalizer.cli.cmd_status import status

    # Test workchain filter
    result = runner.invoke(status, ['-t', 'workchain'])
    assert result.exit_code == 0

    # Test calculation filter
    result = runner.invoke(status, ['-t', 'calculation'])
    assert result.exit_code == 0
