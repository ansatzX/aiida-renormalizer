# -*- coding: utf-8 -*-
""" verdi reno ground-state command."""
import click
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.cmdline.utils import echo
from aiida import orm
from aiida.engine import submit, run


@click.command('ground-state')
@click.option(
    '-m', '--model',
    type=click.Path(exists=True),
    required=True,
    help='Path to model configuration TOML or JSON file'
)
@click.option(
    '-b', '--basis',
    type=click.Path(exists=True),
    required=True,
    help='Path to basis set configuration TOML or JSON file'
)
@click.option(
    '-c', '--config',
    type=click.Path(exists=True),
    default=None,
    help='Path to calculation configuration TOML or JSON file'
)
@click.option(
    '-t', '--tensor-network',
    type=click.Choice(['mps', 'ttn']),
    default='mps',
    help='Tensor network type: mps (default) or ttn'
)
@click.option(
    '-T', '--tree-topology',
    type=click.Path(exists=True),
    default=None,
    help='Path to tree topology configuration (required for TTN)'
)
@click.option(
    '-C', '--code',
    type=str,
    default=None,
    help='AiiDA code label or ID to use for calculations'
)
@click.option(
    '-M', '--max-bond-dim',
    type=int,
    default=None,
    help='Maximum bond dimension for MPS/TTN'
)
@click.option(
    '-e', '--energy-convergence',
    type=float,
    default=1e-6,
    help='Energy convergence threshold (default: 1e-6)'
)
@click.option(
    '--artifact-storage-base',
    type=click.Path(),
    default=None,
    help='Base directory for external MPS/TTNS artifacts'
)
@click.option(
    '--artifact-storage-backend',
    type=click.Choice(['posix']),
    default='posix',
    show_default=True,
    help='Artifact storage backend'
)
@click.option(
    '--submit',
    is_flag=True,
    default=False,
    help='Submit to daemon instead of running synchronously'
)
@click.option(
    '-l', '--label',
    type=str,
    default='',
    help='Label for the calculation node'
)
@click.option(
    '-d', '--description',
    type=str,
    default='',
    help='Description for the calculation node'
)
@with_dbenv()
def ground_state(model, basis, config, tensor_network, tree_topology, code,
                 max_bond_dim, energy_convergence, artifact_storage_base,
                 artifact_storage_backend, submit, label, description):
    """Run ground state calculation using MPS or TTN.

    This command runs a ground state calculation using DMRG (for MPS) or
    variational optimization (for TTN).

    Examples:

        # MPS ground state with default settings
        verdi reno ground-state -m model.toml -b basis.toml

        # TTN ground state with tree topology
        verdi reno ground-state -m model.toml -b basis.toml -t ttn -T tree.toml

        # Submit to daemon for async execution
        verdi reno ground-state -m model.toml -b basis.toml --submit

        # Specify code and convergence criteria
        verdi reno ground-state -m model.toml -b basis.toml -C renormalizer@localhost -e 1e-8
    """
    import json
    from pathlib import Path

    try:
        import tomllib  # py>=3.11
    except ModuleNotFoundError:  # py<=3.10
        import tomli as tomllib
    from aiida_renormalizer.data.model import ModelData
    from aiida_renormalizer.data.basis import BasisSetData
    from aiida_renormalizer.data.basis_tree import BasisTreeData
    from aiida_renormalizer.data.mpo import MpoData
    from aiida_renormalizer.data.ttno import TtnoData

    # Load model configuration
    model_path = Path(model)
    if model_path.suffix == '.toml':
        with open(model_path, 'rb') as f:
            model_config = tomllib.load(f)
    elif model_path.suffix == '.json':
        with open(model_path) as f:
            model_config = json.load(f)
    else:
        echo.echo_critical(f"Unsupported model file format: {model_path.suffix}")

    # Load basis configuration
    basis_path = Path(basis)
    if basis_path.suffix == '.toml':
        with open(basis_path, 'rb') as f:
            basis_config = tomllib.load(f)
    elif basis_path.suffix == '.json':
        with open(basis_path) as f:
            basis_config = json.load(f)
    else:
        echo.echo_critical(f"Unsupported basis file format: {basis_path.suffix}")

    # Load calculation configuration
    calc_config = {}
    if config:
        config_path = Path(config)
        if config_path.suffix == '.toml':
            with open(config_path, 'rb') as f:
                calc_config = tomllib.load(f)
        elif config_path.suffix == '.json':
            with open(config_path) as f:
                calc_config = json.load(f)

    # Override with command-line options
    if max_bond_dim:
        calc_config['max_bond_dim'] = max_bond_dim

    # Get code
    if code:
        try:
            code_obj = orm.load_code(code)
        except Exception as e:
            echo.echo_critical(f"Failed to load code '{code}': {e}")
    else:
        # Try to find a default code
        codes = orm.QueryBuilder().append(orm.Code, filters={
            'attributes.input_plugin': 'reno.script'
        }).all()
        if codes:
            code_obj = codes[0][0]
            echo.echo_info(f"Using code: {code_obj.label}")
        else:
            echo.echo_critical(
                "No code found. Please specify with -C option or setup a code."
            )

    # Create ModelData and BasisSetData
    echo.echo_info("Creating model and basis set...")
    model_data = ModelData.from_dict(model_config)
    basis_data = BasisSetData.from_dict(basis_config)

    # Run calculation based on tensor network type
    if tensor_network == 'mps':
        from aiida_renormalizer.workchains.ground_state import GroundStateWorkChain
        from aiida_renormalizer.calculations.basic.build_mpo import BuildMpoCalcJob

        # Build MPO
        echo.echo_info("Building MPO...")
        builder = BuildMpoCalcJob.get_builder()
        builder.model = model_data
        builder.basis = basis_data
        builder.code = code_obj

        mpo_result = run(builder)
        mpo = mpo_result['mpo']

        # Build MPS
        echo.echo_info("Running ground state calculation...")
        inputs = {
            'model': model_data,
            'mpo': mpo,
            'code': code_obj,
            'energy_convergence': orm.Float(energy_convergence),
        }

        if calc_config:
            inputs['config'] = orm.Dict(calc_config)

        metadata = {}
        if label or description:
            metadata.update({'label': label, 'description': description})
        if artifact_storage_base:
            metadata.setdefault('options', {})
            metadata['options']['artifact_storage_base'] = artifact_storage_base
            metadata['options']['artifact_storage_backend'] = artifact_storage_backend
        if metadata:
            inputs['metadata'] = metadata

        if submit:
            future = submit(GroundStateWorkChain, **inputs)
            echo.echo_success(f"Submitted GroundStateWorkChain: {future.pk}")
            echo.echo_info(f"Check status with: verdi process show {future.pk}")
        else:
            result = run(GroundStateWorkChain, **inputs)
            echo.echo_success(f"Ground state energy: {result['energy'].value:.10f}")
            echo.echo_info(f"MPS node: {result['ground_state'].pk}")

    elif tensor_network == 'ttn':
        from aiida_renormalizer.workchains.ttn_ground_state import TtnGroundStateWorkChain

        if not tree_topology:
            echo.echo_critical(
                "Tree topology required for TTN calculations. Use -T option."
            )

        # Load tree topology
        tree_path = Path(tree_topology)
        if tree_path.suffix == '.toml':
            with open(tree_path, 'rb') as f:
                tree_config = tomllib.load(f)
        elif tree_path.suffix == '.json':
            with open(tree_path) as f:
                tree_config = json.load(f)
        else:
            echo.echo_critical(
                f"Unsupported tree topology file format: {tree_path.suffix}"
            )

        # Create BasisTreeData and TTNO
        echo.echo_info("Creating tree topology and TTNO...")
        basis_tree = BasisTreeData.from_dict(tree_config)
        ttno = TtnoData.from_model_and_tree(model_data, basis_tree)

        # Run TTN ground state calculation
        echo.echo_info("Running TTN ground state calculation...")
        inputs = {
            'basis_tree': basis_tree,
            'ttno': ttno,
            'code': code_obj,
            'energy_convergence': orm.Float(energy_convergence),
        }

        if calc_config:
            inputs['config'] = orm.Dict(calc_config)

        metadata = {}
        if label or description:
            metadata.update({'label': label, 'description': description})
        if artifact_storage_base:
            metadata.setdefault('options', {})
            metadata['options']['artifact_storage_base'] = artifact_storage_base
            metadata['options']['artifact_storage_backend'] = artifact_storage_backend
        if metadata:
            inputs['metadata'] = metadata

        if submit:
            future = submit(TtnGroundStateWorkChain, **inputs)
            echo.echo_success(f"Submitted TtnGroundStateWorkChain: {future.pk}")
            echo.echo_info(f"Check status with: verdi process show {future.pk}")
        else:
            result = run(TtnGroundStateWorkChain, **inputs)
            echo.echo_success(f"Ground state energy: {result['energy'].value:.10f}")
            echo.echo_info(f"TTNS node: {result['ground_state'].pk}")
