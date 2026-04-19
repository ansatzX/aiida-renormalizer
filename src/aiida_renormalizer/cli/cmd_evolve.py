# -*- coding: utf-8 -*-
""" verdi reno evolve command."""
import click
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.cmdline.utils import echo
from aiida import orm
from aiida.engine import submit, run
from aiida.orm import load_node


@click.command('evolve')
@click.option(
    '-s', '--state',
    type=int,
    required=True,
    help='Node ID (PK) of the initial MPS/TTNS state'
)
@click.option(
    '-H', '--hamiltonian',
    type=int,
    required=True,
    help='Node ID (PK) of the MPO/TTNO Hamiltonian'
)
@click.option(
    '-t', '--total-time',
    type=float,
    required=True,
    help='Total evolution time'
)
@click.option(
    '-d', '--timestep',
    type=float,
    default=0.01,
    help='Time step dt (default: 0.01)'
)
@click.option(
    '-m', '--method',
    type=click.Choice(['tdvp', 'tddmrp', 'rk4']),
    default='tdvp',
    help='Time evolution method (default: tdvp)'
)
@click.option(
    '-C', '--checkpoint-time',
    type=float,
    default=None,
    help='Time between checkpoints (for long evolutions)'
)
@click.option(
    '--max-energy-drift',
    type=float,
    default=1e-6,
    help='Maximum allowed energy drift (default: 1e-6)'
)
@click.option(
    '-i', '--trajectory-interval',
    type=int,
    default=None,
    help='Save trajectory every N steps'
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
    '-D', '--description',
    type=str,
    default='',
    help='Description for the calculation node'
)
@with_dbenv()
def evolve(state, hamiltonian, total_time, timestep, method, checkpoint_time,
           max_energy_drift, trajectory_interval, artifact_storage_base,
           artifact_storage_backend, submit, label, description):
    """Run time evolution calculation.

    This command runs real-time evolution of an MPS or TTNS state under
    a given Hamiltonian using TDVP or other methods.

    Examples:

        # Basic time evolution
        verdi reno evolve -s 123 -H 456 -t 100.0

        # Specify method and time step
        verdi reno evolve -s 123 -H 456 -t 100.0 -m tdvp -d 0.001

        # Long evolution with checkpoints
        verdi reno evolve -s 123 -H 456 -t 1000.0 -C 10.0

        # Submit to daemon
        verdi reno evolve -s 123 -H 456 -t 100.0 --submit
    """
    from aiida_renormalizer.data.mps import MpsData
    from aiida_renormalizer.data.ttns import TTNSData
    from aiida_renormalizer.data.mpo import MpoData
    from aiida_renormalizer.data.ttno import TtnoData
    from aiida_renormalizer.data.model import ModelData
    from aiida_renormalizer.data.basis_tree import BasisTreeData

    # Load initial state
    echo.echo_info(f"Loading initial state (PK: {state})...")
    try:
        initial_state = load_node(state)
    except Exception as e:
        echo.echo_critical(f"Failed to load state {state}: {e}")

    # Determine tensor network type from state
    if isinstance(initial_state, MpsData):
        tn_type = 'mps'
        echo.echo_info("Detected MPS state")
    elif isinstance(initial_state, TTNSData):
        tn_type = 'ttn'
        echo.echo_info("Detected TTNS state")
    else:
        echo.echo_critical(
            f"Unsupported state type: {type(initial_state).__name__}. "
            "Must be MpsData or TTNSData."
        )

    # Load Hamiltonian
    echo.echo_info(f"Loading Hamiltonian (PK: {hamiltonian})...")
    try:
        hamiltonian_obj = load_node(hamiltonian)
    except Exception as e:
        echo.echo_critical(f"Failed to load Hamiltonian {hamiltonian}: {e}")

    # Validate Hamiltonian type
    if tn_type == 'mps' and not isinstance(hamiltonian_obj, MpoData):
        echo.echo_critical(
            f"Hamiltonian must be MpoData for MPS evolution, got "
            f"{type(hamiltonian_obj).__name__}"
        )
    elif tn_type == 'ttn' and not isinstance(hamiltonian_obj, TtnoData):
        echo.echo_critical(
            f"Hamiltonian must be TtnoData for TTN evolution, got "
            f"{type(hamiltonian_obj).__name__}"
        )

    # Find code
    codes = orm.QueryBuilder().append(orm.Code, filters={
        'attributes.input_plugin': 'reno.script'
    }).all()
    if codes:
        code = codes[0][0]
        echo.echo_info(f"Using code: {code.label}")
    else:
        echo.echo_critical(
            "No code found. Please setup a code with 'reno.script' plugin."
        )

    # Build configuration
    config = {
        'method': method,
        'dt': timestep,
    }

    # Run evolution based on tensor network type
    if tn_type == 'mps':
        from aiida_renormalizer.workchains.time_evolution import TimeEvolutionWorkChain

        # Need model for MPS evolution
        # Try to get it from the initial state
        try:
            model = initial_state.creator.outputs.model
        except (AttributeError, KeyError):
            echo.echo_critical(
                "Cannot find model for MPS evolution. "
                "Initial state must have model as input."
            )

        echo.echo_info("Running MPS time evolution...")
        inputs = {
            'model': model,
            'mpo': hamiltonian_obj,
            'initial_mps': initial_state,
            'total_time': orm.Float(total_time),
            'config': orm.Dict(config),
            'max_energy_drift': orm.Float(max_energy_drift),
            'code': code,
        }

        if checkpoint_time:
            inputs['checkpoint_time'] = orm.Float(checkpoint_time)

        if trajectory_interval:
            inputs['trajectory_interval'] = orm.Int(trajectory_interval)

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
            future = submit(TimeEvolutionWorkChain, **inputs)
            echo.echo_success(f"Submitted TimeEvolutionWorkChain: {future.pk}")
            echo.echo_info(f"Check status with: verdi process show {future.pk}")
        else:
            result = run(TimeEvolutionWorkChain, **inputs)
            echo.echo_success(f"Evolution completed")
            echo.echo_info(f"Final state: {result['final_mps'].pk}")
            if 'trajectory' in result:
                echo.echo_info(f"Trajectory: {result['trajectory'].pk}")

    elif tn_type == 'ttn':
        from aiida_renormalizer.workchains.ttn_time_evolution import TtnTimeEvolutionWorkChain

        # Need basis tree for TTN evolution
        try:
            basis_tree = initial_state.creator.outputs.basis_tree
        except (AttributeError, KeyError):
            echo.echo_critical(
                "Cannot find basis tree for TTN evolution. "
                "Initial state must have basis_tree as input."
            )

        echo.echo_info("Running TTN time evolution...")
        inputs = {
            'basis_tree': basis_tree,
            'ttno': hamiltonian_obj,
            'initial_ttns': initial_state,
            'total_time': orm.Float(total_time),
            'config': orm.Dict(config),
            'max_energy_drift': orm.Float(max_energy_drift),
            'code': code,
        }

        if checkpoint_time:
            inputs['checkpoint_time'] = orm.Float(checkpoint_time)

        if trajectory_interval:
            inputs['trajectory_interval'] = orm.Int(trajectory_interval)

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
            future = submit(TtnTimeEvolutionWorkChain, **inputs)
            echo.echo_success(f"Submitted TtnTimeEvolutionWorkChain: {future.pk}")
            echo.echo_info(f"Check status with: verdi process show {future.pk}")
        else:
            result = run(TtnTimeEvolutionWorkChain, **inputs)
            echo.echo_success(f"Evolution completed")
            echo.echo_info(f"Final state: {result['final_ttns'].pk}")
            if 'trajectory' in result:
                echo.echo_info(f"Trajectory: {result['trajectory'].pk}")
