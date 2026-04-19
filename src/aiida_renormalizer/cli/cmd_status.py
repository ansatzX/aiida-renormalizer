# -*- coding: utf-8 -*-
""" verdi reno status command."""
import click
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.cmdline.utils import echo
from aiida import orm
from aiida.engine import ProcessState


@click.command('status')
@click.option(
    '-p', '--process',
    type=int,
    default=None,
    help='Show detailed status for a specific process (PK)'
)
@click.option(
    '-a', '--all',
    is_flag=True,
    default=False,
    help='Show all processes (including finished)'
)
@click.option(
    '-l', '--limit',
    type=int,
    default=20,
    help='Maximum number of processes to show (default: 20)'
)
@click.option(
    '-t', '--type',
    type=click.Choice(['workchain', 'calculation', 'all']),
    default='all',
    help='Filter by process type (default: all)'
)
@with_dbenv()
def status(process, all, limit, type):
    """Show status of aiida-renormalizer processes.

    This command shows the status of recent or specific processes created
    by aiida-renormalizer commands.

    Examples:

        # Show recent active processes
        verdi reno status

        # Show all processes including finished
        verdi reno status --all

        # Show detailed status for specific process
        verdi reno status -p 123

        # Show only WorkChains
        verdi reno status -t workchain

        # Show last 50 processes
        verdi reno status -l 50
    """
    from aiida_renormalizer.workchains.ground_state import GroundStateWorkChain
    from aiida_renormalizer.workchains.time_evolution import TimeEvolutionWorkChain
    from aiida_renormalizer.workchains.ttn_ground_state import TtnGroundStateWorkChain
    from aiida_renormalizer.workchains.ttn_time_evolution import TtnTimeEvolutionWorkChain
    from aiida_renormalizer.workchains.convergence import ConvergenceWorkChain
    from aiida_renormalizer.workchains.sweep import (
        BondDimensionSweepWorkChain,
        TemperatureSweepWorkChain,
        FrequencySweepWorkChain,
        ParameterSweepWorkChain
    )

    # Define renormalizer process types
    reno_workchains = [
        GroundStateWorkChain,
        TimeEvolutionWorkChain,
        TtnGroundStateWorkChain,
        TtnTimeEvolutionWorkChain,
        ConvergenceWorkChain,
        BondDimensionSweepWorkChain,
        TemperatureSweepWorkChain,
        FrequencySweepWorkChain,
        ParameterSweepWorkChain,
    ]

    if process:
        # Show detailed status for specific process
        try:
            node = orm.load_node(process)
        except Exception as e:
            echo.echo_critical(f"Failed to load process {process}: {e}")

        echo.echo_info(f"Process {process} details:")
        echo.echo(f"  Type: {node.process_label}")
        echo.echo(f"  State: {node.process_state.value}")
        echo.echo(f"  Label: {node.label or '(none)'}")
        echo.echo(f"  Description: {node.description or '(none)'}")
        echo.echo(f"  Creation time: {node.ctime}")
        echo.echo(f"  Modification time: {node.mtime}")

        if node.process_state in [ProcessState.FINISHED]:
            if hasattr(node, 'outputs'):
                echo.echo("\nOutputs:")
                for key, output in node.outputs.items():
                    echo.echo(f"  {key}: {output.__class__.__name__} (PK: {output.pk})")

        if node.process_state in [ProcessState.EXCEPTED, ProcessState.KILLED]:
            echo.echo("\nExit code:")
            echo.echo(f"  {node.exit_status}: {node.exit_message}")

        return

    # Build query for renormalizer processes
    qb = orm.QueryBuilder()

    if type == 'workchain':
        # Query for WorkChains only
        qb.append(orm.WorkChainNode, tag='process')
    elif type == 'calculation':
        # Query for CalcJobs
        qb.append(orm.CalcJobNode, tag='process')
    else:
        # Query for all process types
        qb.append(orm.ProcessNode, tag='process')

    # Filter by process state if not showing all
    if not all:
        qb.add_filter('process', {
            'attributes.process_state': {'in': [
                ProcessState.CREATED.value,
                ProcessState.WAITING.value,
                ProcessState.RUNNING.value
            ]}
        })

    # Order by creation time (most recent first)
    qb.order_by({'process': {'ctime': 'desc'}})
    qb.limit(limit)

    results = qb.all()

    if not results:
        if all:
            echo.echo_info("No processes found.")
        else:
            echo.echo_info(
                "No active processes found. Use --all to show finished processes."
            )
        return

    # Display results
    echo.echo_info(f"Found {len(results)} processes:\n")
    echo.echo(
        f"{'PK':<8} {'Type':<30} {'State':<12} {'Label':<20} {'Created':<20}"
    )
    echo.echo("=" * 90)

    for row in results:
        process_node = row[0]
        pk = process_node.pk
        process_type = process_node.node_type.split('.')[-2] if process_node.node_type else 'unknown'
        state = process_node.process_state.value if hasattr(process_node, 'process_state') and process_node.process_state else 'unknown'
        label = process_node.label[:18] if process_node.label else ''
        created = process_node.ctime.strftime('%Y-%m-%d %H:%M:%S')

        # Color-code the state
        if state == 'finished':
            state_str = echo.echo_success(state, show=False)
        elif state == 'running':
            state_str = echo.echo_warning(state, show=False)
        elif state in ['excepted', 'killed']:
            state_str = echo.echo_error(state, show=False)
        else:
            state_str = state

        echo.echo(f"{pk:<8} {process_type:<30} {state:<12} {label:<20} {created:<20}")

    echo.echo("")
    echo.echo_info(
        "Use 'verdi reno status -p <PK>' for detailed information about a process."
    )
