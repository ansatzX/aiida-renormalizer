# -*- coding: utf-8 -*-
""" verdi reno spectrum command."""
import click
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.cmdline.utils import echo
from aiida import orm
from aiida.engine import submit, run
from aiida.orm import load_node


@click.command('spectrum')
@click.option(
    '-s', '--state',
    type=int,
    required=True,
    help='Node ID (PK) of the MPS/TTNS state (ground state or thermal state)'
)
@click.option(
    '-H', '--hamiltonian',
    type=int,
    required=True,
    help='Node ID (PK) of the MPO/TTNO Hamiltonian'
)
@click.option(
    '-o', '--operator',
    type=str,
    required=True,
    help='Name of the operator for spectrum calculation (e.g., "a", "a_dag", "n")'
)
@click.option(
    '-m', '--method',
    type=click.Choice(['spectra_zero_t', 'spectra_finite_t', 'kubo',
                       'correction_vector', 'charge_diffusion', 'spectral_function']),
    default='spectra_zero_t',
    help='Spectrum calculation method (default: spectra_zero_t)'
)
@click.option(
    '-f', '--frequency-range',
    type=str,
    default='-5,5,0.1',
    help='Frequency range as "min,max,step" (default: -5,5,0.1)'
)
@click.option(
    '-e', '--eta',
    type=float,
    default=0.01,
    help='Broadening parameter eta (default: 0.01)'
)
@click.option(
    '-T', '--temperature',
    type=float,
    default=None,
    help='Temperature for finite-T calculations'
)
@click.option(
    '--publication-bundle',
    type=click.Path(),
    default=None,
    help='Optional output directory for publication-bundle export metadata'
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
def spectrum(state, hamiltonian, operator, method, frequency_range, eta,
             temperature, publication_bundle, submit, label, description):
    """Calculate spectral properties.

    This command calculates various spectral functions and transport properties
    using different methods (zero-T spectra, finite-T spectra, Kubo, etc.).

    Examples:

        # Zero-temperature absorption spectrum
        verdi reno spectrum -s 123 -H 456 -o "a_dag" -m spectra_zero_t

        # Finite-temperature spectrum
        verdi reno spectrum -s 123 -H 456 -o "a" -m spectra_finite_t -T 300

        # Kubo transport
        verdi reno spectrum -s 123 -H 456 -o "j" -m kubo

        # Custom frequency range
        verdi reno spectrum -s 123 -H 456 -o "a" -f "-10,10,0.05"
    """
    from aiida_renormalizer.data.mps import MpsData
    from aiida_renormalizer.data.ttns import TTNSData
    from aiida_renormalizer.data.mpo import MpoData
    from aiida_renormalizer.data.ttno import TtnoData
    from aiida_renormalizer.data.model import ModelData
    from aiida_renormalizer.data.basis_tree import BasisTreeData

    # Load state
    echo.echo_info(f"Loading state (PK: {state})...")
    try:
        state_obj = load_node(state)
    except Exception as e:
        echo.echo_critical(f"Failed to load state {state}: {e}")

    # Determine tensor network type
    if isinstance(state_obj, MpsData):
        tn_type = 'mps'
        echo.echo_info("Detected MPS state")
    elif isinstance(state_obj, TTNSData):
        tn_type = 'ttn'
        echo.echo_info("Detected TTNS state")
    else:
        echo.echo_critical(
            f"Unsupported state type: {type(state_obj).__name__}. "
            "Must be MpsData or TTNSData."
        )

    # Load Hamiltonian
    echo.echo_info(f"Loading Hamiltonian (PK: {hamiltonian})...")
    try:
        hamiltonian_obj = load_node(hamiltonian)
    except Exception as e:
        echo.echo_critical(f"Failed to load Hamiltonian {hamiltonian}: {e}")

    # Parse frequency range
    try:
        freq_parts = frequency_range.split(',')
        if len(freq_parts) != 3:
            raise ValueError()
        freq_min = float(freq_parts[0])
        freq_max = float(freq_parts[1])
        freq_step = float(freq_parts[2])
    except (ValueError, AttributeError):
        echo.echo_critical(
            f"Invalid frequency range format: {frequency_range}. "
            "Expected format: 'min,max,step'"
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

    # Run spectrum calculation
    echo.echo_info(f"Running {method} calculation...")

    if tn_type == 'mps':
        # MPS spectrum calculation
        if method == 'spectra_zero_t':
            from aiida_renormalizer.calculations.spectra.spectra_zero_t import SpectraZeroTCalcJob

            # Need model
            try:
                model = state_obj.creator.outputs.model
            except (AttributeError, KeyError):
                echo.echo_critical(
                    "Cannot find model for MPS spectrum. "
                    "State must have model as input."
                )

            builder = SpectraZeroTCalcJob.get_builder()
            builder.model = model
            builder.mpo = hamiltonian_obj
            builder.mps = state_obj
            builder.operator = orm.Str(operator)
            builder.omega = orm.List([freq_min, freq_max, freq_step])
            builder.eta = orm.Float(eta)
            builder.code = code

        elif method == 'spectra_finite_t':
            from aiida_renormalizer.calculations.spectra.spectra_finite_t import SpectraFiniteTCalcJob

            if temperature is None:
                echo.echo_critical(
                    "Temperature required for finite-T spectrum. Use -T option."
                )

            try:
                model = state_obj.creator.outputs.model
            except (AttributeError, KeyError):
                echo.echo_critical(
                    "Cannot find model for MPS spectrum. "
                    "State must have model as input."
                )

            builder = SpectraFiniteTCalcJob.get_builder()
            builder.model = model
            builder.mpo = hamiltonian_obj
            builder.mps = state_obj
            builder.operator = orm.Str(operator)
            builder.omega = orm.List([freq_min, freq_max, freq_step])
            builder.eta = orm.Float(eta)
            builder.temperature = orm.Float(temperature)
            builder.code = code

        elif method == 'kubo':
            from aiida_renormalizer.calculations.spectra.kubo import KuboCalcJob

            try:
                model = state_obj.creator.outputs.model
            except (AttributeError, KeyError):
                echo.echo_critical(
                    "Cannot find model for MPS spectrum. "
                    "State must have model as input."
                )

            builder = KuboCalcJob.get_builder()
            builder.model = model
            builder.mpo = hamiltonian_obj
            builder.mps = state_obj
            builder.operator = orm.Str(operator)
            builder.omega = orm.List([freq_min, freq_max, freq_step])
            builder.eta = orm.Float(eta)
            builder.code = code

        elif method == 'correction_vector':
            from aiida_renormalizer.calculations.spectra.correction_vector import CorrectionVectorCalcJob

            try:
                model = state_obj.creator.outputs.model
            except (AttributeError, KeyError):
                echo.echo_critical(
                    "Cannot find model for MPS spectrum. "
                    "State must have model as input."
                )

            builder = CorrectionVectorCalcJob.get_builder()
            builder.model = model
            builder.mpo = hamiltonian_obj
            builder.mps = state_obj
            builder.operator = orm.Str(operator)
            builder.omega = orm.List([freq_min, freq_max, freq_step])
            builder.eta = orm.Float(eta)
            builder.code = code

        elif method == 'charge_diffusion':
            from aiida_renormalizer.calculations.spectra.charge_diffusion import ChargeDiffusionCalcJob

            try:
                model = state_obj.creator.outputs.model
            except (AttributeError, KeyError):
                echo.echo_critical(
                    "Cannot find model for MPS spectrum. "
                    "State must have model as input."
                )

            builder = ChargeDiffusionCalcJob.get_builder()
            builder.model = model
            builder.mpo = hamiltonian_obj
            builder.mps = state_obj
            builder.operator = orm.Str(operator)
            builder.omega = orm.List([freq_min, freq_max, freq_step])
            builder.eta = orm.Float(eta)
            builder.code = code

        elif method == 'spectral_function':
            from aiida_renormalizer.calculations.spectra.spectral_function import SpectralFunctionCalcJob

            try:
                model = state_obj.creator.outputs.model
            except (AttributeError, KeyError):
                echo.echo_critical(
                    "Cannot find model for MPS spectrum. "
                    "State must have model as input."
                )

            builder = SpectralFunctionCalcJob.get_builder()
            builder.model = model
            builder.mpo = hamiltonian_obj
            builder.mps = state_obj
            builder.operator = orm.Str(operator)
            builder.omega = orm.List([freq_min, freq_max, freq_step])
            builder.eta = orm.Float(eta)
            builder.code = code

        if label:
            builder.metadata.label = label
        if description:
            builder.metadata.description = description

        if submit:
            future = submit(builder)
            echo.echo_success(f"Submitted {method} calculation: {future.pk}")
            echo.echo_info(f"Check status with: verdi process show {future.pk}")
        else:
            result = run(builder)
            echo.echo_success("Spectrum calculation completed")
            if 'spectrum' in result:
                echo.echo_info(f"Spectrum data: {result['spectrum'].pk}")

    elif tn_type == 'ttn':
        # TTN spectrum calculation (limited methods available)
        echo.echo_critical(
            f"TTN spectrum calculations not yet supported for method '{method}'. "
            "Only MPS spectrum calculations are currently available."
        )
