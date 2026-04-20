# -*- coding: utf-8 -*-
""" verdi reno convergence command."""
import click
from aiida.cmdline.utils.decorators import with_dbenv
from aiida.cmdline.utils import echo
from aiida import orm
from aiida.engine import submit, run

from aiida_renormalizer.cli.code_resolver import find_default_script_code


@click.command('convergence')
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
    '-t', '--sweep-type',
    type=click.Choice(['bond_dimension', 'temperature', 'frequency', 'parameter']),
    default='bond_dimension',
    help='Type of convergence sweep (default: bond_dimension)'
)
@click.option(
    '-r', '--range',
    type=str,
    required=True,
    help='Sweep range (format depends on sweep type)'
)
@click.option(
    '-e', '--energy-convergence',
    type=float,
    default=1e-6,
    help='Energy convergence threshold (default: 1e-6)'
)
@click.option(
    '-C', '--code',
    type=str,
    default=None,
    help='AiiDA code label or ID to use for calculations'
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
def convergence(model, basis, sweep_type, range, energy_convergence, code,
                submit, label, description):
    """Run convergence study.

    This command runs a convergence study by sweeping over bond dimensions,
    temperatures, frequencies, or other parameters.

    Examples:

        # Bond dimension convergence study
        verdi reno convergence -m model.toml -b basis.toml -t bond_dimension -r "10,20,50,100"

        # Temperature sweep
        verdi reno convergence -m model.toml -b basis.toml -t temperature -r "0,100,200,300"

        # Frequency sweep
        verdi reno convergence -m model.toml -b basis.toml -t frequency -r "-5:5:0.5"

        # Submit to daemon
        verdi reno convergence -m model.toml -b basis.toml -t bond_dimension -r "10,20,50" --submit
    """
    import json
    from pathlib import Path

    try:
        import tomllib  # py>=3.11
    except ModuleNotFoundError:  # py<=3.10
        import tomli as tomllib
    from aiida_renormalizer.data.model import ModelData
    from aiida_renormalizer.data.basis import BasisSetData
    from aiida_renormalizer.data.mpo import MpoData
    from aiida_renormalizer.calculations.basic.build_mpo import BuildMpoCalcJob

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

    # Get code
    if code:
        try:
            code_obj = orm.load_code(code)
        except Exception as e:
            echo.echo_critical(f"Failed to load code '{code}': {e}")
    else:
        code_obj = find_default_script_code()
        if code_obj is not None:
            echo.echo_info(f"Using code: {code_obj.label}")
        else:
            echo.echo_critical(
                "No healthy code found for plugin 'reno.script'. "
                "Please specify -C or configure a valid code."
            )

    # Create ModelData and BasisSetData
    echo.echo_info("Creating model and basis set...")
    model_data = ModelData.from_dict(model_config)
    basis_data = BasisSetData.from_dict(basis_config)

    # Build MPO
    echo.echo_info("Building MPO...")
    builder = BuildMpoCalcJob.get_builder()
    builder.model = model_data
    builder.basis = basis_data
    builder.code = code_obj
    mpo_result = run(builder)
    mpo = mpo_result['mpo']

    # Parse sweep range based on type
    if sweep_type == 'bond_dimension':
        # Format: comma-separated list of integers
        try:
            sweep_values = [int(x.strip()) for x in range.split(',')]
        except ValueError:
            echo.echo_critical(
                f"Invalid bond dimension range: {range}. "
                "Expected comma-separated integers, e.g., '10,20,50,100'"
            )

        from aiida_renormalizer.workchains.sweep import BondDimensionSweepWorkChain

        echo.echo_info(f"Running bond dimension convergence study: {sweep_values}")
        inputs = {
            'model': model_data,
            'mpo': mpo,
            'bond_dimensions': orm.List(sweep_values),
            'energy_convergence': orm.Float(energy_convergence),
            'code': code_obj,
        }

        if label:
            inputs['metadata'] = {'label': label, 'description': description}

        if submit:
            future = submit(BondDimensionSweepWorkChain, **inputs)
            echo.echo_success(f"Submitted BondDimensionSweepWorkChain: {future.pk}")
            echo.echo_info(f"Check status with: verdi process show {future.pk}")
        else:
            result = run(BondDimensionSweepWorkChain, **inputs)
            echo.echo_success("Convergence study completed")
            if 'energies' in result:
                echo.echo_info(f"Energies: {result['energies'].get_array('energies')}")
            if 'convergence_data' in result:
                echo.echo_info(f"Convergence data: {result['convergence_data'].pk}")

    elif sweep_type == 'temperature':
        # Format: comma-separated list of floats
        try:
            sweep_values = [float(x.strip()) for x in range.split(',')]
        except ValueError:
            echo.echo_critical(
                f"Invalid temperature range: {range}. "
                "Expected comma-separated floats, e.g., '0,100,200,300'"
            )

        from aiida_renormalizer.workchains.sweep import TemperatureSweepWorkChain

        echo.echo_info(f"Running temperature sweep: {sweep_values}")
        inputs = {
            'model': model_data,
            'mpo': mpo,
            'temperatures': orm.List(sweep_values),
            'code': code_obj,
        }

        if label:
            inputs['metadata'] = {'label': label, 'description': description}

        if submit:
            future = submit(TemperatureSweepWorkChain, **inputs)
            echo.echo_success(f"Submitted TemperatureSweepWorkChain: {future.pk}")
            echo.echo_info(f"Check status with: verdi process show {future.pk}")
        else:
            result = run(TemperatureSweepWorkChain, **inputs)
            echo.echo_success("Temperature sweep completed")
            if 'thermal_states' in result:
                echo.echo_info(f"Thermal states: {result['thermal_states']}")

    elif sweep_type == 'frequency':
        # Format: "min:max:step" or comma-separated
        if ':' in range:
            try:
                parts = range.split(':')
                if len(parts) != 3:
                    raise ValueError()
                freq_min = float(parts[0])
                freq_max = float(parts[1])
                freq_step = float(parts[2])
                import numpy as np
                sweep_values = np.arange(freq_min, freq_max + freq_step, freq_step).tolist()
            except ValueError:
                echo.echo_critical(
                    f"Invalid frequency range: {range}. "
                    "Expected format 'min:max:step', e.g., '-5:5:0.5'"
                )
        else:
            try:
                sweep_values = [float(x.strip()) for x in range.split(',')]
            except ValueError:
                echo.echo_critical(
                    f"Invalid frequency range: {range}. "
                    "Expected comma-separated floats or 'min:max:step' format"
                )

        from aiida_renormalizer.workchains.sweep import FrequencySweepWorkChain

        echo.echo_info(f"Running frequency sweep: {len(sweep_values)} points")
        inputs = {
            'model': model_data,
            'mpo': mpo,
            'frequencies': orm.List(sweep_values),
            'code': code_obj,
        }

        if label:
            inputs['metadata'] = {'label': label, 'description': description}

        if submit:
            future = submit(FrequencySweepWorkChain, **inputs)
            echo.echo_success(f"Submitted FrequencySweepWorkChain: {future.pk}")
            echo.echo_info(f"Check status with: verdi process show {future.pk}")
        else:
            result = run(FrequencySweepWorkChain, **inputs)
            echo.echo_success("Frequency sweep completed")
            if 'spectra' in result:
                echo.echo_info(f"Spectra data: {result['spectra'].pk}")

    elif sweep_type == 'parameter':
        # Generic parameter sweep
        # Format: "param_name:value1,value2,value3"
        try:
            parts = range.split(':', 1)
            if len(parts) != 2:
                raise ValueError()
            param_name = parts[0].strip()
            param_values = [float(x.strip()) for x in parts[1].split(',')]
        except ValueError:
            echo.echo_critical(
                f"Invalid parameter range: {range}. "
                "Expected format 'param_name:value1,value2,value3', "
                "e.g., 'J:0.1,0.5,1.0'"
            )

        from aiida_renormalizer.workchains.sweep import ParameterSweepWorkChain

        echo.echo_info(
            f"Running parameter sweep: {param_name} = {param_values}"
        )
        inputs = {
            'model': model_data,
            'mpo': mpo,
            'parameter_name': orm.Str(param_name),
            'parameter_values': orm.List(param_values),
            'energy_convergence': orm.Float(energy_convergence),
            'code': code_obj,
        }

        if label:
            inputs['metadata'] = {'label': label, 'description': description}

        if submit:
            future = submit(ParameterSweepWorkChain, **inputs)
            echo.echo_success(f"Submitted ParameterSweepWorkChain: {future.pk}")
            echo.echo_info(f"Check status with: verdi process show {future.pk}")
        else:
            result = run(ParameterSweepWorkChain, **inputs)
            echo.echo_success("Parameter sweep completed")
            if 'results' in result:
                echo.echo_info(f"Results: {result['results'].pk}")
