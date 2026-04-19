# -*- coding: utf-8 -*-
"""Main verdi reno command group."""
import click
from aiida.cmdline.utils.decorators import with_dbenv


@click.group('reno')
def reno():
    """Commands for aiida-renormalizer tensor network calculations.

    This plugin provides high-level commands for running MPS and TTN calculations
    for quantum many-body systems using the Renormalizer library.
    """
    pass


# Import and add subcommands
from aiida_renormalizer.cli.cmd_ground_state import ground_state
from aiida_renormalizer.cli.cmd_evolve import evolve
from aiida_renormalizer.cli.cmd_spectrum import spectrum
from aiida_renormalizer.cli.cmd_convergence import convergence
from aiida_renormalizer.cli.cmd_status import status
from aiida_renormalizer.cli.cmd_bundle import bundle

reno.add_command(ground_state)
reno.add_command(evolve)
reno.add_command(spectrum)
reno.add_command(convergence)
reno.add_command(status)
reno.add_command(bundle)
