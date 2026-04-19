"""LEGO Observable CalcJobs for modular measurements."""
from aiida_renormalizer.calculations.lego.occupations import ComputeOccupationsCalcJob
from aiida_renormalizer.calculations.lego.msd import ComputeMsdCalcJob

__all__ = [
    'ComputeOccupationsCalcJob',
    'ComputeMsdCalcJob',
]
