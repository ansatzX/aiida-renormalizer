"""Integration tests for WorkChain entry points and imports."""
import pytest


def test_workchain_imports():
    """Test that all WorkChains can be imported."""
    from aiida_renormalizer.workchains import (
        RenoRestartWorkChain,
        TimeEvolutionWorkChain,
        GroundStateWorkChain,
        AbsorptionWorkChain,
        ConvergenceWorkChain,
        ThermalStateWorkChain,
        KuboTransportWorkChain,
        CustomPipelineWorkChain,
        ParameterSweepWorkChain,
        TemperatureSweepWorkChain,
        BondDimensionSweepWorkChain,
        FrequencySweepWorkChain,
        CorrectionVectorWorkChain,
        ChargeDiffusionWorkChain,
        BathMpoPipelineWorkChain,
        BathSpinBosonModelWorkChain,
        SpinBosonWorkChain,
        VibronicWorkChain,
    )

    assert RenoRestartWorkChain is not None
    assert TimeEvolutionWorkChain is not None
    assert GroundStateWorkChain is not None
    assert AbsorptionWorkChain is not None
    assert ConvergenceWorkChain is not None
    assert ThermalStateWorkChain is not None
    assert KuboTransportWorkChain is not None
    assert CustomPipelineWorkChain is not None
    assert ParameterSweepWorkChain is not None
    assert TemperatureSweepWorkChain is not None
    assert BondDimensionSweepWorkChain is not None
    assert FrequencySweepWorkChain is not None
    assert CorrectionVectorWorkChain is not None
    assert ChargeDiffusionWorkChain is not None
    assert BathMpoPipelineWorkChain is not None
    assert BathSpinBosonModelWorkChain is not None
    assert SpinBosonWorkChain is not None
    assert VibronicWorkChain is not None


def test_workchain_entry_points():
    """Test that all WorkChains are registered as entry points."""
    from aiida.plugins.entry_point import get_entry_point_names

    entry_points = get_entry_point_names("aiida.workflows")

    # Check that all WorkChains are registered
    expected = [
        "reno.restart",
        "reno.time_evolution",
        "reno.ground_state",
        "reno.absorption",
        "reno.convergence",
        "reno.thermal_state",
        "reno.kubo_transport",
        "reno.custom_pipeline",
        "reno.parameter_sweep",
        "reno.temperature_sweep",
        "reno.bond_dimension_sweep",
        "reno.frequency_sweep",
        "reno.correction_vector",
        "reno.charge_diffusion",
        "reno.bath_mpo_pipeline",
        "reno.bath_spin_boson_model",
        "reno.spin_boson",
        "reno.vibronic",
    ]

    for ep_name in expected:
        assert ep_name in entry_points, f"Entry point {ep_name} not found"


def test_workchain_entry_point_loading():
    """Test that WorkChains can be loaded via entry points."""
    from aiida.plugins import WorkflowFactory

    # Test loading each WorkChain
    RenoRestartWorkChain = WorkflowFactory("reno.restart")
    assert RenoRestartWorkChain is not None

    TimeEvolutionWorkChain = WorkflowFactory("reno.time_evolution")
    assert TimeEvolutionWorkChain is not None

    GroundStateWorkChain = WorkflowFactory("reno.ground_state")
    assert GroundStateWorkChain is not None

    AbsorptionWorkChain = WorkflowFactory("reno.absorption")
    assert AbsorptionWorkChain is not None

    ConvergenceWorkChain = WorkflowFactory("reno.convergence")
    assert ConvergenceWorkChain is not None


def test_restart_workchain_base_class():
    """Test RenoRestartWorkChain base class functionality."""
    from aiida_renormalizer.workchains.restart import RenoRestartWorkChain
    from aiida.engine import BaseRestartWorkChain

    # Check inheritance
    assert issubclass(RenoRestartWorkChain, BaseRestartWorkChain)

    # Check that it has the required methods
    assert hasattr(RenoRestartWorkChain, "setup")
    assert hasattr(RenoRestartWorkChain, "results")
    assert hasattr(RenoRestartWorkChain, "handle_not_converged")
    assert hasattr(RenoRestartWorkChain, "handle_physical_validation")


def test_time_evolution_workchain_outline():
    """Test TimeEvolutionWorkChain outline."""
    from aiida_renormalizer.workchains.time_evolution import TimeEvolutionWorkChain
    from aiida.engine import WorkChain

    assert issubclass(TimeEvolutionWorkChain, WorkChain)

    # Check outline methods
    assert hasattr(TimeEvolutionWorkChain, "setup")
    assert hasattr(TimeEvolutionWorkChain, "not_finished")
    assert hasattr(TimeEvolutionWorkChain, "run_checkpoint")
    assert hasattr(TimeEvolutionWorkChain, "inspect_checkpoint")
    assert hasattr(TimeEvolutionWorkChain, "finalize")


def test_ground_state_workchain_outline():
    """Test GroundStateWorkChain outline."""
    from aiida_renormalizer.workchains.ground_state import GroundStateWorkChain
    from aiida.engine import WorkChain

    assert issubclass(GroundStateWorkChain, WorkChain)

    # Check outline methods
    assert hasattr(GroundStateWorkChain, "use_dmrg")
    assert hasattr(GroundStateWorkChain, "use_imag_time")
    assert hasattr(GroundStateWorkChain, "run_dmrg")
    assert hasattr(GroundStateWorkChain, "run_imag_time")


def test_absorption_workchain_outline():
    """Test AbsorptionWorkChain outline."""
    from aiida_renormalizer.workchains.absorption import AbsorptionWorkChain
    from aiida.engine import WorkChain

    assert issubclass(AbsorptionWorkChain, WorkChain)

    # Check outline methods
    assert hasattr(AbsorptionWorkChain, "needs_ground_state")
    assert hasattr(AbsorptionWorkChain, "is_zero_temperature")
    assert hasattr(AbsorptionWorkChain, "run_zero_t_spectrum")
    assert hasattr(AbsorptionWorkChain, "run_finite_t_spectrum")


def test_convergence_workchain_outline():
    """Test ConvergenceWorkChain outline."""
    from aiida_renormalizer.workchains.convergence import ConvergenceWorkChain
    from aiida.engine import WorkChain

    assert issubclass(ConvergenceWorkChain, WorkChain)

    # Check outline methods
    assert hasattr(ConvergenceWorkChain, "not_converged")
    assert hasattr(ConvergenceWorkChain, "run_calculation")
    assert hasattr(ConvergenceWorkChain, "inspect_calculation")
    assert hasattr(ConvergenceWorkChain, "finalize")


def test_workchain_exit_codes():
    """Test that WorkChains define appropriate exit codes."""
    from aiida_renormalizer.workchains.time_evolution import TimeEvolutionWorkChain
    from aiida_renormalizer.workchains.ground_state import GroundStateWorkChain
    from aiida_renormalizer.workchains.absorption import AbsorptionWorkChain
    from aiida_renormalizer.workchains.convergence import ConvergenceWorkChain

    # TimeEvolutionWorkChain exit codes
    assert hasattr(TimeEvolutionWorkChain.exit_codes, "ERROR_ENERGY_DRIFT")
    assert hasattr(TimeEvolutionWorkChain.exit_codes, "ERROR_EVOLUTION_FAILED")

    # GroundStateWorkChain exit codes
    assert hasattr(GroundStateWorkChain.exit_codes, "ERROR_STRATEGY_NOT_SUPPORTED")
    assert hasattr(GroundStateWorkChain.exit_codes, "ERROR_NOT_CONVERGED")

    # AbsorptionWorkChain exit codes
    assert hasattr(AbsorptionWorkChain.exit_codes, "ERROR_GROUND_STATE_FAILED")
    assert hasattr(AbsorptionWorkChain.exit_codes, "ERROR_SPECTRUM_FAILED")

    # ConvergenceWorkChain exit codes
    assert hasattr(ConvergenceWorkChain.exit_codes, "ERROR_NO_CONVERGENCE")
    assert hasattr(ConvergenceWorkChain.exit_codes, "ERROR_CALCULATION_FAILED")
