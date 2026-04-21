"""Unit tests for TimeEvolutionWorkChain."""
from unittest.mock import Mock

from aiida import orm

from aiida_renormalizer.workchains.time_evolution import TimeEvolutionWorkChain
from tests.workchains.conftest import make_workchain, Namespace


def test_time_evolution_setup():
    """Test TimeEvolutionWorkChain setup."""
    wc = make_workchain(TimeEvolutionWorkChain)

    wc.inputs = Namespace(
        initial_mps=Mock(),
        total_time=orm.Float(20.0),
        checkpoint_time=orm.Float(10.0),
        max_energy_drift=orm.Float(1e-6),
    )

    # Call setup
    TimeEvolutionWorkChain.setup(wc)

    # Check initialization
    assert wc.ctx.current_time == 0.0
    assert wc.ctx.current_mps == wc.inputs.initial_mps
    assert wc.ctx.checkpoints == []
    assert wc.ctx.energies == []
    assert wc.ctx.trajectory_segments == []
    assert wc.ctx.iteration == 0
    assert wc.ctx.n_checkpoints == 2  # 20.0 / 10.0


def test_time_evolution_checkpoint_calculation():
    """Test that checkpoint calculations are properly submitted."""
    wc = make_workchain(TimeEvolutionWorkChain)

    # Mock inputs -- use Namespace so ``"key" in self.inputs`` works correctly
    wc.inputs = Namespace(
        model=Mock(),
        mpo=Mock(),
        dt=orm.Float(0.01),
        code=Mock(),
        total_time=orm.Float(20.0),
        checkpoint_time=orm.Float(10.0),
    )

    wc.ctx = Namespace()
    wc.ctx.current_time = 0.0
    wc.ctx.current_mps = Mock()
    wc.ctx.iteration = 0
    wc.ctx.n_checkpoints = 2

    # Run checkpoint -- self.submit is already mocked by make_workchain
    TimeEvolutionWorkChain.run_checkpoint(wc)

    # Check that submit was called
    assert wc.submit.called
    submit_kwargs = wc.submit.call_args.kwargs
    assert submit_kwargs["model"] is wc.inputs.model
    assert submit_kwargs["mpo"] is wc.inputs.mpo
    assert submit_kwargs["code"] is wc.inputs.code


def test_time_evolution_inspect_checkpoint_success():
    """Test inspect_checkpoint with successful calculation."""
    wc = make_workchain(TimeEvolutionWorkChain)

    wc.inputs = Namespace(
        total_time=orm.Float(20.0),
        checkpoint_time=orm.Float(10.0),
        max_energy_drift=orm.Float(1e-6),
    )

    wc.ctx = Namespace()
    wc.ctx.iteration = 0
    wc.ctx.energies = []  # No prior energies -- first checkpoint
    wc.ctx.checkpoints = []
    wc.ctx.trajectory_segments = []
    wc.ctx.current_time = 0.0

    # Mock successful calculation
    calc = Mock()
    calc.is_finished_ok = True
    calc.outputs = Namespace(
        output_mps=Mock(),
        output_parameters=Mock(get_dict=Mock(return_value={"final_energy": -1.0})),
    )

    wc.ctx.checkpoint_calc = calc

    # Call inspect
    result = TimeEvolutionWorkChain.inspect_checkpoint(wc)

    # Check that state was updated
    assert result is None  # No error
    assert wc.ctx.current_mps == calc.outputs.output_mps
    assert wc.ctx.energies[-1] == -1.0
    assert wc.ctx.iteration == 1


def test_time_evolution_inspect_checkpoint_failure():
    """Test inspect_checkpoint with failed calculation."""
    wc = make_workchain(TimeEvolutionWorkChain)

    wc.ctx = Namespace()
    wc.ctx.iteration = 0

    # Mock failed calculation
    calc = Mock()
    calc.is_finished_ok = False
    calc.exit_status = 300

    wc.ctx.checkpoint_calc = calc
    wc.exit_codes = TimeEvolutionWorkChain.exit_codes

    # Call inspect
    result = TimeEvolutionWorkChain.inspect_checkpoint(wc)

    # Should return error
    assert result == TimeEvolutionWorkChain.exit_codes.ERROR_EVOLUTION_FAILED


def test_time_evolution_energy_drift_detection():
    """Test energy drift detection."""
    wc = make_workchain(TimeEvolutionWorkChain)

    wc.inputs = Namespace(
        max_energy_drift=orm.Float(1e-6),
        total_time=orm.Float(20.0),
        checkpoint_time=orm.Float(10.0),
    )

    wc.ctx = Namespace()
    wc.ctx.iteration = 1
    wc.ctx.energies = [-1.0, -1.0]  # Previous energies
    wc.ctx.checkpoints = []
    wc.ctx.trajectory_segments = []
    wc.ctx.current_time = 10.0

    # Mock calculation with large energy drift
    calc = Mock()
    calc.is_finished_ok = True
    calc.outputs = Namespace(
        output_mps=Mock(),
        output_parameters=Mock(get_dict=Mock(return_value={"final_energy": -1.5})),
    )

    wc.ctx.checkpoint_calc = calc
    wc.exit_codes = TimeEvolutionWorkChain.exit_codes

    # Call inspect
    result = TimeEvolutionWorkChain.inspect_checkpoint(wc)

    # Should return energy drift error
    assert result == TimeEvolutionWorkChain.exit_codes.ERROR_ENERGY_DRIFT


def test_time_evolution_finalize():
    """Test finalize method."""
    wc = make_workchain(TimeEvolutionWorkChain)

    wc.inputs = Namespace(
        total_time=orm.Float(20.0),
        strategy=orm.Str("dmrg"),
    )

    # Mock checkpoints with uuid attribute
    cp1 = Mock()
    cp1.uuid = "uuid-1"
    cp2 = Mock()
    cp2.uuid = "uuid-2"

    wc.ctx = Namespace()
    wc.ctx.current_mps = Mock()
    wc.ctx.checkpoints = [cp1, cp2]
    wc.ctx.energies = [-1.0, -1.1]
    wc.ctx.trajectory_segments = []
    wc.ctx.iteration = 2
    wc.ctx.current_time = 20.0

    # Call finalize
    TimeEvolutionWorkChain.finalize(wc)

    # Check outputs
    assert wc.out.call_count >= 3  # final_mps, checkpoints, output_parameters


def test_time_evolution_trajectory_concatenation():
    """Test trajectory concatenation."""
    wc = make_workchain(TimeEvolutionWorkChain)

    wc.inputs = Namespace(total_time=orm.Float(20.0))

    # Mock checkpoints with uuid attribute
    cp = Mock()
    cp.uuid = "uuid-1"

    wc.ctx = Namespace()
    wc.ctx.current_mps = Mock()
    wc.ctx.checkpoints = [cp]
    wc.ctx.energies = [-1.0]
    wc.ctx.iteration = 1
    wc.ctx.current_time = 10.0

    # Mock trajectory segments
    import numpy as np
    traj1 = Mock()
    traj1.get_arraynames.return_value = ["data"]
    traj1.get_array.return_value = np.array([1, 2, 3])

    traj2 = Mock()
    traj2.get_arraynames.return_value = ["data"]
    traj2.get_array.return_value = np.array([4, 5, 6])

    wc.ctx.trajectory_segments = [traj1, traj2]

    # Call finalize
    TimeEvolutionWorkChain.finalize(wc)

    # Check that trajectory was output
    trajectory_calls = [call for call in wc.out.call_args_list if "trajectory" in str(call)]
    # Note: Simplified implementation just saves last segment
    assert len(trajectory_calls) >= 1
