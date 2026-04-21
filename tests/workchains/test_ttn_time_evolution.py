"""Unit tests for TTNTimeEvolutionWorkChain."""
import pytest
from unittest.mock import Mock, MagicMock, patch

from aiida import orm

from aiida_renormalizer.workchains.ttn_time_evolution import TTNTimeEvolutionWorkChain
from tests.workchains.conftest import make_workchain, Namespace


def test_ttn_time_evolution_setup():
    """Test setup method."""
    wc = make_workchain(TTNTimeEvolutionWorkChain)

    wc.inputs = Namespace(
        total_time=orm.Float(20.0),
        checkpoint_time=orm.Float(5.0),
        dt=orm.Float(0.5),
        initial_ttns=Mock(),
    )

    # Setup
    TTNTimeEvolutionWorkChain.setup(wc)

    # Check initialization
    assert wc.ctx.current_time == 0.0
    assert wc.ctx.current_step == 0
    assert wc.ctx.total_steps == 40
    assert wc.ctx.steps_per_checkpoint == 10
    assert wc.ctx.checkpoints == []
    assert wc.ctx.energies == []
    assert wc.ctx.n_checkpoints == 4  # 20.0 / 5.0


def test_ttn_time_evolution_not_finished():
    """Test not_finished method."""
    wc = make_workchain(TTNTimeEvolutionWorkChain)

    wc.inputs = Namespace(total_time=orm.Float(20.0), dt=orm.Float(0.5))
    wc.ctx = Namespace()
    wc.ctx.current_step = 15
    wc.ctx.total_steps = 20

    # Should continue
    assert TTNTimeEvolutionWorkChain.not_finished(wc) is True

    wc.ctx.current_step = 20
    # Should stop
    assert TTNTimeEvolutionWorkChain.not_finished(wc) is False


def test_ttn_time_evolution_run_checkpoint():
    """Test run_checkpoint method."""
    wc = make_workchain(TTNTimeEvolutionWorkChain)

    # Use Namespace so ``"key" in self.inputs`` works
    wc.inputs = Namespace(
        basis_tree=Mock(),
        ttno=Mock(),
        config=Mock(),
        code=Mock(),
        dt=orm.Float(0.5),
        total_time=orm.Float(20.0),
        checkpoint_time=orm.Float(5.0),
    )

    wc.ctx = Namespace()
    wc.ctx.current_time = 5.0
    wc.ctx.current_step = 10
    wc.ctx.total_steps = 40
    wc.ctx.steps_per_checkpoint = 10
    wc.ctx.current_ttns = Mock()
    wc.ctx.iteration = 0
    wc.ctx.n_checkpoints = 4

    # Run checkpoint -- self.submit is already mocked by make_workchain
    result = TTNTimeEvolutionWorkChain.run_checkpoint(wc)

    assert wc.submit.called
    assert result is not None


def test_ttn_time_evolution_inspect_checkpoint_success():
    """Test inspect_checkpoint with successful calculation."""
    wc = make_workchain(TTNTimeEvolutionWorkChain)

    # Mock successful calculation
    calc = Mock()
    calc.is_finished_ok = True
    calc.outputs = Namespace(
        output_ttns=Mock(),
        output_parameters=Mock(
            get_dict=Mock(return_value={"final_energy": -1.5})
        ),
    )

    wc.inputs = Namespace(
        total_time=orm.Float(20.0),
        checkpoint_time=orm.Float(5.0),
        dt=orm.Float(0.5),
        max_energy_drift=orm.Float(1e-6),
    )

    wc.ctx = Namespace()
    wc.ctx.checkpoint_calc = calc
    wc.ctx.current_time = 10.0
    wc.ctx.current_step = 20
    wc.ctx.total_steps = 40
    wc.ctx.steps_per_checkpoint = 10
    wc.ctx.checkpoints = []
    wc.ctx.energies = []
    wc.ctx.trajectory_segments = []
    wc.ctx.iteration = 0

    # Inspect
    result = TTNTimeEvolutionWorkChain.inspect_checkpoint(wc)

    # Check results
    assert result is None  # No error
    assert wc.ctx.current_ttns == calc.outputs.output_ttns
    assert len(wc.ctx.checkpoints) == 1
    assert len(wc.ctx.energies) == 1


def test_ttn_time_evolution_inspect_checkpoint_failure():
    """Test inspect_checkpoint with failed calculation."""
    wc = make_workchain(TTNTimeEvolutionWorkChain)

    calc = Mock()
    calc.is_finished_ok = False
    calc.exit_status = 340

    wc.ctx = Namespace()
    wc.ctx.checkpoint_calc = calc
    wc.exit_codes = TTNTimeEvolutionWorkChain.exit_codes

    result = TTNTimeEvolutionWorkChain.inspect_checkpoint(wc)

    assert result == TTNTimeEvolutionWorkChain.exit_codes.ERROR_EVOLUTION_FAILED


def test_ttn_time_evolution_energy_drift_detection():
    """Test energy drift detection."""
    wc = make_workchain(TTNTimeEvolutionWorkChain)

    # Mock calculation
    calc = Mock()
    calc.is_finished_ok = True
    calc.outputs = Namespace(
        output_ttns=Mock(),
        output_parameters=Mock(
            get_dict=Mock(return_value={"final_energy": -1.5})
        ),
    )

    wc.inputs = Namespace(
        total_time=orm.Float(20.0),
        checkpoint_time=orm.Float(5.0),
        dt=orm.Float(0.5),
        max_energy_drift=orm.Float(1e-6),
    )

    wc.ctx = Namespace()
    wc.ctx.checkpoint_calc = calc
    wc.ctx.current_time = 15.0
    wc.ctx.current_step = 30
    wc.ctx.total_steps = 40
    wc.ctx.steps_per_checkpoint = 10
    wc.ctx.checkpoints = []
    wc.ctx.energies = [-1.0]  # Previous energy
    wc.ctx.trajectory_segments = []
    wc.ctx.iteration = 1

    wc.exit_codes = TTNTimeEvolutionWorkChain.exit_codes

    # Large drift should trigger error
    result = TTNTimeEvolutionWorkChain.inspect_checkpoint(wc)

    assert result == TTNTimeEvolutionWorkChain.exit_codes.ERROR_ENERGY_DRIFT


def test_ttn_time_evolution_finalize():
    """Test finalize method."""
    wc = make_workchain(TTNTimeEvolutionWorkChain)

    wc.inputs = Namespace(total_time=orm.Float(20.0), dt=orm.Float(0.5))

    wc.ctx = Namespace()
    wc.ctx.current_ttns = Mock()
    wc.ctx.checkpoints = [Mock(), Mock()]
    wc.ctx.energies = [-1.0, -1.5]
    wc.ctx.iteration = 2
    wc.ctx.current_time = 20.0
    wc.ctx.total_steps = 40
    wc.ctx.trajectory_segments = []  # No trajectory segments

    # Finalize
    TTNTimeEvolutionWorkChain.finalize(wc)

    # Check outputs
    assert wc.out.call_count >= 3  # final_ttns, checkpoints, output_parameters


def test_ttn_time_evolution_trajectory_concatenation():
    """Test trajectory concatenation."""
    import numpy as np

    wc = make_workchain(TTNTimeEvolutionWorkChain)

    wc.inputs = Namespace(total_time=orm.Float(20.0), dt=orm.Float(0.5))

    # Mock trajectory segments
    mock_traj = Mock()
    mock_traj.get_arraynames.return_value = ["times", "values"]
    mock_traj.get_array.return_value = np.array([1, 2, 3])

    wc.ctx = Namespace()
    wc.ctx.current_ttns = Mock()
    wc.ctx.checkpoints = [Mock()]
    wc.ctx.energies = [-1.0]
    wc.ctx.iteration = 1
    wc.ctx.current_time = 20.0
    wc.ctx.total_steps = 40
    wc.ctx.trajectory_segments = [mock_traj]

    # Finalize
    TTNTimeEvolutionWorkChain.finalize(wc)

    # Check that trajectory was output
    trajectory_calls = [call for call in wc.out.call_args_list if call[0][0] == "trajectory"]
    assert len(trajectory_calls) == 1


def test_ttn_time_evolution_trajectory_concatenation_multi_segment():
    """Trajectory concatenation should offset times and drop duplicated boundaries."""
    import numpy as np

    wc = make_workchain(TTNTimeEvolutionWorkChain)
    wc.inputs = Namespace(total_time=orm.Float(20.0), dt=orm.Float(0.5))

    seg1 = Mock()
    seg1.get_arraynames.return_value = ["times", "energies"]
    seg1.get_array.side_effect = lambda name: {
        "times": np.array([0.0, 0.5, 1.0]),
        "energies": np.array([1.0, 0.9, 0.8]),
    }[name]

    seg2 = Mock()
    seg2.get_arraynames.return_value = ["times", "energies"]
    seg2.get_array.side_effect = lambda name: {
        "times": np.array([0.0, 0.5, 1.0]),
        "energies": np.array([0.8, 0.7, 0.6]),
    }[name]

    wc.ctx = Namespace()
    wc.ctx.current_ttns = Mock()
    wc.ctx.checkpoints = [Mock(), Mock()]
    wc.ctx.energies = [-1.0, -1.1]
    wc.ctx.iteration = 2
    wc.ctx.current_time = 2.0
    wc.ctx.total_steps = 4
    wc.ctx.trajectory_segments = [(0.0, seg1), (1.0, seg2)]

    TTNTimeEvolutionWorkChain.finalize(wc)

    trajectory_calls = [call for call in wc.out.call_args_list if call[0][0] == "trajectory"]
    assert len(trajectory_calls) == 1
    traj_node = trajectory_calls[0][0][1]
    np.testing.assert_allclose(traj_node.get_array("times"), np.array([0.0, 0.5, 1.0, 1.5, 2.0]))
    np.testing.assert_allclose(traj_node.get_array("energies"), np.array([1.0, 0.9, 0.8, 0.7, 0.6]))


def test_ttn_time_evolution_checkpoint_time_calculation():
    """Test checkpoint time calculation for non-integer divisions."""
    wc = make_workchain(TTNTimeEvolutionWorkChain)

    wc.inputs = Namespace(
        total_time=orm.Float(23.0),  # Non-integer multiple
        checkpoint_time=orm.Float(5.0),
        dt=orm.Float(0.3),
        initial_ttns=Mock(),
    )

    # Setup
    result = TTNTimeEvolutionWorkChain.setup(wc)

    # Should fail because total_time/checkpoint_time are not integer multiples of dt
    assert result == TTNTimeEvolutionWorkChain.exit_codes.ERROR_TIME_GRID_MISMATCH
