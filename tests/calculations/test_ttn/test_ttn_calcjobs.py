"""Tests for TTN CalcJobs."""
from __future__ import annotations

import pytest
from aiida import orm

from aiida_renormalizer.data import BasisTreeData, TTNSData, TTNOData


def _make_calcjob(cls, inputs_dict):
    """Create a CalcJob instance without triggering plumpy Process.__init__."""
    from plumpy.utils import AttributesFrozendict
    calcjob = object.__new__(cls)
    calcjob._parsed_inputs = AttributesFrozendict(inputs_dict)
    return calcjob


@pytest.fixture
def sho_basis_tree(sho_basis):
    """Create a binary BasisTree for testing."""
    from renormalizer.tn.treebase import BasisTree

    return BasisTree.binary(sho_basis)


@pytest.fixture
def sho_TTNS(sho_basis_tree):
    """Create a random TTNS for testing."""
    from renormalizer.tn.tree import TTNS

    return TTNS.random(sho_basis_tree, qntot=0, m_max=10)


@pytest.fixture
def sho_TTNO(sho_basis_tree, sho_ham):
    """Create a TTNO from Hamiltonian for testing."""
    from renormalizer.tn.tree import TTNO

    return TTNO(sho_basis_tree, sho_ham)


@pytest.fixture
def basis_tree_data(sho_basis_tree):
    """Create and store BasisTreeData node."""
    node = BasisTreeData.from_basis_tree(sho_basis_tree)
    node.store()
    return node


@pytest.fixture
def ttns_data(sho_TTNS, basis_tree_data, artifact_storage_base):
    """Create and store TTNSData node."""
    node = TTNSData.from_ttns(
        sho_TTNS,
        basis_tree_data,
        storage_backend="posix",
        storage_base=str(artifact_storage_base),
        relative_path="ttn/calcjob_input.npz",
    )
    node.store()
    return node


@pytest.fixture
def ttno_data(sho_TTNO, basis_tree_data):
    """Create and store TTNOData node."""
    node = TTNOData.from_ttno(sho_TTNO, basis_tree_data)
    node.store()
    return node


class TestOptimizeTTNSCalcJob:
    """Tests for OptimizeTTNSCalcJob."""

    def test_inputs_outputs(self):
        """OptimizeTTNSCalcJob should define correct inputs/outputs."""
        from aiida_renormalizer.calculations.ttn.optimize_ttns import OptimizeTTNSCalcJob
        from unittest.mock import Mock

        spec = Mock()
        spec.input = Mock()
        spec.output = Mock()
        spec.exit_code = Mock()
        spec.options = {}

        OptimizeTTNSCalcJob.define(spec)

        # Check inputs
        input_calls = [call for call in spec.input.call_args_list]
        input_names = [call[0][0] for call in input_calls]

        assert "basis_tree" in input_names
        assert "ttno" in input_names
        assert "initial_ttns" in input_names

        # Check outputs
        output_calls = [call for call in spec.output.call_args_list]
        output_names = [call[0][0] for call in output_calls]

        assert "output_ttns" in output_names

    def test_template_context(self, basis_tree_data, ttno_data):
        """OptimizeTTNSCalcJob should provide correct template context."""
        from aiida_renormalizer.calculations.ttn.optimize_ttns import OptimizeTTNSCalcJob

        calcjob = _make_calcjob(OptimizeTTNSCalcJob, {
            "basis_tree": basis_tree_data,
            "ttno": ttno_data,
        })

        context = calcjob._get_template_context()

        assert "has_initial_ttns" in context
        assert context["has_initial_ttns"] is False

    def test_template_context_with_initial_ttns(self, basis_tree_data, ttno_data, ttns_data):
        """OptimizeTTNSCalcJob should handle initial TTNS input."""
        from aiida_renormalizer.calculations.ttn.optimize_ttns import OptimizeTTNSCalcJob

        calcjob = _make_calcjob(OptimizeTTNSCalcJob, {
            "basis_tree": basis_tree_data,
            "ttno": ttno_data,
            "initial_ttns": ttns_data,
        })

        context = calcjob._get_template_context()

        assert context["has_initial_ttns"] is True


class TestTTNSEvolveCalcJob:
    """Tests for TTNSEvolveCalcJob."""

    def test_inputs_outputs(self):
        """TTNSEvolveCalcJob should define correct inputs/outputs."""
        from aiida_renormalizer.calculations.ttn.ttns_evolve import TTNSEvolveCalcJob
        from unittest.mock import Mock

        spec = Mock()
        spec.input = Mock()
        spec.output = Mock()
        spec.exit_code = Mock()
        spec.options = {}

        TTNSEvolveCalcJob.define(spec)

        # Check inputs
        input_calls = [call for call in spec.input.call_args_list]
        input_names = [call[0][0] for call in input_calls]

        assert "basis_tree" in input_names
        assert "initial_ttns" in input_names
        assert "ttno" in input_names
        assert "config" in input_names
        assert "dt" in input_names
        assert "nsteps" in input_names

        # Check outputs
        output_calls = [call for call in spec.output.call_args_list]
        output_names = [call[0][0] for call in output_calls]

        assert "output_ttns" in output_names


class TestTTNSSymbolicEvolveCalcJob:
    """Tests for TTNSSymbolicEvolveCalcJob."""

    def test_inputs_outputs(self):
        from aiida_renormalizer.calculations.ttn.ttns_symbolic_evolve import (
            TTNSSymbolicEvolveCalcJob,
        )
        from unittest.mock import Mock

        spec = Mock()
        spec.input = Mock()
        spec.output = Mock()
        spec.exit_code = Mock()
        spec.options = {}

        TTNSSymbolicEvolveCalcJob.define(spec)

        input_names = [call[0][0] for call in spec.input.call_args_list]
        assert "symbolic_inputs" in input_names
        assert "dt" in input_names
        assert "nsteps" in input_names
        assert "method" in input_names

        output_names = [call[0][0] for call in spec.output.call_args_list]
        assert "output_ttns" in output_names
        assert "output_basis_tree" in output_names

    def test_symbolic_validator(self):
        from aiida_renormalizer.calculations.ttn.ttns_symbolic_evolve import (
            TTNSSymbolicEvolveCalcJob,
        )

        good = {
            "basis": [
                {"kind": "half_spin", "dof": "spin"},
                {"kind": "sho", "dof": "v0", "omega": 1.0, "nbas": 4},
            ],
            "hamiltonian": [{"symbol": "sigma_x", "dofs": "spin"}],
            "tree_type": "binary",
            "m_max": 16,
        }
        bad = {
            "basis": [{"kind": "sho", "dof": "v0"}],  # missing omega/nbas
            "hamiltonian": [],
        }

        assert TTNSSymbolicEvolveCalcJob._validate_symbolic_dict(good) is None
        assert TTNSSymbolicEvolveCalcJob._validate_symbolic_dict(bad) is not None


class TestTTNSymbolicModelCalcJob:
    """Tests for TTNSymbolicModelCalcJob."""

    def test_inputs_outputs(self):
        from aiida_renormalizer.calculations.ttn.symbolic_model import TTNSymbolicModelCalcJob
        from unittest.mock import Mock

        spec = Mock()
        spec.input = Mock()
        spec.output = Mock()
        spec.exit_code = Mock()
        spec.options = {}

        TTNSymbolicModelCalcJob.define(spec)

        input_names = [call[0][0] for call in spec.input.call_args_list]
        assert "alpha" in input_names
        assert "s_exponent" in input_names
        assert "omega_c" in input_names
        assert "n_modes" in input_names
        assert "process" in input_names

        output_names = [call[0][0] for call in spec.output.call_args_list]
        assert "output_parameters" in output_names
        assert "output_basis_tree" in output_names

    def test_process_validator(self):
        from aiida_renormalizer.calculations.ttn.symbolic_model import TTNSymbolicModelCalcJob

        good = orm.List(list=["build_sdf", "discretize_bath", "build_symbolic_hamiltonian"])
        bad = orm.List(list=["unknown", "build_symbolic_hamiltonian"])
        missing = orm.List(list=["build_sdf", "discretize_bath"])

        assert TTNSymbolicModelCalcJob._validate_process(good, None) is None
        assert TTNSymbolicModelCalcJob._validate_process(bad, None) is not None
        assert TTNSymbolicModelCalcJob._validate_process(missing, None) is not None


class TestTTNSObservableCalcJobs:
    """Tests for TTN observable CalcJobs."""

    def test_expectation_inputs_outputs(self):
        """TTNSExpectationCalcJob should define correct inputs/outputs."""
        from aiida_renormalizer.calculations.ttn.observables import TTNSExpectationCalcJob
        from unittest.mock import Mock

        spec = Mock()
        spec.input = Mock()
        spec.output = Mock()
        spec.exit_code = Mock()
        spec.options = {}

        TTNSExpectationCalcJob.define(spec)

        # Check inputs
        input_calls = [call for call in spec.input.call_args_list]
        input_names = [call[0][0] for call in input_calls]

        assert "basis_tree" in input_names
        assert "ttns" in input_names
        assert "ttno" in input_names

    def test_rdm_inputs_outputs(self):
        """TTNSRdmCalcJob should define correct inputs/outputs."""
        from aiida_renormalizer.calculations.ttn.observables import TTNSRdmCalcJob
        from unittest.mock import Mock

        spec = Mock()
        spec.input = Mock()
        spec.output = Mock()
        spec.exit_code = Mock()
        spec.options = {}

        TTNSRdmCalcJob.define(spec)

        # Check inputs
        input_calls = [call for call in spec.input.call_args_list]
        input_names = [call[0][0] for call in input_calls]

        assert "basis_tree" in input_names
        assert "ttns" in input_names
        assert "node_indices" in input_names

        # Check outputs
        output_calls = [call for call in spec.output.call_args_list]
        output_names = [call[0][0] for call in output_calls]

        assert "output_rdm" in output_names
