"""Tests for TTN CalcJobs."""
from __future__ import annotations

import pytest
from aiida import orm
from aiida.common import AttributeDict

from aiida_renormalizer.data import BasisTreeData, TTNSData, TtnoData, ConfigData


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
def sho_ttns(sho_basis_tree):
    """Create a random TTNS for testing."""
    from renormalizer.tn.tree import TTNS

    return TTNS.random(sho_basis_tree, qntot=0, m_max=10)


@pytest.fixture
def sho_ttno(sho_basis_tree, sho_ham):
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
def ttns_data(sho_ttns, basis_tree_data, artifact_storage_base):
    """Create and store TTNSData node."""
    node = TTNSData.from_ttns(
        sho_ttns,
        basis_tree_data,
        storage_backend="posix",
        storage_base=str(artifact_storage_base),
        relative_path="ttn/calcjob_input.npz",
    )
    node.store()
    return node


@pytest.fixture
def ttno_data(sho_ttno, basis_tree_data):
    """Create and store TtnoData node."""
    node = TtnoData.from_ttno(sho_ttno, basis_tree_data)
    node.store()
    return node


class TestOptimizeTtnsCalcJob:
    """Tests for OptimizeTtnsCalcJob."""

    def test_calcjob_defined(self):
        """OptimizeTtnsCalcJob should be properly defined."""
        from aiida_renormalizer.calculations.ttn.optimize_ttns import OptimizeTtnsCalcJob

        assert hasattr(OptimizeTtnsCalcJob, "_template_name")
        assert OptimizeTtnsCalcJob._template_name == "ttn_optimize_driver.py.jinja"

    def test_inputs_outputs(self):
        """OptimizeTtnsCalcJob should define correct inputs/outputs."""
        from aiida_renormalizer.calculations.ttn.optimize_ttns import OptimizeTtnsCalcJob
        from unittest.mock import Mock

        spec = Mock()
        spec.input = Mock()
        spec.output = Mock()
        spec.exit_code = Mock()
        spec.options = {}

        OptimizeTtnsCalcJob.define(spec)

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
        """OptimizeTtnsCalcJob should provide correct template context."""
        from aiida_renormalizer.calculations.ttn.optimize_ttns import OptimizeTtnsCalcJob

        calcjob = _make_calcjob(OptimizeTtnsCalcJob, {
            "basis_tree": basis_tree_data,
            "ttno": ttno_data,
        })

        context = calcjob._get_template_context()

        assert "has_initial_ttns" in context
        assert context["has_initial_ttns"] is False

    def test_template_context_with_initial_ttns(self, basis_tree_data, ttno_data, ttns_data):
        """OptimizeTtnsCalcJob should handle initial TTNS input."""
        from aiida_renormalizer.calculations.ttn.optimize_ttns import OptimizeTtnsCalcJob

        calcjob = _make_calcjob(OptimizeTtnsCalcJob, {
            "basis_tree": basis_tree_data,
            "ttno": ttno_data,
            "initial_ttns": ttns_data,
        })

        context = calcjob._get_template_context()

        assert context["has_initial_ttns"] is True

    def test_template_exists(self):
        """OptimizeTtnsCalcJob template file should exist."""
        from pathlib import Path

        import aiida_renormalizer

        pkg_dir = Path(aiida_renormalizer.__file__).parent
        template_path = pkg_dir / "templates" / "ttn_optimize_driver.py.jinja"

        assert template_path.exists(), f"Template not found: {template_path}"


class TestTtnsEvolveCalcJob:
    """Tests for TtnsEvolveCalcJob."""

    def test_calcjob_defined(self):
        """TtnsEvolveCalcJob should be properly defined."""
        from aiida_renormalizer.calculations.ttn.ttns_evolve import TtnsEvolveCalcJob

        assert hasattr(TtnsEvolveCalcJob, "_template_name")
        assert TtnsEvolveCalcJob._template_name == "ttns_evolve_driver.py.jinja"

    def test_inputs_outputs(self):
        """TtnsEvolveCalcJob should define correct inputs/outputs."""
        from aiida_renormalizer.calculations.ttn.ttns_evolve import TtnsEvolveCalcJob
        from unittest.mock import Mock

        spec = Mock()
        spec.input = Mock()
        spec.output = Mock()
        spec.exit_code = Mock()
        spec.options = {}

        TtnsEvolveCalcJob.define(spec)

        # Check inputs
        input_calls = [call for call in spec.input.call_args_list]
        input_names = [call[0][0] for call in input_calls]

        assert "basis_tree" in input_names
        assert "initial_ttns" in input_names
        assert "ttno" in input_names
        assert "config" in input_names

        # Check outputs
        output_calls = [call for call in spec.output.call_args_list]
        output_names = [call[0][0] for call in output_calls]

        assert "output_ttns" in output_names

    def test_template_exists(self):
        """TtnsEvolveCalcJob template file should exist."""
        from pathlib import Path

        import aiida_renormalizer

        pkg_dir = Path(aiida_renormalizer.__file__).parent
        template_path = pkg_dir / "templates" / "ttns_evolve_driver.py.jinja"

        assert template_path.exists(), f"Template not found: {template_path}"


class TestTtnsObservableCalcJobs:
    """Tests for TTN observable CalcJobs."""

    def test_expectation_calcjob_defined(self):
        """TtnsExpectationCalcJob should be properly defined."""
        from aiida_renormalizer.calculations.ttn.observables import TtnsExpectationCalcJob

        assert hasattr(TtnsExpectationCalcJob, "_template_name")
        assert TtnsExpectationCalcJob._template_name == "ttns_expectation_driver.py.jinja"

    def test_entropy_calcjob_defined(self):
        """TtnsEntropyCalcJob should be properly defined."""
        from aiida_renormalizer.calculations.ttn.observables import TtnsEntropyCalcJob

        assert hasattr(TtnsEntropyCalcJob, "_template_name")
        assert TtnsEntropyCalcJob._template_name == "ttns_entropy_driver.py.jinja"

    def test_mutual_info_calcjob_defined(self):
        """TtnsMutualInfoCalcJob should be properly defined."""
        from aiida_renormalizer.calculations.ttn.observables import TtnsMutualInfoCalcJob

        assert hasattr(TtnsMutualInfoCalcJob, "_template_name")
        assert TtnsMutualInfoCalcJob._template_name == "ttns_mutual_info_driver.py.jinja"

    def test_rdm_calcjob_defined(self):
        """TtnsRdmCalcJob should be properly defined."""
        from aiida_renormalizer.calculations.ttn.observables import TtnsRdmCalcJob

        assert hasattr(TtnsRdmCalcJob, "_template_name")
        assert TtnsRdmCalcJob._template_name == "ttns_rdm_driver.py.jinja"

    def test_expectation_inputs_outputs(self):
        """TtnsExpectationCalcJob should define correct inputs/outputs."""
        from aiida_renormalizer.calculations.ttn.observables import TtnsExpectationCalcJob
        from unittest.mock import Mock

        spec = Mock()
        spec.input = Mock()
        spec.output = Mock()
        spec.exit_code = Mock()
        spec.options = {}

        TtnsExpectationCalcJob.define(spec)

        # Check inputs
        input_calls = [call for call in spec.input.call_args_list]
        input_names = [call[0][0] for call in input_calls]

        assert "basis_tree" in input_names
        assert "ttns" in input_names
        assert "ttno" in input_names

    def test_rdm_inputs_outputs(self):
        """TtnsRdmCalcJob should define correct inputs/outputs."""
        from aiida_renormalizer.calculations.ttn.observables import TtnsRdmCalcJob
        from unittest.mock import Mock

        spec = Mock()
        spec.input = Mock()
        spec.output = Mock()
        spec.exit_code = Mock()
        spec.options = {}

        TtnsRdmCalcJob.define(spec)

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

    def test_templates_exist(self):
        """All observable CalcJob template files should exist."""
        from pathlib import Path

        import aiida_renormalizer

        pkg_dir = Path(aiida_renormalizer.__file__).parent
        templates_dir = pkg_dir / "templates"

        templates = [
            "ttns_expectation_driver.py.jinja",
            "ttns_entropy_driver.py.jinja",
            "ttns_mutual_info_driver.py.jinja",
            "ttns_rdm_driver.py.jinja",
        ]

        for template in templates:
            template_path = templates_dir / template
            assert template_path.exists(), f"Template not found: {template_path}"
