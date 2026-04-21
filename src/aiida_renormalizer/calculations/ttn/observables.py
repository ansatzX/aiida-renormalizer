"""CalcJobs for computing TTN observables."""
from __future__ import annotations

import os
import tempfile
from typing import List

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import BasisTreeData, TTNSData, TTNOData


class TTNSExpectationCalcJob(RenoBaseCalcJob):
    """Compute expectation value <ttns|ttno|ttns>.

    Corresponds to Reno API: ttns.expectation(ttno)

    Inputs:
        basis_tree: BasisTreeData - Tree topology and basis
        ttns: TTNSData - TTN state
        ttno: TTNOData - TTN operator

    Outputs:
        output_parameters: Dict - Expectation value
    """

    _template_name = "ttns_expectation_driver.py.jinja"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        # Additional inputs
        spec.input(
            "basis_tree",
            valid_type=BasisTreeData,
            help="Tree topology and basis grouping",
        )
        spec.input("ttns", valid_type=TTNSData, help="TTN state")
        spec.input("ttno", valid_type=TTNOData, help="TTN operator")

    def _write_input_files(self, folder) -> None:
        """Write input files."""
        super()._write_input_files(folder)

        # Write basis tree
        basis_tree_data = self.inputs.basis_tree
        basis_tree = basis_tree_data.load_basis_tree()

        with tempfile.TemporaryDirectory() as tmpdir:
            basis_tree_path = os.path.join(tmpdir, "basis_tree")
            basis_tree.dump(basis_tree_path)
            actual = basis_tree_path + ".npz" if os.path.exists(basis_tree_path + ".npz") else basis_tree_path
            with open(actual, "rb") as src:
                with folder.open("input_basis_tree.npz", "wb") as dst:
                    dst.write(src.read())

        # Write TTNS
        ttns_data = self.inputs.ttns
        TTNS_obj = ttns_data.load_ttns(basis_tree_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            ttns_path = os.path.join(tmpdir, "ttns")
            TTNS_obj.dump(ttns_path)
            actual = ttns_path + ".npz" if os.path.exists(ttns_path + ".npz") else ttns_path
            with open(actual, "rb") as src:
                with folder.open("initial_ttns.npz", "wb") as dst:
                    dst.write(src.read())

        # Write TTNO
        ttno_data = self.inputs.ttno
        TTNO_obj = ttno_data.load_ttno(basis_tree_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            ttno_path = os.path.join(tmpdir, "ttno")
            TTNO_obj.dump(ttno_path)
            actual = ttno_path + ".npz" if os.path.exists(ttno_path + ".npz") else ttno_path
            with open(actual, "rb") as src:
                with folder.open("initial_ttno.npz", "wb") as dst:
                    dst.write(src.read())

    def _get_retrieve_list(self) -> list[str]:
        """Get list of files to retrieve."""
        return [
            "output_parameters.json",
            "aiida.out",
            "aiida.err",
        ]


class TTNSEntropyCalcJob(RenoBaseCalcJob):
    """Compute von Neumann entropy at each node.

    Computes the entanglement entropy by partitioning the tree at each node.

    Inputs:
        basis_tree: BasisTreeData - Tree topology and basis
        ttns: TTNSData - TTN state

    Outputs:
        output_parameters: Dict - Entropy values at each node
    """

    _template_name = "ttns_entropy_driver.py.jinja"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        # Additional inputs
        spec.input(
            "basis_tree",
            valid_type=BasisTreeData,
            help="Tree topology and basis grouping",
        )
        spec.input("ttns", valid_type=TTNSData, help="TTN state")

    def _write_input_files(self, folder) -> None:
        """Write input files."""
        super()._write_input_files(folder)

        # Write basis tree
        basis_tree_data = self.inputs.basis_tree
        basis_tree = basis_tree_data.load_basis_tree()

        with tempfile.TemporaryDirectory() as tmpdir:
            basis_tree_path = os.path.join(tmpdir, "basis_tree")
            basis_tree.dump(basis_tree_path)
            actual = basis_tree_path + ".npz" if os.path.exists(basis_tree_path + ".npz") else basis_tree_path
            with open(actual, "rb") as src:
                with folder.open("input_basis_tree.npz", "wb") as dst:
                    dst.write(src.read())

        # Write TTNS
        ttns_data = self.inputs.ttns
        TTNS_obj = ttns_data.load_ttns(basis_tree_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            ttns_path = os.path.join(tmpdir, "ttns")
            TTNS_obj.dump(ttns_path)
            actual = ttns_path + ".npz" if os.path.exists(ttns_path + ".npz") else ttns_path
            with open(actual, "rb") as src:
                with folder.open("initial_ttns.npz", "wb") as dst:
                    dst.write(src.read())

    def _get_retrieve_list(self) -> list[str]:
        """Get list of files to retrieve."""
        return [
            "output_parameters.json",
            "aiida.out",
            "aiida.err",
        ]


class TTNSMutualInfoCalcJob(RenoBaseCalcJob):
    """Compute mutual information between nodes.

    Computes the mutual information between pairs of nodes in the tree.

    Inputs:
        basis_tree: BasisTreeData - Tree topology and basis
        ttns: TTNSData - TTN state

    Outputs:
        output_parameters: Dict - Mutual information matrix
    """

    _template_name = "ttns_mutual_info_driver.py.jinja"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        # Additional inputs
        spec.input(
            "basis_tree",
            valid_type=BasisTreeData,
            help="Tree topology and basis grouping",
        )
        spec.input("ttns", valid_type=TTNSData, help="TTN state")

    def _write_input_files(self, folder) -> None:
        """Write input files."""
        super()._write_input_files(folder)

        # Write basis tree
        basis_tree_data = self.inputs.basis_tree
        basis_tree = basis_tree_data.load_basis_tree()

        with tempfile.TemporaryDirectory() as tmpdir:
            basis_tree_path = os.path.join(tmpdir, "basis_tree")
            basis_tree.dump(basis_tree_path)
            actual = basis_tree_path + ".npz" if os.path.exists(basis_tree_path + ".npz") else basis_tree_path
            with open(actual, "rb") as src:
                with folder.open("input_basis_tree.npz", "wb") as dst:
                    dst.write(src.read())

        # Write TTNS
        ttns_data = self.inputs.ttns
        TTNS_obj = ttns_data.load_ttns(basis_tree_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            ttns_path = os.path.join(tmpdir, "ttns")
            TTNS_obj.dump(ttns_path)
            actual = ttns_path + ".npz" if os.path.exists(ttns_path + ".npz") else ttns_path
            with open(actual, "rb") as src:
                with folder.open("initial_ttns.npz", "wb") as dst:
                    dst.write(src.read())

    def _get_retrieve_list(self) -> list[str]:
        """Get list of files to retrieve."""
        return [
            "output_parameters.json",
            "aiida.out",
            "aiida.err",
        ]


class TTNSRdmCalcJob(RenoBaseCalcJob):
    """Compute reduced density matrix for specific nodes.

    Inputs:
        basis_tree: BasisTreeData - Tree topology and basis
        ttns: TTNSData - TTN state
        node_indices: List - Indices of nodes to compute RDM for

    Outputs:
        output_parameters: Dict - RDM information
        output_rdm: orm.ArrayData - Reduced density matrices
    """

    _template_name = "ttns_rdm_driver.py.jinja"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        # Additional inputs
        spec.input(
            "basis_tree",
            valid_type=BasisTreeData,
            help="Tree topology and basis grouping",
        )
        spec.input("ttns", valid_type=TTNSData, help="TTN state")
        spec.input(
            "node_indices",
            valid_type=orm.List,
            help="Indices of nodes to compute RDM for",
        )

        # Outputs
        spec.output(
            "output_rdm",
            valid_type=orm.ArrayData,
            help="Reduced density matrices",
        )

    def _write_input_files(self, folder) -> None:
        """Write input files."""
        import json

        super()._write_input_files(folder)

        # Write basis tree
        basis_tree_data = self.inputs.basis_tree
        basis_tree = basis_tree_data.load_basis_tree()

        with tempfile.TemporaryDirectory() as tmpdir:
            basis_tree_path = os.path.join(tmpdir, "basis_tree")
            basis_tree.dump(basis_tree_path)
            actual = basis_tree_path + ".npz" if os.path.exists(basis_tree_path + ".npz") else basis_tree_path
            with open(actual, "rb") as src:
                with folder.open("input_basis_tree.npz", "wb") as dst:
                    dst.write(src.read())

        # Write TTNS
        ttns_data = self.inputs.ttns
        TTNS_obj = ttns_data.load_ttns(basis_tree_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            ttns_path = os.path.join(tmpdir, "ttns")
            TTNS_obj.dump(ttns_path)
            actual = ttns_path + ".npz" if os.path.exists(ttns_path + ".npz") else ttns_path
            with open(actual, "rb") as src:
                with folder.open("initial_ttns.npz", "wb") as dst:
                    dst.write(src.read())

        # Write node indices
        node_indices = self.inputs.node_indices.get_list()
        with folder.open("input_node_indices.json", "w") as f:
            json.dump({"node_indices": node_indices}, f)

    def _get_retrieve_list(self) -> list[str]:
        """Get list of files to retrieve."""
        return [
            "output_parameters.json",
            "output_rdm.npz",
            "aiida.out",
            "aiida.err",
        ]
