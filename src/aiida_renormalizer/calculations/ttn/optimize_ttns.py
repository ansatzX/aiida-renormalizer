"""CalcJob for TTN ground state optimization."""
from __future__ import annotations

import os
import tempfile

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import BasisTreeData, TTNSData, TTNOData


class OptimizeTTNSCalcJob(RenoBaseCalcJob):
    """TTN variational optimization for ground state.

    Corresponds to Reno API: optimize_ttns(ttns, ttno, procedure)

    Inputs:
        basis_tree: BasisTreeData - Tree topology and basis
        ttno: TTNOData - Hamiltonian operator
        initial_ttns: TTNSData (optional) - Initial guess, random if not provided
        config: ConfigData - OptimizeConfig with convergence criteria

    Outputs:
        output_ttns: TTNSData - Optimized ground state
        output_parameters: Dict - Energy trajectory, convergence info
    """

    _template_name = "ttn_optimize_driver.py.jinja"

    @classmethod
    def define(cls, spec: CalcJobProcessSpec) -> None:
        super().define(spec)

        # Additional inputs
        spec.input(
            "basis_tree",
            valid_type=BasisTreeData,
            help="Tree topology and basis grouping",
        )
        spec.input(
            "ttno",
            valid_type=TTNOData,
            help="Hamiltonian TTNO for optimization",
        )
        spec.input(
            "initial_ttns",
            valid_type=TTNSData,
            required=False,
            help="Initial TTNS guess (random if not provided)",
        )

        # Outputs
        spec.output(
            "output_ttns",
            valid_type=TTNSData,
            help="Optimized ground state TTNS",
        )

        # Exit codes
        spec.exit_code(
            300,
            "ERROR_NOT_CONVERGED",
            message="TTN optimization did not converge",
        )

    def _get_template_context(self) -> dict:
        """Provide context for Jinja2 template rendering."""
        context = super()._get_template_context()
        context["has_initial_ttns"] = "initial_ttns" in self.inputs
        return context

    def _write_input_files(self, folder) -> None:
        """Write input files for TTN optimization calculation."""
        import json

        # Write model (from ttno)
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

        # Write TTNO
        ttno_data = self.inputs.ttno
        TTNO_obj = ttno_data.load_ttno(basis_tree_data)

        with tempfile.TemporaryDirectory() as tmpdir:
            ttno_path = os.path.join(tmpdir, "ttno")
            TTNO_obj.dump(ttno_path)
            actual = ttno_path + ".npz" if os.path.exists(ttno_path + ".npz") else ttno_path
            with open(actual, "rb") as src:
                with folder.open("input_ttno.npz", "wb") as dst:
                    dst.write(src.read())

        # Write initial TTNS (if provided)
        if "initial_ttns" in self.inputs:
            ttns_data = self.inputs.initial_ttns
            TTNS_obj = ttns_data.load_ttns(basis_tree_data)

            with tempfile.TemporaryDirectory() as tmpdir:
                ttns_path = os.path.join(tmpdir, "ttns")
                TTNS_obj.dump(ttns_path)
                actual = ttns_path + ".npz" if os.path.exists(ttns_path + ".npz") else ttns_path
                with open(actual, "rb") as src:
                    with folder.open("initial_ttns.npz", "wb") as dst:
                        dst.write(src.read())

    def _get_retrieve_list(self) -> list[str]:
        """Get list of files to retrieve after calculation."""
        return [
            "output_parameters.json",
            "output_ttns.npz",
            "aiida.out",
            "aiida.err",
        ]
