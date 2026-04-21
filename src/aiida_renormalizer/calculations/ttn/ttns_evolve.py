"""CalcJob for TTN time evolution."""
from __future__ import annotations

import os
import pickle
import tempfile

from aiida import orm
from aiida.engine import CalcJobProcessSpec

from aiida_renormalizer.calculations.base import RenoBaseCalcJob
from aiida_renormalizer.data import BasisTreeData, TTNSData, TTNOData, ConfigData


class TTNSEvolveCalcJob(RenoBaseCalcJob):
    """TTN time evolution.

    Corresponds to Reno API: ttns.evolve(ttno, tau) or evolve_tdvp_* functions

    Inputs:
        basis_tree: BasisTreeData - Tree topology and basis
        initial_ttns: TTNSData - Initial TTNS state
        ttno: TTNOData - Hamiltonian operator
        config: ConfigData - EvolveConfig with evolution parameters

    Outputs:
        output_ttns: TTNSData - Evolved TTNS state
        output_parameters: Dict - Energy trajectory, observables
    """

    _template_name = "ttns_evolve_driver.py.jinja"

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
            "initial_ttns",
            valid_type=TTNSData,
            help="Initial TTNS state to evolve",
        )
        spec.input(
            "ttno",
            valid_type=TTNOData,
            help="Hamiltonian TTNO for evolution",
        )
        spec.input(
            "config",
            valid_type=ConfigData,
            required=False,
            help="EvolveConfig with method and integration settings",
        )
        spec.input("dt", valid_type=orm.Float, help="Time step for each evolution step")
        spec.input("nsteps", valid_type=orm.Int, help="Number of evolution steps")

        # Outputs
        spec.output(
            "output_ttns",
            valid_type=TTNSData,
            help="Evolved TTNS state",
        )

        # Exit codes
        spec.exit_code(
            300,
            "ERROR_EVOLUTION_FAILED",
            message="TTN time evolution failed",
        )

    def _write_input_files(self, folder) -> None:
        """Write input files for TTN evolution calculation."""
        import json

        # Write model (from ttno)
        super()._write_input_files(folder)

        # Write basis tree cache (fast path: reuse BasisTreeData serialized payload)
        basis_tree_data = self.inputs.basis_tree
        with folder.open("input_basis_tree.pkl", "wb") as dst:
            if not basis_tree_data.write_cached_pickle(dst):
                dst.write(
                    pickle.dumps(
                        basis_tree_data.load_basis_tree(),
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
                )

        # Write initial TTNS
        ttns_data = self.inputs.initial_ttns
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
                with folder.open("input_ttno.npz", "wb") as dst:
                    dst.write(src.read())

        with folder.open("input_evolution_params.json", "w") as f:
            json.dump(
                {
                    "dt": self.inputs.dt.value,
                    "nsteps": self.inputs.nsteps.value,
                },
                f,
                indent=2,
            )

    def _get_retrieve_list(self) -> list[str]:
        """Get list of files to retrieve after calculation."""
        return [
            "output_parameters.json",
            "output_ttns.npz",
            "trajectory.npz",
            "aiida.out",
            "aiida.err",
        ]
