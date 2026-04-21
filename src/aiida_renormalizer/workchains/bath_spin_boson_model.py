"""WorkChain wrapper: compose bath/model CalcJobs for spin-boson model construction."""

from __future__ import annotations

import numpy as np
from aiida import orm
from aiida.engine import ToContext, WorkChain

from aiida_renormalizer.calculations.bath import (
    BathSpinBosonModelCalcJob,
    SbmSymbolicSpecFromModesCalcJob,
)
from aiida_renormalizer.calculations.basic.model_from_symbolic import ModelFromSymbolicSpecCalcJob
from aiida_renormalizer.data import ModelData


class BathSpinBosonModelWorkChain(WorkChain):
    """High-level SBM model assembly from bath parameters using CalcJob composition.

    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("code", valid_type=orm.AbstractCode, help="Code for CalcJob execution")
        spec.input(
            "construction",
            valid_type=orm.Str,
            default=lambda: orm.Str("param2mollist"),
            help="Construction mode: param2mollist | discrete",
        )
        spec.input("alpha", valid_type=orm.Float, required=False, help="Ohmic strength alpha")
        spec.input(
            "raw_delta", valid_type=orm.Float, required=False, help="Bare tunneling / gap parameter"
        )
        spec.input("omega_c", valid_type=orm.Float, required=False, help="Bath cutoff frequency")
        spec.input(
            "renormalization_p",
            valid_type=orm.Float,
            required=False,
            help="Adiabatic renormalization p",
        )
        spec.input("n_phonons", valid_type=orm.Int, required=False, help="Number of bath modes")
        spec.input(
            "epsilon",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.0),
            help="Spin bias epsilon",
        )
        spec.input(
            "delta_eff",
            valid_type=orm.Float,
            required=False,
            help="Effective delta for discrete mode path",
        )
        spec.input("dipole", valid_type=orm.Float, required=False, help="Optional dipole moment")
        spec.input(
            "omega_k", valid_type=orm.ArrayData, required=False, help="Discrete bath frequencies"
        )
        spec.input(
            "c_j2", valid_type=orm.ArrayData, required=False, help="Discrete bath c_j^2 values"
        )
        spec.input(
            "spectral_density_type", valid_type=orm.Str, default=lambda: orm.Str("ohmic_exp")
        )
        spec.input(
            "beta",
            valid_type=orm.Float,
            default=lambda: orm.Float(0.7),
            help="Reserved for compatibility",
        )
        spec.input("tree_type", valid_type=orm.Str, default=lambda: orm.Str("binary"))
        spec.input("m_max", valid_type=orm.Int, default=lambda: orm.Int(16))
        spec.input("symbol_map", valid_type=orm.Dict, default=lambda: orm.Dict(dict={}))
        spec.input("min_nbas", valid_type=orm.Float, default=lambda: orm.Float(4.0))
        spec.input("nbas_prefactor", valid_type=orm.Float, default=lambda: orm.Float(16.0))
        spec.input("vib_prefix", valid_type=orm.Str, default=lambda: orm.Str("v_"))

        spec.output("bath_model", valid_type=ModelData, help="Constructed SpinBoson ModelData")
        spec.output(
            "bath_modes", valid_type=orm.ArrayData, help="Bath mode frequencies and couplings"
        )
        spec.output("output_parameters", valid_type=orm.Dict, help="Model construction summary")

        spec.exit_code(480, "ERROR_INVALID_BATH_MODEL_INPUT", message="Invalid bath-model inputs")
        spec.exit_code(481, "ERROR_BATH_MODEL_CALCJOB_FAILED", message="Bath model CalcJob failed")
        spec.exit_code(
            482,
            "ERROR_SYMBOLIC_SPEC_CALCJOB_FAILED",
            message="SBM symbolic-spec CalcJob failed",
        )
        spec.exit_code(
            483,
            "ERROR_MODEL_BUILD_CALCJOB_FAILED",
            message="Model build CalcJob failed",
        )

        spec.outline(
            cls.setup,
            cls.run_bath_model_calcjob,
            cls.inspect_bath_model_calcjob,
            cls.run_symbolic_spec_calcjob,
            cls.inspect_symbolic_spec_calcjob,
            cls.run_model_build_calcjob,
            cls.inspect_model_build_calcjob,
            cls.finalize,
        )

    def setup(self):
        self.report("Starting bath SpinBoson model construction")

        construction = self.inputs.construction.value
        if construction not in ("param2mollist", "discrete"):
            self.report(f"Invalid construction mode: {construction}")
            return self.exit_codes.ERROR_INVALID_BATH_MODEL_INPUT

        if construction == "param2mollist":
            required = ["alpha", "raw_delta", "omega_c", "renormalization_p", "n_phonons"]
            missing = [name for name in required if name not in self.inputs]
            if missing:
                self.report(f"Missing param2mollist inputs: {missing}")
                return self.exit_codes.ERROR_INVALID_BATH_MODEL_INPUT
        else:
            if "omega_k" not in self.inputs or "c_j2" not in self.inputs:
                self.report("Discrete construction requires omega_k and c_j2")
                return self.exit_codes.ERROR_INVALID_BATH_MODEL_INPUT

    def run_bath_model_calcjob(self):
        inputs = {
            "code": self.inputs.code,
            "construction": self.inputs.construction,
            "epsilon": self.inputs.epsilon,
            "spectral_density_type": self.inputs.spectral_density_type,
            "beta": self.inputs.beta,
        }
        passthrough = (
            "alpha",
            "raw_delta",
            "omega_c",
            "renormalization_p",
            "n_phonons",
            "delta_eff",
            "dipole",
            "omega_k",
            "c_j2",
        )
        for key in passthrough:
            if key in self.inputs:
                inputs[key] = getattr(self.inputs, key)
        return ToContext(bath_calc=self.submit(BathSpinBosonModelCalcJob, **inputs))

    def inspect_bath_model_calcjob(self):
        calc = self.ctx.bath_calc
        if not calc.is_finished_ok:
            self.report(f"BathSpinBosonModelCalcJob failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_BATH_MODEL_CALCJOB_FAILED
        self.ctx.bath_params = calc.outputs.output_parameters.get_dict()

    def run_symbolic_spec_calcjob(self):
        params = self.ctx.bath_params

        omega_k = params.get("omega_k")
        c_j2 = params.get("c_j2")
        if not isinstance(omega_k, list) or not isinstance(c_j2, list):
            self.report("Bath model output missing omega_k/c_j2 arrays")
            return self.exit_codes.ERROR_INVALID_BATH_MODEL_INPUT

        omega_k_data = orm.ArrayData()
        omega_k_data.set_array("omega_k", np.asarray(omega_k, dtype=float))
        c_j2_data = orm.ArrayData()
        c_j2_data.set_array("c_j2", np.asarray(c_j2, dtype=float))

        return ToContext(
            symbolic_calc=self.submit(
                SbmSymbolicSpecFromModesCalcJob,
                code=self.inputs.code,
                omega_k=omega_k_data,
                c_j2=c_j2_data,
                delta_eff=orm.Float(float(params.get("delta_eff", 0.0))),
                epsilon=self.inputs.epsilon,
                tree_type=self.inputs.tree_type,
                m_max=self.inputs.m_max,
                symbol_map=self.inputs.symbol_map,
                min_nbas=self.inputs.min_nbas,
                nbas_prefactor=self.inputs.nbas_prefactor,
                vib_prefix=self.inputs.vib_prefix,
            )
        )

    def inspect_symbolic_spec_calcjob(self):
        calc = self.ctx.symbolic_calc
        if not calc.is_finished_ok:
            self.report(f"SbmSymbolicSpecFromModesCalcJob failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_SYMBOLIC_SPEC_CALCJOB_FAILED

        payload = calc.outputs.output_parameters.get_dict()
        symbolic_inputs = payload.get("symbolic_inputs")
        metadata = payload.get("metadata")

        if not isinstance(symbolic_inputs, dict):
            self.report("Missing or invalid symbolic_inputs in symbolic-spec calcjob output")
            return self.exit_codes.ERROR_INVALID_BATH_MODEL_INPUT
        if not isinstance(metadata, dict):
            self.report("Missing or invalid metadata in symbolic-spec calcjob output")
            return self.exit_codes.ERROR_INVALID_BATH_MODEL_INPUT

        self.ctx.symbolic_inputs = symbolic_inputs
        self.ctx.symbolic_metadata = metadata

    def run_model_build_calcjob(self):
        return ToContext(
            model_calc=self.submit(
                ModelFromSymbolicSpecCalcJob,
                code=self.inputs.code,
                symbolic_inputs=orm.Dict(dict=self.ctx.symbolic_inputs),
            )
        )

    def inspect_model_build_calcjob(self):
        calc = self.ctx.model_calc
        if not calc.is_finished_ok:
            self.report(f"ModelFromSymbolicSpecCalcJob failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_MODEL_BUILD_CALCJOB_FAILED
        if "output_model" not in calc.outputs:
            self.report("ModelFromSymbolicSpecCalcJob missing output_model")
            return self.exit_codes.ERROR_MODEL_BUILD_CALCJOB_FAILED
        self.ctx.model_data = calc.outputs.output_model

    def finalize(self):
        """Expose assembled model and bath payloads."""
        params = self.ctx.bath_params
        omega_k = np.asarray(params["omega_k"], dtype=float)
        c_j2 = np.asarray(params["c_j2"], dtype=float)

        bath_modes = orm.ArrayData()
        bath_modes.set_array("omega_k", omega_k)
        bath_modes.set_array("c_j2", c_j2)
        bath_modes.set_array(
            "displacement",
            np.asarray(params.get("displacement", []), dtype=float),
        )

        output_params = dict(params)
        output_params["symbolic_metadata"] = self.ctx.symbolic_metadata

        self.out("bath_model", self.ctx.model_data)
        self.out("bath_modes", bath_modes)
        self.out("output_parameters", orm.Dict(dict=output_params))

        self.report("Bath SpinBoson model construction completed")
