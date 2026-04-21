"""WorkChain: bath modes + delta -> spin-boson ModelData via CalcJob composition."""

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


class SbmModelFromModesWorkChain(WorkChain):
    """Build a spin-boson model from explicit bath modes and effective delta."""

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("code", valid_type=orm.AbstractCode)
        spec.input("bath_modes", valid_type=orm.ArrayData)
        spec.input("delta_eff", valid_type=orm.Float)
        spec.input("epsilon", valid_type=orm.Float, default=lambda: orm.Float(0.0))
        spec.input("symbol_map", valid_type=orm.Dict, default=lambda: orm.Dict(dict={}))

        spec.output("model", valid_type=ModelData)
        spec.output("output_parameters", valid_type=orm.Dict)

        spec.exit_code(560, "ERROR_INVALID_INPUT", message="Invalid bath mode arrays")
        spec.exit_code(561, "ERROR_CALCJOB_FAILED", message="Bath model calcjob failed")
        spec.exit_code(
            562,
            "ERROR_SYMBOLIC_SPEC_FAILED",
            message="SBM symbolic spec calcjob failed",
        )
        spec.exit_code(
            563,
            "ERROR_MODEL_BUILD_FAILED",
            message="ModelFromSymbolicSpecCalcJob failed",
        )

        spec.outline(
            cls.run_bath_model_calcjob,
            cls.inspect_bath_model_calcjob,
            cls.run_symbolic_spec_calcjob,
            cls.inspect_symbolic_spec_calcjob,
            cls.run_model_build_calcjob,
            cls.inspect_model_build_calcjob,
            cls.finalize,
        )

    def run_bath_model_calcjob(self):
        omega_k_data = orm.ArrayData()
        omega_k_data.set_array("omega_k", self.inputs.bath_modes.get_array("omega_k"))
        c_j2_data = orm.ArrayData()
        c_j2_data.set_array("c_j2", self.inputs.bath_modes.get_array("c_j2"))
        return ToContext(
            bath_model_calc=self.submit(
                BathSpinBosonModelCalcJob,
                code=self.inputs.code,
                construction=orm.Str("discrete"),
                omega_k=omega_k_data,
                c_j2=c_j2_data,
                delta_eff=self.inputs.delta_eff,
                epsilon=self.inputs.epsilon,
            )
        )

    def inspect_bath_model_calcjob(self):
        calc = self.ctx.bath_model_calc
        if not calc.is_finished_ok:
            self.report(f"BathSpinBosonModelCalcJob failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_CALCJOB_FAILED

        self.ctx.bath_model_params = calc.outputs.output_parameters.get_dict()

    def run_symbolic_spec_calcjob(self):
        params = self.ctx.bath_model_params

        omega_k = params.get("omega_k")
        c_j2 = params.get("c_j2")
        if not isinstance(omega_k, list) or not isinstance(c_j2, list):
            self.report("Bath model output missing omega_k/c_j2 arrays")
            return self.exit_codes.ERROR_INVALID_INPUT

        omega_k_data = orm.ArrayData()
        omega_k_data.set_array("omega_k", np.asarray(omega_k, dtype=float))
        c_j2_data = orm.ArrayData()
        c_j2_data.set_array("c_j2", np.asarray(c_j2, dtype=float))

        return ToContext(
            symbolic_spec_calc=self.submit(
                SbmSymbolicSpecFromModesCalcJob,
                code=self.inputs.code,
                omega_k=omega_k_data,
                c_j2=c_j2_data,
                delta_eff=orm.Float(float(params.get("delta_eff", self.inputs.delta_eff.value))),
                epsilon=self.inputs.epsilon,
                symbol_map=self.inputs.symbol_map,
            )
        )

    def inspect_symbolic_spec_calcjob(self):
        calc = self.ctx.symbolic_spec_calc
        if not calc.is_finished_ok:
            self.report(f"SbmSymbolicSpecFromModesCalcJob failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_SYMBOLIC_SPEC_FAILED

        payload = calc.outputs.output_parameters.get_dict()
        symbolic_inputs = payload.get("symbolic_inputs")
        metadata = payload.get("metadata")

        if not isinstance(symbolic_inputs, dict):
            self.report("Missing or invalid symbolic_inputs in calcjob output")
            return self.exit_codes.ERROR_INVALID_INPUT
        if not isinstance(metadata, dict):
            self.report("Missing or invalid metadata in calcjob output")
            return self.exit_codes.ERROR_INVALID_INPUT

        self.ctx.symbolic_inputs = symbolic_inputs
        self.ctx.summary = metadata

    def run_model_build_calcjob(self):
        return ToContext(
            model_build_calc=self.submit(
                ModelFromSymbolicSpecCalcJob,
                code=self.inputs.code,
                symbolic_inputs=orm.Dict(dict=self.ctx.symbolic_inputs),
            )
        )

    def inspect_model_build_calcjob(self):
        calc = self.ctx.model_build_calc
        if not calc.is_finished_ok:
            self.report(f"ModelFromSymbolicSpecCalcJob failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_MODEL_BUILD_FAILED
        if "output_model" not in calc.outputs:
            self.report("ModelFromSymbolicSpecCalcJob missing output_model")
            return self.exit_codes.ERROR_MODEL_BUILD_FAILED
        self.ctx.model_data = calc.outputs.output_model

    def finalize(self):
        self.out("model", self.ctx.model_data)
        self.out("output_parameters", orm.Dict(dict=self.ctx.summary))
