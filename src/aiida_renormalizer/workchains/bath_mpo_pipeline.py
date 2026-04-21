"""WorkChain for bath spectral density to MPO-coefficient pipeline."""
from __future__ import annotations

import numpy as np

from aiida import orm
from aiida.engine import WorkChain, ToContext, if_

from aiida_renormalizer.calculations.bath import (
    BathSpectralDensityCalcJob,
    BathDiscretizationCalcJob,
    BathToMPOCoeffCalcJob,
)
from aiida_renormalizer.data import ModelData


class BathMPOPipelineWorkChain(WorkChain):
    """Construct bath MPO coefficients from continuous or discrete spectrum data.

    Branching behavior:
    - Continuous SDF inputs -> generate J(omega) -> discretize -> map coeffs
    - Pre-discretized inputs (omega_k + c_j2) -> skip discretization
    """

    @classmethod
    def define(cls, spec):
        super().define(spec)

        spec.input("model", valid_type=ModelData, help="Renormalizer model")
        spec.input("code", valid_type=orm.AbstractCode, help="Code for CalcJob execution")

        spec.input(
            "spectral_density_type",
            valid_type=orm.Str,
            default=lambda: orm.Str("ohmic_exp"),
            help="SDF type: ohmic_exp | debye | cole_davidson | custom",
        )
        spec.input("omega_min", valid_type=orm.Float, default=lambda: orm.Float(0.0))
        spec.input("omega_max", valid_type=orm.Float, default=lambda: orm.Float(10.0))
        spec.input("num_points", valid_type=orm.Int, default=lambda: orm.Int(512))
        spec.input("alpha", valid_type=orm.Float, default=lambda: orm.Float(0.05))
        spec.input("s_exponent", valid_type=orm.Float, default=lambda: orm.Float(1.0))
        spec.input("cutoff", valid_type=orm.Float, default=lambda: orm.Float(1.0))
        spec.input("lambda_reorg", valid_type=orm.Float, default=lambda: orm.Float(1.0))
        spec.input("beta", valid_type=orm.Float, default=lambda: orm.Float(0.7))

        spec.input("custom_omega", valid_type=orm.ArrayData, required=False)
        spec.input("custom_j_omega", valid_type=orm.ArrayData, required=False)
        spec.input("omega_k", valid_type=orm.ArrayData, required=False)
        spec.input("c_j2", valid_type=orm.ArrayData, required=False)

        spec.input("n_modes", valid_type=orm.Int, default=lambda: orm.Int(16))
        spec.input(
            "discretization_method",
            valid_type=orm.Str,
            default=lambda: orm.Str("trapz"),
            help="trapz | wang1_like | equal_area",
        )

        spec.input("frequency_scale", valid_type=orm.Float, default=lambda: orm.Float(1.0))
        spec.input("coupling_scale", valid_type=orm.Float, default=lambda: orm.Float(1.0))

        spec.output("bath_spectrum", valid_type=orm.ArrayData, required=False)
        spec.output("bath_modes", valid_type=orm.ArrayData, help="Discretized bath modes")
        spec.output("mpo_coefficients", valid_type=orm.Dict, help="MPO-ready coefficient dict")
        spec.output("output_parameters", valid_type=orm.Dict, help="Pipeline summary")

        spec.exit_code(470, "ERROR_SPECTRAL_DENSITY_FAILED", message="Bath spectral density calculation failed")
        spec.exit_code(471, "ERROR_DISCRETIZATION_FAILED", message="Bath discretization failed")
        spec.exit_code(472, "ERROR_MAPPING_FAILED", message="Bath coefficient mapping failed")
        spec.exit_code(473, "ERROR_INPUT_VALIDATION", message="Invalid workflow inputs")

        spec.outline(
            cls.setup,
            if_(cls.needs_spectral_density)(
                cls.run_spectral_density,
                cls.inspect_spectral_density,
            ),
            if_(cls.needs_discretization)(
                cls.run_discretization,
                cls.inspect_discretization,
            ),
            cls.run_mapping,
            cls.inspect_mapping,
            cls.finalize,
        )

    def setup(self):
        self.report("Starting bath MPO coefficient pipeline")
        has_continuous = ("custom_omega" in self.inputs and "custom_j_omega" in self.inputs)
        has_discrete = ("omega_k" in self.inputs and "c_j2" in self.inputs)
        has_partial_continuous = ("custom_omega" in self.inputs) ^ ("custom_j_omega" in self.inputs)
        has_partial_discrete = ("omega_k" in self.inputs) ^ ("c_j2" in self.inputs)
        if has_partial_continuous or has_partial_discrete:
            self.report("Continuous/discrete inputs must be provided as complete pairs")
            return self.exit_codes.ERROR_INPUT_VALIDATION
        if has_continuous and has_discrete:
            self.report("Both continuous and pre-discretized inputs were given")
            return self.exit_codes.ERROR_INPUT_VALIDATION

    def needs_spectral_density(self):
        has_discrete = ("omega_k" in self.inputs and "c_j2" in self.inputs)
        return not has_discrete

    def needs_discretization(self):
        has_discrete = ("omega_k" in self.inputs and "c_j2" in self.inputs)
        return not has_discrete

    def run_spectral_density(self):
        self.report("Running bath spectral density calculation")
        inputs = {
            "model": self.inputs.model,
            "code": self.inputs.code,
            "spectral_density_type": self.inputs.spectral_density_type,
            "omega_min": self.inputs.omega_min,
            "omega_max": self.inputs.omega_max,
            "num_points": self.inputs.num_points,
            "alpha": self.inputs.alpha,
            "s_exponent": self.inputs.s_exponent,
            "cutoff": self.inputs.cutoff,
            "lambda_reorg": self.inputs.lambda_reorg,
            "beta": self.inputs.beta,
        }
        if "custom_omega" in self.inputs and "custom_j_omega" in self.inputs:
            inputs["custom_omega"] = self.inputs.custom_omega
            inputs["custom_j_omega"] = self.inputs.custom_j_omega
        return ToContext(spectral_calc=self.submit(BathSpectralDensityCalcJob, **inputs))

    def inspect_spectral_density(self):
        calc = self.ctx.spectral_calc
        if not calc.is_finished_ok:
            self.report(f"Spectral density failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_SPECTRAL_DENSITY_FAILED

        params = calc.outputs.output_parameters.get_dict()
        omega = np.array(params["omega_grid"], dtype=float)
        j_omega = np.array(params["j_omega"], dtype=float)

        omega_data = orm.ArrayData()
        omega_data.set_array("omega_grid", omega)
        j_data = orm.ArrayData()
        j_data.set_array("j_omega", j_omega)
        spectrum = orm.ArrayData()
        spectrum.set_array("omega_grid", omega)
        spectrum.set_array("j_omega", j_omega)

        self.ctx.omega_grid = omega_data
        self.ctx.j_omega = j_data
        self.ctx.bath_spectrum = spectrum

    def run_discretization(self):
        self.report("Running bath discretization")
        if "omega_k" in self.inputs and "c_j2" in self.inputs:
            return
        inputs = {
            "model": self.inputs.model,
            "code": self.inputs.code,
            "omega_grid": self.ctx.omega_grid,
            "j_omega": self.ctx.j_omega,
            "n_modes": self.inputs.n_modes,
            "method": self.inputs.discretization_method,
        }
        return ToContext(discretization_calc=self.submit(BathDiscretizationCalcJob, **inputs))

    def inspect_discretization(self):
        calc = self.ctx.discretization_calc
        if not calc.is_finished_ok:
            self.report(f"Discretization failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_DISCRETIZATION_FAILED

        params = calc.outputs.output_parameters.get_dict()
        omega_k = np.array(params["omega_k"], dtype=float)
        c_j2 = np.array(params["c_j2"], dtype=float)

        omega_k_data = orm.ArrayData()
        omega_k_data.set_array("omega_k", omega_k)
        c_j2_data = orm.ArrayData()
        c_j2_data.set_array("c_j2", c_j2)

        self.ctx.omega_k = omega_k_data
        self.ctx.c_j2 = c_j2_data

    def run_mapping(self):
        self.report("Running bath-to-MPO coefficient mapping")
        if "omega_k" in self.inputs and "c_j2" in self.inputs:
            omega_k = self.inputs.omega_k
            c_j2 = self.inputs.c_j2
        else:
            omega_k = self.ctx.omega_k
            c_j2 = self.ctx.c_j2

        inputs = {
            "model": self.inputs.model,
            "code": self.inputs.code,
            "omega_k": omega_k,
            "c_j2": c_j2,
            "frequency_scale": self.inputs.frequency_scale,
            "coupling_scale": self.inputs.coupling_scale,
        }
        return ToContext(mapping_calc=self.submit(BathToMPOCoeffCalcJob, **inputs))

    def inspect_mapping(self):
        calc = self.ctx.mapping_calc
        if not calc.is_finished_ok:
            self.report(f"Coefficient mapping failed: exit_status={calc.exit_status}")
            return self.exit_codes.ERROR_MAPPING_FAILED
        self.ctx.mapping_params = calc.outputs.output_parameters.get_dict()

    def finalize(self):
        if "bath_spectrum" in self.ctx:
            self.out("bath_spectrum", self.ctx.bath_spectrum)

        mapping = self.ctx.mapping_params
        modes = orm.ArrayData()
        modes.set_array("omega_k", np.array(mapping["omega_k"], dtype=float))
        modes.set_array("c_j2", np.array(mapping["c_j2"], dtype=float))
        modes.set_array("displacement", np.array(mapping["displacement"], dtype=float))
        modes.set_array("g_k", np.array(mapping["g_k"], dtype=float))
        self.out("bath_modes", modes)

        coeffs = orm.Dict(dict=mapping)
        self.out("mpo_coefficients", coeffs)
        self.out(
            "output_parameters",
            orm.Dict(
                dict={
                    "pipeline": "bath_mpo_pipeline",
                    "used_pre_discretized_input": ("omega_k" in self.inputs and "c_j2" in self.inputs),
                    "n_modes": len(mapping["omega_k"]),
                    "reorganization_energy": mapping.get("reorganization_energy"),
                    "converged": bool(mapping.get("converged", True)),
                }
            ),
        )
        self.report("Bath MPO pipeline completed")
