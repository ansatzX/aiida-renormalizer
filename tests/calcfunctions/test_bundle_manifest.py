from __future__ import annotations

from aiida import orm

from aiida_renormalizer.calcfunction.calcfunction_ttn_sbm_zt import (
    ColeDavidsonSDF_setup,
    build_bundle_manifest as ttn_sbm_zt_build_bundle_manifest,
    build_calculation as ttn_sbm_zt_build_calculation,
    build_ttn_model as ttn_sbm_zt_build_ttn_model,
    define_basis as ttn_sbm_zt_define_basis,
    define_hamiltonian_terms as ttn_sbm_zt_define_hamiltonian_terms,
)
from aiida_renormalizer.calculations.manifest_ops import (
    build_bundle_manifest_payload,
    bundle_manifest_for_python_script,
)


def test_build_bundle_manifest_payload_rejects_duplicate_stage_names():
    try:
        build_bundle_manifest_payload(
            [
                {"name": "dup", "script": "x = 1\n"},
                {"name": "dup", "script": "y = 2\n"},
            ]
        )
    except Exception:
        return
    raise AssertionError("expected duplicate stage names to be rejected")


def test_bundle_manifest_for_python_script_materializes_bundle(aiida_profile):
    manifest = bundle_manifest_for_python_script(
        script_name=orm.Str("demo.py"),
        script_text=orm.Str("print('ok')\n"),
        work_dir=orm.Str("generated_scripts"),
    ).get_dict()

    assert manifest["schema"] == "bundle_manifest_v1"
    assert manifest["stage_count"] == 2
    assert "generated_scripts/demo.py" in manifest["stages"][0]["script"]


def test_symbolic_ttn_sbm_zt_lifecycle_renders_script(aiida_profile):
    environment = ColeDavidsonSDF_setup(
        ita=orm.Float(1.0),
        omega_c=orm.Float(0.1),
        beta=orm.Float(0.5),
        upper_limit=orm.Float(30.0),
        raw_delta=orm.Float(1.0),
        n_modes=orm.Int(8),
    )

    env_dict = {
        "omega_k": environment.get_array("omega_k").tolist(),
        "c_j2": environment.get_array("c_j2").tolist(),
    }
    hamiltonian_terms_py = [
        ["sigma_z", "spin", 0.0, 0],
        ["sigma_x", "spin", "delta_eff", 0],
    ]
    for imode, omega in enumerate(env_dict["omega_k"]):
        mode_dof = f"v_{imode}"
        hamiltonian_terms_py.extend([["p^2", mode_dof, 0.5, 0], ["x^2", mode_dof, 0.5 * omega**2, 0]])
    for imode, c_j2 in enumerate(env_dict["c_j2"]):
        mode_dof = f"v_{imode}"
        hamiltonian_terms_py.append(["sigma_z x", ["spin", mode_dof], c_j2**0.5, [0, 0]])

    hamiltonian_terms = ttn_sbm_zt_define_hamiltonian_terms(hamiltonian_terms=orm.List(list=hamiltonian_terms_py))

    basis_py = [["half_spin", "spin", [0, 0]]]
    for imode, omega in enumerate(env_dict["omega_k"]):
        safe_omega = max(float(omega), 1e-12)
        nbas = int(round(max(16 * float(env_dict["c_j2"][imode]) / safe_omega**3, 4.0)))
        basis_py.append(["sho", f"v_{imode}", float(omega), nbas])
    basis = ttn_sbm_zt_define_basis(basis=orm.List(list=basis_py))

    model_section = ttn_sbm_zt_build_ttn_model(
        hamiltonian_terms=hamiltonian_terms,
        basis=basis,
        tree_type=orm.Str("binary"),
        m_max=orm.Int(12),
    )
    calculation_section = ttn_sbm_zt_build_calculation(
        dt=orm.Float(0.2),
        nsteps=orm.Int(10),
        method=orm.Str("tdvp_ps"),
    )

    bundle_outputs = ttn_sbm_zt_build_bundle_manifest(
        environment=environment,
        model_section=model_section,
        calculation_section=calculation_section,
        real_run=orm.Bool(False),
        work_dir=orm.Str("generated_scripts"),
    )

    payload = bundle_outputs["script_payload"].get_dict()
    assert payload["script_name"] == "symbolic_ttn_dynamics_generated.py"
    assert "{{" not in payload["script_text"]
    assert "ham_terms.extend([" in payload["script_text"]
    assert "EvolveConfig(" in payload["script_text"]
