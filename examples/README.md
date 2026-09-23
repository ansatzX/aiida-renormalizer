# Examples

New unified layout:

1. `examples/mps/*`
- MPS-family migrated cases from `ori_examples/*.py`.

2. `examples/ttn/*`
- TTN-family migrated cases from `ori_examples/ttns/*.py`.

The TTN SBM ZT gold target now uses its own formal library under
`src/aiida_renormalizer/cases/ttn_sbm_zt/`. It always generates a complete runnable
Reno-only script without executing it, plus a CalcJob-compatible manifest. See
[the gold case README](ttn/sbm_zt/README.md).

## Existing contract for cases not yet reconstructed

1. One editable `run_one_shot.py` per case.
2. Every script exposes `INPUT`, `MODEL`, `CALC` explicitly.
3. Generation-first default: `REAL_RUN = False`.
4. Current recording path: direct calcfunction calls for script payloads and manifests.
5. Generated script stages include `write_generated_script` and `compile_generated_script`; examples do not submit an execute stage.
6. Generated scripts resolve bundled example input files relative to their concrete example directory, not the caller's current working directory.
7. Each case owns `aiida_scaffold/calcfunctions.py` and its local Jinja templates. A case must not import another case's scaffold while examples are still being reconstructed independently.
8. Direct generated-script execution is not an AiiDA process. Until CalcJob execution is introduced, manifests must state that execution failures, result files, and checkpoint files are not recorded by AiiDA provenance.
