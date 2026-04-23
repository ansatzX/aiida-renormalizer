# Examples

New unified layout:

1. `examples/mps/*`
- MPS-family migrated cases from `ori_examples/*.py`.

2. `examples/ttn/*`
- TTN-family migrated cases from `ori_examples/ttns/*.py`.

## Contract

1. One editable `run_one_shot.py` per case.
2. Every script exposes `INPUT`, `MODEL`, `CALC` explicitly.
3. Generation-first default: `REAL_RUN = False`.
4. Runtime path: workfunction manifest generation + `BundleRunnerWorkChain` / `BundleRunnerCalcJob` single submit.
5. Generated script stages are always `write_generated_script` and `compile_generated_script`.
