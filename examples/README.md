# Examples

## Directory Contract

1. `examples/calcjob/`
- Direct low-level brick execution (`CalcJob` path).

2. `examples/workchain/`
- Process-level orchestration (`WorkChain + CalcJob` path).

3. Case mirroring
- Case folders are mirrored between `calcjob/` and `workchain/` so users can compare semantic equivalence directly.

## Script Contract

1. Single-file entrypoint
- Each case uses one `run_one_shot.py` as the editable script.

2. Explicit process blocks
- Preferred top-level blocks: `INPUT`, `BUILD`, `RUN`.
- No hidden helper that auto-builds the physical model behind the script.

3. Physical semantics first
- The script should make it clear where Hamiltonian terms and numerical parameters come from.
- For SBM/TTN chains: spectral parameters -> renormalization/discretization -> symbolic terms -> TN object -> dynamics.

4. No per-case README sprawl
- Keep narrative docs at `examples/README.md`, `examples/calcjob/README.md`, and `examples/workchain/README.md`.
- Case folders should primarily contain runnable scripts and data files.
