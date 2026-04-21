# WorkChain Examples

This folder groups process-level orchestration examples.

Current cases remain under:
- `examples/workchain/mps/*`
- `examples/workchain/ttn/*`

Execution contract:

1. WorkChain orchestrates only processes.
2. Physics/model/tensor construction lives in CalcJobs.
3. Each case stays single-file (`run_one_shot.py`) and directly editable.
