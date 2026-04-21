# CalcJob Examples

This folder groups direct CalcJob examples.

Case layout:
- `mps/dynamics/run_one_shot.py`
- `mps/fmo/run_one_shot.py`
- `mps/h2o_qc/run_one_shot.py`
- `mps/hubbard/run_one_shot.py`
- `mps/sbm/run_one_shot.py`
- `mps/ssh/run_one_shot.py`
- `mps/transport_kubo/run_one_shot.py`
- `ttn/junction_ft/run_one_shot.py`
- `ttn/junction_zt/run_one_shot.py`
- `ttn/sbm_ft/run_one_shot.py`
- `ttn/sbm_zt/run_one_shot.py`

Execution contract:

1. Scripts are low-level brick calls with minimal abstraction.
2. Parameters are intentionally in-script and human-editable.
3. Case layout mirrors `examples/workchain/*` for semantic comparison.
