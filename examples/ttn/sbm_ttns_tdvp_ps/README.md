# TTNS TDVP-PS Example

This example focuses on plugin correctness:
- one-shot run uses `TtnsSymbolicEvolveCalcJob` with symbolic inputs only
- no local `renormalizer` object construction in user script
- heavy TTNS/TTNO build + evolution are executed inside CalcJob runtime
- `t=0` trajectory point is explicitly checked

## 1) One-shot Python script (symbolic input)

```bash
verdi run examples/ttn/sbm_ttns_tdvp_ps/run_one_shot.py
```

## 2) Multi-step `verdi` CLI flow (symbolic inputs)

```bash
bash examples/ttn/sbm_ttns_tdvp_ps/run_via_verdi.sh
```

The shell script runs:
1. `verdi run prepare_symbolic_inputs.py`
2. `verdi run launch_symbolic_calcjob.py ...`
3. `verdi process show <PROCESS_PK>`
