# New Example: MPS SBM

This example now uses a **single editable script**:

- `run_one_shot.py`

Edit parameters directly at the top of the script and run:

```bash
verdi run examples/mps/sbm/run_one_shot.py
```

Notes:

- The flow is explicit LEGO composition in script form:
  `spectral_modes -> model_from_modes -> model_to_mpo -> mpo_to_initial_mps -> dynamics`.
- Adiabatic renormalization is included in the `spectral_modes` step.
