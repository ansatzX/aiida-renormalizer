# CD spin-boson TTN gold target

`run_one_shot.py` is the human-edited scientific script. Its implementation library
is `src/aiida_renormalizer/cases/ttn_sbm_zt/`, owned only by this case. The script
imports formal packages and uses Renormalizer `Op`, `BasisHalfSpin`, `BasisSHO`,
and `EvolveConfig` objects directly. The old `aiida_scaffold` directory is no longer used by this
case; other cases are reconstructed separately.

## Edit the scientific problem

The script exposes the CD bath parameters, Wang1 discretization, tail-integration
bounds, effective tunneling coefficient, explicit Hamiltonian terms, basis sizes,
binary tree and contraction labels, local product states, fixed bond dimension,
Hamiltonian-guided expansion, evolution configuration, and observation timing.

- The spin starts in local state 0 (`sigma_z=+1`); each oscillator starts in the
  undisplaced SHO vacuum. This is not a coupled-system ground-state calculation.
- CD renormalization integrates from the largest discrete frequency to 1000 times
  that frequency. `BATH_FREQUENCY_LIMIT` controls bath discretization, not that tail bound.
- `discretize_cole_davidson_spectrum` turns the continuous spectral density into
  oscillator frequencies and squared coordinate-coupling coefficients. `CD_STRENGTH`
  is the spectral-density prefactor (Reno's `ita`); `CD_BETA` is a shape exponent,
  not inverse temperature. `OMEGA_C` controls the spectral shape rather than a hard cutoff.
- The script uses hbar = 1 and unit oscillator masses. Plain energy/frequency
  inputs share one chosen unit, and `TIME_STEP` uses its inverse. `EPSILON` and
  bare `DELTA` enter the Pauli terms without a factor of 1/2.
- `MAX_BOND_DIMENSION` controls virtual TTN bonds. Each oscillator's `basis_size`
  instead counts its retained SHO states; the empirical basis rule and primitive
  contraction choices are explained beside the construction loop.
- Add native `Op` objects to `EXTRA_SYSTEM_TERMS` without replacing the defaults.
  To change a built-in term, edit its explicit `Op` construction in `main()`.
- Observations map readable names to `Op` or `OpSum` objects. The default records
  `sigma_z` and `sigma_x` after every step, without a measurement at time zero.
  Real expectations are numbers; complex expectations use `{"real": ..., "imag": ...}`.
- Restart remains replay-only; no TTNS checkpoint is persisted.

To start in the +x spin state, use ordinary local amplitudes in the script:

```python
INITIAL_SPIN_STATE = [2**-0.5, 2**-0.5]
```

Each local state can be an occupation integer or a normalized real coefficient
vector with one entry per basis function. Lists, tuples, and one-dimensional
NumPy arrays are accepted. Vectors are preserved, not automatically normalized.
Non-finite values, incompatible quantum numbers, wrong sizes, and nonzero imaginary
amplitudes are rejected before rendering. The native Reno Hartree-product path
used here does not preserve complex local vectors, so arbitrary complex initial
states require a separate implementation.

For VMF convergence controls, edit the native configuration:

```python
EVOLVE_CONFIG = EvolveConfig(
    method=EvolveMethod.tdvp_vmf,
    ivp_rtol=1e-8,
    ivp_atol=1e-11,
    reg_epsilon=1e-12,
)
```

The package records the resolved settings as an AiiDA input and reconstructs them
in the standalone script. Constructor defaults are recorded as concrete values;
the remote program does not silently substitute its own defaults. Changes made
to supported config fields after construction are also captured.

In the current Reno TTN implementation, VMF reads the three controls above;
PS, PS2, and the fixed fourth-order propagator do not. The supported TTN routes
also ignore such general `EvolveConfig` options as `adaptive` and `ivp_solver`.
Requesting nondefault values for inactive settings raises a descriptive error.
Custom RK/Taylor coefficients and unknown configuration attributes are rejected
instead of silently lost during serialization.

The current case API supports string dofs, HalfSpin and undisplaced unscaled non-DVR
SHO bases, the local product states described above, fixed compression, and optional
Hamiltonian-guided expansion. Unsupported choices fail explicitly; this is not a
universal TN API.

## State lifetime within one runtime process

The generated `evolve_segment` function consumes a live TTNS and returns
`(final_state, observation_rows)`. It never reconstructs the model or reinitializes
the state. The default generated `main()` retains that state, writes the same
observation JSON, and returns `(final_state, result)` to its caller.

Within the generated program, ordinary Python calls can continue a state through
multiple segments. For example, after the model has been built:

```python
observables = {"sigma_z": Op("sigma_z", "spin", qn=0)}
state, first_rows = evolve_segment(
    state, hamiltonian,
    dt=0.2, nsteps=10, evolve_config=build_evolve_config(),
    observations=observables, observe_initial=True, observe_every=1,
    start_time=0.0, start_step=0,
)
state, second_rows = evolve_segment(
    state, hamiltonian,
    dt=0.1, nsteps=20, evolve_config=build_evolve_config(),
    observations=observables, observe_initial=False, observe_every=2,
    start_time=2.0, start_step=10,
)
spin_rdm = state.calc_1dof_rdm("spin")
```

Use the returned state at each step: Reno can return a new TTNS. To branch, call
`branch = state.copy()` explicitly. A nonempty segment assigns its supplied solver
configuration to the input state before evolving; copy first if the input object's
configuration must be retained. Direct segment calls apply the same configuration
validation as recording, before measurements or state configuration changes.
New Hamiltonians and prebuilt observation TTNOs
must use that state's exact `state.basis` tree. Named observations also accept
native Op/OpSum objects, which are built on that tree.

Time and step offsets are supplied explicitly, independently of observation
spacing. Cadence counts from the start of each segment; requesting an initial
measurement on both sides of a boundary intentionally records that boundary twice.
Sparse observations do not force a final measurement or determine the next start
time. An empty observations dictionary gives pure evolution and an empty row list.
Zero steps returns the exact input object, optionally measuring it at the supplied
start time, without changing its configuration.

All of these objects live in one Python process. A normal command-line run ends
their lifetime when it exits. JSON contains observations and clock metadata, not
a saved TTNS; cross-CalcJob continuation and checkpoint recovery remain separate.

## Generate the standalone runtime script

Use the Python 3.10 `aiida` environment with this package installed. Configure a
local AiiDA profile with the bootstrap described in the root README, then run:

```bash
conda run -n aiida python examples/ttn/sbm_zt/run_one_shot.py
```

This is a dry run: it records preprocessing and rendering, then writes a fresh
`generated_scripts/` directory containing:

- `symbolic_ttn_dynamics_generated.py`: complete executable Python whose only
  non-standard-library import dependency is Renormalizer;
- `manifest.json`: source hash/provenance, output declarations, replay policy,
  and write/compile/execute stages compatible with `BundleRunnerCalcJob`.

The generated Python embeds all resolved scientific data and implementation.
It does not import AiiDA, `aiida_renormalizer`, Jinja, or local helpers, and does
not load sidecar input files. It has no `REAL_RUN` switch: executing it runs the
specified calculation. The launcher never submits a CalcJob or executes a solver.
The manifest is ready for that later execution boundary, not evidence that a
remote calculation has run. The existing runner parser records its stage summary;
it does not create a dedicated scientific output node from `sbm_zt_result.json`.

An existing output directory is rejected before preprocessing. Keep prior runs
and edit `OUTPUT_DIR` in the script to select a fresh destination:

```python
OUTPUT_DIR = Path("/absolute/path/to/new-run")
```

The Python entrypoint also accepts `main(output_dir=...)` for callers that already
hold the script's module; no CLI or temporary helper library is required.

For a small runtime check, set `AIIDA_RENO_SBM_ZT_SMOKE=1` while generating. This
selects four modes, bond dimension four, and one step. Execute the resulting
standalone file separately:

```bash
AIIDA_RENO_SBM_ZT_SMOKE=1 conda run -n aiida python examples/ttn/sbm_zt/run_one_shot.py
conda run -n aiida python examples/ttn/sbm_zt/generated_scripts/symbolic_ttn_dynamics_generated.py
```

The result is written beside the generated script as `sbm_zt_result.json`.
Wang1 still uses upstream's fixed 10^7-point preprocessing grid in smoke mode.

## Library responsibilities and checks

`bath.py` implements the selected bath transformations; `api.py` adapts native
Reno objects to recorded inputs; `representation.py`, `validation.py`, and
`recording.py` validate and emit the requested construction/evolution. `evolution.py`
records native solver settings and rejects ineffective or lossy configurations. Packaged
templates contain the runtime implementation. `artifacts.py` records the
CalcJob-compatible manifest and writes fresh dry-run outputs without deleting
existing results. None of these modules imports another case library.

Focused checks cover upstream CD preprocessing equivalence, custom terms, explicit
construction choices, isolated Reno-only execution, analytic uncoupled-spin
trajectories for both initial spins and +x precession, native VMF settings reaching
the solver, split/unsplit evolution equivalence, changed-Hamiltonian continuation,
final-state analysis and branch independence, summed/complex observables, and the existing
CalcJob driver's success/failure and retrieval-path contracts. Small runtime tests
do not establish convergence of the full interacting 1000-mode calculation.
