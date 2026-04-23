# aiida-renormalizer

## Refactor Status (WIP)

This repository is **currently under active refactor** toward a research-first, composable architecture.

Current guidance:

1. API/process interfaces may still change without compatibility guarantees.
2. Prefer using `examples/mps/*` and `examples/ttn/*` as the current reference path.
3. Legacy paths/modules are being incrementally removed or rewritten.
4. Generated-script workflow (`REAL_RUN=False`) is the recommended validation path during refactor.

AiiDA plugin refactored for **research-oriented composable execution**:
`workfunction (Jinja code generation) + BundleRunnerCalcJob + BundleRunnerWorkChain`.

## Process Layer (Strict)

AiiDA process layer exposes only:

1. `BundleRunnerCalcJob`
2. `BundleRunnerWorkChain`
3. code-generation workfunctions in `aiida_renormalizer.calcfunctions`

No nested WorkChain orchestration is used in active process paths.

## Core Design

1. Generation-first examples (`REAL_RUN=False` by default)
- Examples generate editable single-file scripts first.
- Heavy compute execution is optional.

2. Symbolic model construction
- Scripts explicitly define `INPUT`, `MODEL`, `CALC`.
- Hamiltonian construction is symbolic and visible in generated script.

3. Composable bricks
- Spectral parameters
- Renormalization/discretization
- Symbolic Hamiltonian terms
- Tensor-network construction
- Dynamics

4. Reusable topology datatype
- `TopologyData` stores reusable tensor-network topology skeleton metadata to avoid repeated topology reconstruction setup.

## Examples

Unified migrated layout:

- `examples/mps/*` from `ori_examples/*.py`
- `examples/ttn/*` from `ori_examples/ttns/*.py`

Representative runs:

```bash
python examples/mps/hubbard/run_one_shot.py
python examples/ttn/sbm_zt/run_one_shot.py
```

## Install

```bash
pip install -e ".[dev]"
```

## Test

```bash
pytest -q tests
```

## Entry points

- Workflow: `reno.bundle_runner`
- Calculation: `reno.bundle_runner`
