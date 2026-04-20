# aiida-renormalizer

`aiida-renormalizer` is an AiiDA plugin for tensor-network workflows built on top of Renormalizer. Its job is not to expose raw Renormalizer scripting to end users, but to make those calculations reproducible, restartable, and automation-friendly inside AiiDA.

## What This Plugin Is For

The plugin provides AiiDA-native building blocks for common tensor-network workloads:

- AiiDA data nodes for models, operators, basis sets, MPOs, MPSs, TTNOs, and TTNSs
- CalcJobs for DMRG, TDVP, spectra, transport, and TTN operations
- WorkChains for multi-step workflows such as ground-state search, time evolution, convergence studies, and spectroscopy
- `verdi reno ...` commands for day-to-day usage

The intended user-facing interfaces are:

- AiiDA Python API
- `verdi` CLI

The intended interface is not “write custom Renormalizer scripts in the README”. Renormalizer remains a runtime dependency of the plugin, but it should stay behind the plugin boundary in normal usage.

## Storage Model

Wavefunction-sized data such as MPS and TTNS are treated as external artifacts.

- AiiDA stores provenance, metadata, and lightweight manifests
- Large tensor payloads live outside the AiiDA repository
- Each wavefunction node records a logical artifact address:
  - `storage_backend`
  - `storage_base`
  - `relative_path`
  - checksum and size metadata

This is deliberate. For large tensor-network calculations, keeping dense wavefunction payloads inside the AiiDA repository quickly becomes awkward for archiving, sharing, and publication.

The design goal is:

- AiiDA tracks the workflow and the scientific object identity
- external storage holds the heavy payload
- publication/export steps can remap those artifacts into a shareable bundle

## Installation

### Runtime

```bash
pip install git+https://github.com/ansatzX/aiida-renormalizer
```

### Development

```bash
git clone https://github.com/ansatzX/aiida-renormalizer.git
cd aiida-renormalizer
pip install -e ".[dev]"
```

## AiiDA Setup

For production usage, use PostgreSQL plus RabbitMQ.

### Conda example

```bash
conda create -n aiida -c conda-forge python=3.12 postgresql rabbitmq aiida-core=2.8.0
conda activate aiida

# Initialize and start PostgreSQL inside the conda environment
mkdir -p "$CONDA_PREFIX/var/postgresql"
initdb -D "$CONDA_PREFIX/var/postgresql"
pg_ctl -D "$CONDA_PREFIX/var/postgresql" -l "$CONDA_PREFIX/var/postgresql/logfile" start

# Start RabbitMQ inside the conda environment
rabbitmq-server -detached
```

### Create database and profile

```bash
createuser -s aiida
createdb -O aiida aiida
verdi quicksetup \
  --profile default \
  --email "you@example.com" \
  --first-name "Your" \
  --last-name "Name" \
  --institution "Your Org" \
  --db-engine postgresql_psycopg \
  --db-backend core.psql_dos \
  --db-username aiida \
  --db-name aiida \
  --db-hostname localhost \
  --db-port 5432 \
  --non-interactive
verdi daemon start
```

## Python API Quickstart

The Python API is meant to work with AiiDA nodes and AiiDA process factories.

```python
from aiida import engine, orm
from aiida.plugins import DataFactory, WorkflowFactory

ModelData = DataFactory("reno.model")
MpoData = DataFactory("reno.mpo")
GroundStateWorkChain = WorkflowFactory("reno.ground_state")

# In the normal plugin workflow, ModelData and MpoData come from
# previous AiiDA steps, imported inputs, or CLI-assisted setup.
model = orm.load_node(<model_pk>)
mpo = orm.load_node(<mpo_pk>)
code = orm.load_code("renormalizer@localhost")

result = engine.run(
    GroundStateWorkChain,
    model=model,
    mpo=mpo,
    code=code,
    energy_convergence=orm.Float(1e-8),
)

print("Ground-state node:", result["ground_state"].pk)
print("Energy:", result["energy"].value)
```

For time evolution:

```python
from aiida import engine, orm
from aiida.plugins import WorkflowFactory

TimeEvolutionWorkChain = WorkflowFactory("reno.time_evolution")

result = engine.run(
    TimeEvolutionWorkChain,
    model=orm.load_node(<model_pk>),
    mpo=orm.load_node(<mpo_pk>),
    initial_mps=orm.load_node(<mps_pk>),
    total_time=orm.Float(10.0),
    dt=orm.Float(0.01),
    code=orm.load_code("renormalizer@localhost"),
)

print("Final state:", result["final_mps"].pk)
```

## TTNS Symbolic Example

Use the symbolic TTNS TDVP-PS example when you want setup and runtime construction to stay separated.

```bash
# one-shot
verdi run examples/ttn/sbm_ttns_tdvp_ps/run_one_shot.py

# multi-step verdi flow
bash examples/ttn/sbm_ttns_tdvp_ps/run_via_verdi.sh
```

The example uses `reno.ttns_symbolic_evolve` and stores artifacts under `<repo>/tmp` by default.

## `verdi` Quickstart

### Ground state

```bash
verdi reno ground-state \
  -m model.toml \
  -b basis.toml \
  -C renormalizer@localhost \
  --artifact-storage-base /data/reno-artifacts
```

Minimal `basis.toml`:

```toml
[[basis]]
type = "BasisHalfSpin"
dof = "spin"

[basis.params]
sigmaqn = [0, 0]

[[basis]]
type = "BasisSHO"
dof = "v0"

[basis.params]
omega = 1.0
nbas = 6
```

Minimal `model.toml`:

```toml
[[hamiltonian]]
symbol = "sigma_x"
dofs = "spin"
factor = 0.4

[[hamiltonian]]
symbol = "sigma_z"
dofs = "spin"
factor = 0.05

[[hamiltonian]]
symbol = "b^\\dagger b"
dofs = "v0"
factor = 1.0

[[hamiltonian]]
symbol = "sigma_z x"
dofs = ["spin", "v0"]
factor = 0.08
```

Ready-to-copy files are included at:
- `examples/cli_inputs/basis.toml`
- `examples/cli_inputs/model.toml`

### Time evolution

```bash
verdi reno evolve \
  -s 123 \
  -H 456 \
  -t 100.0 \
  --artifact-storage-base /data/reno-artifacts
```

### Spectrum

```bash
verdi reno spectrum \
  -s 123 \
  -H 456 \
  -o a_dag \
  -m spectra_zero_t \
  --publication-bundle ./paper-bundle
```

## Publication Bundles

Publication-oriented export is part of the storage design.

- heavy wavefunction files remain external during normal work
- a publication step can gather them into a bundle directory
- the bundle includes a stable artifact filename, machine-readable metadata, and a short human-readable README

Export a node-backed artifact into a publication bundle:

```bash
verdi reno bundle -n 123 -o ./paper-bundle
```

Optionally relink the node so future access resolves against the new bundle:

```bash
verdi reno bundle -n 123 -o ./paper-bundle --relink
```

The exported directory is organized like this:

```text
paper-bundle/
  README.md
  manifest.json
  metadata/
    summary.json
  artifacts/
    mps-<uuid-prefix>.npz
```

`manifest.json` captures provenance, original logical location, exported timestamp, checksum-bearing artifact metadata, and the bundle-relative artifact path. `metadata/summary.json` keeps the most useful node summary fields easy to inspect and script against without reading the full manifest.

This makes it easier to:

- reorganize data before submission
- share reproducible artifacts with collaborators
- publish a clean directory tree for supporting information

## Current Scope

The repository currently contains:

- data-layer types for MPS/MPO/TTNS/TTNO/model/basis/config
- composite calculations and parsers
- workchains for ground state, time evolution, spectroscopy, transport, and TTN workflows
- CLI commands under `verdi reno`

## Tests

Run the plugin suite from the package directory:

```bash
cd aiida-renormalizer
uv run python -m pytest -q tests
```

## Requirements

- Python >= 3.10
- `aiida-core==2.8.0`
- `renormalizer>=0.0.11`
- `numpy<2.0`

## Compatibility Notes

- The plugin drivers target current Renormalizer public APIs (`renormalizer.utils.configs`), not legacy `renormalizer.parameter`.
- For AiiDA 2.8.0 environments, use `verdi run ...` for script-style launches.
- Some advanced Renormalizer branches are upstream-limited and may raise `NotImplementedError` depending on model/configuration (for example complex Kubo coupling paths).

## License



Not Setup now
