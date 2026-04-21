# aiida-renormalizer

AiiDA plugin for Renormalizer tensor-network workflows.

## What This Plugin Does

- Data nodes for models, operators, MPO/MPS/TTNO/TTNS
- CalcJobs for DMRG, TDVP, spectra, transport, TTN operations
- WorkChains for multi-step workflows

User interfaces: AiiDA Python API, `verdi run ...`

## Storage Model

Wavefunction payloads (MPS/TTNS) stored externally. AiiDA tracks provenance and metadata.

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

### Conda example (Linux)

```bash
conda create -n aiida -c conda-forge python=3.12 aiida-core=2.7.3 aiida-core.services postgresql "numpy<2.0"
conda activate aiida

# Initialize and start PostgreSQL inside the conda environment
mkdir -p "$CONDA_PREFIX/var/postgresql"
initdb -D "$CONDA_PREFIX/var/postgresql"
pg_ctl -D "$CONDA_PREFIX/var/postgresql" -l "$CONDA_PREFIX/var/postgresql/logfile" start

# Start services
verdi presto
```

Notes:

- On Linux, `aiida-core.services` can manage PostgreSQL and RabbitMQ.

### Conda example (macOS)

```bash
conda create -n aiida -c conda-forge python=3.12 aiida-core=2.7.3 postgresql rabbitmq-server "numpy<2.0"
conda activate aiida

# Initialize and start PostgreSQL inside the conda environment
mkdir -p "$CONDA_PREFIX/var/postgresql"
initdb -D "$CONDA_PREFIX/var/postgresql"
pg_ctl -D "$CONDA_PREFIX/var/postgresql" -l "$CONDA_PREFIX/var/postgresql/logfile" start

# Start RabbitMQ broker
rabbitmq-server -detached
rabbitmqctl await_startup
```

Notes:

- On macOS, do not rely on `aiida-core.services` due to dependency/platform issues.
- `rabbitmq-server` is the RabbitMQ broker daemon required by AiiDA daemon/task transport.
- If `rabbitmq-server -detached` fails with an address/port error, check:
  `rabbitmq-diagnostics -q ping`

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

## Examples

22 example scripts organized into `calcjob/` (low-level) and `workchain/` (high-level) with mirrored MPS/TTN cases.

TTN example:
```bash
verdi run examples/workchain/ttn/sbm_zt/run_one_shot.py
```

MPS example:
```bash
verdi run examples/workchain/mps/sbm/run_one_shot.py
```

## Publication Bundles

Wavefunction artifacts are stored externally. Publication bundles can be created with provenance metadata and artifact manifests for sharing/archiving.

CLI export command (`verdi reno bundle ...`) is temporarily disabled during API refactor.

## Current Scope

- Data types: ModelData, MPSData, MPOData, OpData, BasisSetData, ConfigData, BasisTreeData, TensorNetworkLayoutData, TTNSData, TTNOData
- Parsers: RenoBaseParser, ScriptedParser
- Examples: 22 scripts in `calcjob/` and `workchain/`

### CalcJobs (33 total)

```mermaid
graph TB
    subgraph L1[L1 Atomic]
        A1[BuildMPO]
        A2[Expectation]
        A3[Compress]
        A4[MaxEntangledMpdm]
        A5[ModelFromSymbolicSpec]
    end

    subgraph L15[L1.5 LEGO]
        L1[ComputeOccupations]
        L2[ComputeMsd]
    end

    subgraph L2[L2 Composite]
        C1[DMRG]
        C2[ImagTime]
        C3[TDVP]
        C4[ThermalProp]
        C5[Property]
    end

    subgraph Spectra[L2 Spectra/Transport]
        S1[SpectraZeroT]
        S2[SpectraFiniteT]
        S3[Kubo]
        S4[CorrectionVector]
        S5[ChargeDiffusion]
        S6[SpectralFunction]
    end

    subgraph Bath[Bath Pipeline]
        B1[BathSpectralDensity]
        B2[OhmicRenormModes]
        B3[BathDiscretization]
        B4[BathSpinBosonModel]
        B5[SbmSymbolicSpecFromModes]
        B6[BathToMPOCoeff]
    end

    subgraph TTN[TTN]
        T1[OptimizeTTNS]
        T2[TTNSymbolicModel]
        T3[TTNSEvolve]
        T4[TTNSSymbolicEvolve]
        T5[TTNSExpectation]
        T6[TTNSEntropy]
        T7[TTNSMutualInfo]
        T8[TTNSRdm]
    end

    subgraph L3[L3 Scripted]
        SC[RenoScriptCalcJob]
    end
```

### WorkChains (28 total)

```mermaid
graph TB
    subgraph Core[Core]
        W1[RenoRestartWorkChain]
        W2[TimeEvolutionWorkChain]
        W3[GroundStateWorkChain]
        W4[AbsorptionWorkChain]
        W5[ConvergenceWorkChain]
    end

    subgraph Extended[Extended]
        W6[ThermalStateWorkChain]
        W7[KuboTransportWorkChain]
        W8[CustomPipelineWorkChain]
    end

    subgraph Sweeps[Sweep WorkChains]
        W9[ParameterSweepWorkChain]
        W10[TemperatureSweepWorkChain]
        W11[BondDimensionSweepWorkChain]
        W12[FrequencySweepWorkChain]
    end

    subgraph Advanced[Advanced Dynamics]
        W13[CorrectionVectorWorkChain]
        W14[ChargeDiffusionWorkChain]
    end

    subgraph Bath[Bath Pipelines]
        W15[BathMPOPipelineWorkChain]
        W16[BathSpinBosonModelWorkChain]
        W17[OhmicRenormModesWorkChain]
        W18[SbmModelFromModesWorkChain]
        W19[ModelToMPOWorkChain]
        W20[MPOToInitialMPSWorkChain]
        W21[MPSDynamicsWorkChain]
    end

    subgraph Models[Model WorkChains]
        W22[SpinBosonWorkChain]
        W23[VibronicWorkChain]
    end

    subgraph TTN[TTN WorkChains]
        W24[TTNGroundStateWorkChain]
        W25[TTNTimeEvolutionWorkChain]
        W26[TTNMPSComparisonWorkChain]
        W27[TTNSymbolicModelWorkChain]
        W28[TTNSymbolicDynamicsWorkChain]
    end
```

## Tests

Run the plugin suite from the package directory:

```bash
cd aiida-renormalizer
uv run python -m pytest -q tests
```

## Requirements

- Python >= 3.9
- `aiida-core==2.7.3`
- `renormalizer`
- `numpy<2.0`

## Compatibility Notes

- The plugin drivers target current Renormalizer public APIs (`renormalizer.utils.configs`), not legacy `renormalizer.parameter`.
- For AiiDA 2.7.3 environments, use `verdi run ...` for script-style launches.
- The custom `verdi reno ...` CLI entrypoint is temporarily disabled during API refactor.
- Some advanced Renormalizer branches are upstream-limited and may raise `NotImplementedError` depending on model/configuration (for example complex Kubo coupling paths).

## License

Not Setup now
