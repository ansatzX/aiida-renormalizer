#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$ROOT_DIR/../../.." && pwd)"
ARTIFACT_BASE="${ARTIFACT_BASE:-$REPO_ROOT/tmp}"
mkdir -p "$ARTIFACT_BASE"

echo "[1/3] preparing symbolic input node"
eval "$(verdi run "$ROOT_DIR/prepare_symbolic_inputs.py")"

echo "[2/3] running symbolic TTNS evolve CalcJob"
LAUNCH_OUTPUT="$(verdi run "$ROOT_DIR/launch_symbolic_calcjob.py" \
  --code-pk "$CODE_PK" \
  --symbolic-inputs-pk "$SYMBOLIC_INPUTS_PK" \
  --dt 0.05 \
  --nsteps 40 \
  --method tdvp_ps \
  --artifact-base "$ARTIFACT_BASE" 2>&1)" || {
  echo "launch failed"
  echo "$LAUNCH_OUTPUT"
  exit 1
}

PROCESS_PK="$(printf '%s\n' "$LAUNCH_OUTPUT" | rg '^PROCESS_PK=' | sed 's/^PROCESS_PK=//')"
if [[ -z "${PROCESS_PK:-}" ]]; then
  echo "launch did not produce PROCESS_PK"
  echo "$LAUNCH_OUTPUT"
  exit 1
fi

echo "[3/3] inspect latest process"
verdi process show "$PROCESS_PK"
