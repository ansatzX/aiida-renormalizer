"""Recorded CalcJob-compatible payload and non-destructive local dry-run output."""

from __future__ import annotations

import ast
import base64
import hashlib
import json
import sys
from pathlib import Path

from aiida import orm
from aiida.engine import calcfunction

SCRIPT_NAME = "mps_transport_kubo_generated.py"
RESULT_NAME = "transport_kubo_result.json"


def validate_standalone_script(source: str) -> None:
    """Require syntactically valid source importing only stdlib and Renormalizer."""
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                raise ValueError("generated script must not use relative imports")
            names = [node.module or ""]
        else:
            continue
        for name in names:
            root = name.split(".")[0]
            if root != "renormalizer" and root not in sys.stdlib_module_names:
                raise ValueError(f"unexpected generated-script dependency: {name}")
    compile(tree, SCRIPT_NAME, "exec")


@calcfunction
def prepare_execution_manifest(script: orm.Str) -> orm.Dict:
    """Record write/compile/execute stages for BundleRunnerCalcJob, without submitting it."""
    source = script.value
    validate_standalone_script(source)
    encoded = base64.b64encode(source.encode()).decode("ascii")
    stages = [
        {
            "name": "write_generated_script",
            "script": (
                "import base64\nfrom pathlib import Path\n"
                f"Path({SCRIPT_NAME!r}).write_bytes(base64.b64decode({encoded!r}))\n"
            ),
        },
        {
            "name": "compile_generated_script",
            "script": (
                "from pathlib import Path\n"
                f"compile(Path({SCRIPT_NAME!r}).read_text(), {SCRIPT_NAME!r}, 'exec')\n"
            ),
        },
        {
            "name": "execute_generated_script",
            "script": (
                "import subprocess\nimport sys\nfrom pathlib import Path\n"
                f"subprocess.run([sys.executable, str(Path({SCRIPT_NAME!r}).resolve())], "
                "check=True)\n"
            ),
        },
    ]
    digest = hashlib.sha256(source.encode()).hexdigest()
    return orm.Dict(
        dict={
            "schema": "bundle_manifest_v1",
            "stage_count": len(stages),
            "stages": stages,
            "artifacts": [
                {
                    "role": "generated_script",
                    "path": SCRIPT_NAME,
                    "required": True,
                    "sha256": digest,
                    "content_recorded_by_aiida": True,
                },
                {
                    "role": "result",
                    "path": RESULT_NAME,
                    "required": True,
                    "recorded_by_aiida": False,
                },
            ],
            "execution_contract": {
                "mode": "dry_run_for_bundle_runner_calcjob",
                "submitted_to_aiida": False,
                "execution_recorded_by_aiida": False,
                "result_recorded_by_aiida": False,
                "runtime_dependencies": ["python_standard_library", "renormalizer"],
            },
            "restart_contract": {"mode": "replay-only", "checkpoint_restart": False},
            "provenance": {
                "script_uuid": script.uuid,
                "render_process_uuid": script.creator.uuid if script.creator else None,
            },
        }
    )


def write_dry_run(script: orm.Str, output_dir: str | Path) -> Path:
    """Write a fresh bundle directory; never clear or overwrite existing run artifacts."""
    output_dir = Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(
            f"dry-run output already exists; choose a new directory: {output_dir}"
        )
    manifest = prepare_execution_manifest(script)
    # exist_ok=False also protects against a concurrent creator after the check above.
    output_dir.mkdir(parents=True, exist_ok=False)
    script_path = output_dir / SCRIPT_NAME
    script_path.write_text(script.value)
    (output_dir / "manifest.json").write_text(json.dumps(manifest.get_dict(), indent=2))
    return script_path
