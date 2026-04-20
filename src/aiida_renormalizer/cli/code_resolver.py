# -*- coding: utf-8 -*-
"""Helpers to resolve usable AiiDA codes for reno CLI commands."""
from __future__ import annotations

import os
from pathlib import Path

from aiida import orm


def _is_healthy_script_code(code: orm.AbstractCode) -> bool:
    """Return True if the code is suitable for ``reno.script`` execution."""
    if code.default_calc_job_plugin != "reno.script":
        return False

    if isinstance(code, orm.InstalledCode):
        # For localhost codes we can validate the executable path eagerly.
        if code.computer.label == "localhost":
            executable = Path(str(code.filepath_executable))
            return executable.exists() and os.access(executable, os.X_OK)
        # Remote codes cannot be validated from local filesystem.
        return True

    if isinstance(code, orm.PortableCode):
        return bool(code.filepath_executable)

    return False


def find_default_script_code() -> orm.AbstractCode | None:
    """Return the first healthy code configured for ``reno.script``."""
    candidates = orm.QueryBuilder().append(orm.InstalledCode, project="*").all(flat=True)
    candidates.extend(orm.QueryBuilder().append(orm.PortableCode, project="*").all(flat=True))

    healthy = [code for code in candidates if _is_healthy_script_code(code)]
    if not healthy:
        return None

    healthy.sort(key=lambda code: code.pk or 0, reverse=True)
    return healthy[0]
