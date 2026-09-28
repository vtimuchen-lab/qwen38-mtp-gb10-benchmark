"""Host identification recorded in every result (name, GPU, driver, OS)."""

from __future__ import annotations

import os
import platform
import socket
import subprocess
from typing import Any


def _nvidia_smi() -> tuple[str | None, str | None]:
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None, None
    if completed.returncode != 0 or not completed.stdout.strip():
        return None, None
    first = completed.stdout.strip().splitlines()[0]
    parts = [part.strip() for part in first.split(",")]
    return (parts[0] or None), (parts[1] if len(parts) > 1 and parts[1] else None)


def collect_host(overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Detect host facts; values in ``overrides`` (from [host]) win."""
    overrides = overrides or {}
    gpu, driver = _nvidia_smi()
    host: dict[str, Any] = {
        "name": socket.gethostname(),
        "gpu": gpu,
        "driver": driver,
        "arch": platform.machine() or None,
        "kernel": platform.release() or None,
        "os": platform.platform(),
        "cpus": os.cpu_count(),
        "python": platform.python_version(),
        "source": "detected",
    }
    labels = overrides.get("labels")
    for key in ("name", "gpu", "driver"):
        if overrides.get(key):
            host[key] = overrides[key]
    if labels:
        host["labels"] = dict(labels)
    return host
