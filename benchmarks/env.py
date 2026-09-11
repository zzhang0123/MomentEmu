"""Environment fingerprint recorded with every result file."""
from __future__ import annotations

import os
import platform
import subprocess
import sys
from datetime import datetime, timezone

import numpy as np

from benchmarks import MOMENTEMU_FILE, REPO_SRC


def _git_sha(path: str) -> str:
    try:
        out = subprocess.run(
            ["git", "-C", path, "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True, timeout=10,
        )
        return out.stdout.strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _cpu_brand() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True, text=True, check=True, timeout=5,
            ).stdout.strip()
        except Exception:  # noqa: BLE001
            pass
    return platform.processor() or platform.machine()


def _blas_name() -> str:
    try:
        cfg = np.show_config(mode="dicts")
        return str(cfg["Build Dependencies"]["blas"]["name"])
    except Exception:  # noqa: BLE001
        return "unknown"


def environment() -> dict:
    import scipy
    import sklearn

    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "cpu": _cpu_brand(),
        "n_cpu": os.cpu_count(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "sklearn": sklearn.__version__,
        "blas": _blas_name(),
        "momentemu_file": MOMENTEMU_FILE,
        "momentemu_git_sha": _git_sha(REPO_SRC),
        "loadavg_at_start": (lambda: [round(x, 2) for x in os.getloadavg()] if hasattr(os, "getloadavg") else None)(),
        "thread_env": {
            k: os.environ.get(k)
            for k in ("OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
            if os.environ.get(k)
        },
    }
