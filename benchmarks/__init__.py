"""MomentEmu benchmark suite.

Run with ``python -m benchmarks`` from the directory that contains ``bench/``.
Every suite writes one JSON file into the results directory and ``report.py``
turns the JSON files into a markdown table. ``gate.py`` compares two result
directories with tolerances and exits non-zero on a regression.

The package under test is imported from ``REPO_SRC`` (set once here so a
stale site-packages copy cannot shadow it).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# benchmarks/ lives at the repository root; the package under test is in src/.
_DEFAULT_SRC = str(Path(__file__).resolve().parent.parent / "src")
REPO_SRC = os.environ.get("MOMENTEMU_SRC", _DEFAULT_SRC)
if REPO_SRC not in sys.path:
    sys.path.insert(0, REPO_SRC)

import MomentEmu  # noqa: E402

MOMENTEMU_FILE = MomentEmu.__file__
if not MOMENTEMU_FILE.startswith(REPO_SRC):
    raise ImportError(
        f"MomentEmu imported from {MOMENTEMU_FILE}, expected a copy under {REPO_SRC}"
    )

__all__ = ["REPO_SRC", "MOMENTEMU_FILE"]
