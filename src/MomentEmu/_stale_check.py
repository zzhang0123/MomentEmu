"""Detect the pre-package flat ``MomentEmu.py`` install (P0.9).

MomentEmu before August 2025 shipped as a single top-level ``MomentEmu.py``.
The package layout replaced it, but pip does not remove a file that a
different install method put there, and both layouts carry version 1.0.0.  A
flat file earlier on ``sys.path`` shadows this package.  This runs from the
package's ``__init__``, so the package has already imported: raising would
break a working process, so it warns.  ``MOMENTEMU_IGNORE_STALE=1`` silences
it; ``tests/conftest.py`` keeps the hard failure for the test suite.
"""
from __future__ import annotations

import os
import sys
import warnings

_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
ENV_IGNORE = "MOMENTEMU_IGNORE_STALE"
FIX_COMMAND = "pip uninstall -y MomentEmu && pip install MomentEmu"


def find_stale_flat_modules(search_path: list[str] | None = None) -> list[str]:
    """Absolute paths of top-level ``MomentEmu.py`` files on ``sys.path``.

    The package submodule ``MomentEmu/MomentEmu.py`` is excluded.
    """
    own = os.path.normcase(os.path.join(_PKG_DIR, "MomentEmu.py"))
    hits: list[str] = []
    for entry in (sys.path if search_path is None else search_path):
        base = entry or os.getcwd()
        cand = os.path.join(base, "MomentEmu.py")
        if os.path.isfile(cand) and os.path.normcase(os.path.abspath(cand)) != own:
            hits.append(os.path.abspath(cand))
    return hits


def stale_dist_owns(path: str) -> bool:
    """True when an installed distribution named MomentEmu lists ``path`` in its RECORD."""
    import importlib.metadata as md

    for dist in md.Distribution.discover(name="MomentEmu"):
        for f in dist.files or []:
            if str(f) == "MomentEmu.py":
                try:
                    if os.path.samefile(str(dist.locate_file(f)), path):
                        return True
                except OSError:
                    pass
    return False


def warn_if_stale() -> None:
    """Warn (once per import) when a flat module shadows this package."""
    if os.environ.get(ENV_IGNORE) == "1":
        return
    hits = find_stale_flat_modules()
    if not hits:
        return
    lines = [
        "MomentEmu: a stale single-file install from the pre-package layout "
        "(MomentEmu <= July 2025) is on sys.path and can shadow this package:",
    ]
    for h in hits:
        owner = (
            "listed in the installed MomentEmu distribution RECORD"
            if stale_dist_owns(h)
            else "NOT owned by any installed distribution; delete it by hand"
        )
        lines.append(f"  {h}  ({owner})")
    lines.append(f"Fix: {FIX_COMMAND}, or set {ENV_IGNORE}=1 to silence this warning.")
    warnings.warn("\n".join(lines), RuntimeWarning, stacklevel=3)
