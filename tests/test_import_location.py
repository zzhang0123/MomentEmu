"""P0.1: the test suite must import the repository package, not a stale flat file."""
from __future__ import annotations

from pathlib import Path

import MomentEmu


def test_momentemu_imported_from_src() -> None:
    pkg = Path(MomentEmu.__file__).resolve()
    assert pkg.name == "__init__.py"
    assert pkg.parent.name == "MomentEmu"
    assert pkg.parent.parent.name == "src", f"MomentEmu imported from {pkg}"


def test_package_exposes_version() -> None:
    assert MomentEmu.__version__
