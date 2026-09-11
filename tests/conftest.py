"""tests/conftest.py for MomentEmu.

Puts <repo>/src at the FRONT of sys.path before any test module is
imported, then asserts that the package actually resolved there. A stale
flat module (e.g. site-packages/MomentEmu.py left by an old pip install)
therefore fails collection with a message naming the shadowing file instead
of silently substituting its own code for the repository's.

Also registers the slow and backend markers used by the suite.
"""
from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pytest


def _find_src(start: Path) -> Path:
    """Walk up from this file until a directory containing src/MomentEmu/__init__.py is found."""
    for candidate in (start, *start.parents):
        src = candidate / "src"
        if (src / "MomentEmu" / "__init__.py").is_file():
            return src
    raise RuntimeError(
        f"no src/MomentEmu/__init__.py found above {start}; conftest.py must live inside the repository "
        f"(normally <repo>/tests/conftest.py)"
    )


SRC = _find_src(Path(__file__).resolve().parent)
REPO_ROOT = SRC.parent

# Drop any earlier entry pointing at src/ so the insert below is unambiguous.
sys.path[:] = [p for p in sys.path if Path(p or ".").resolve() != SRC.resolve()]
sys.path.insert(0, str(SRC))

# Evict a previously imported (possibly stale) MomentEmu so the assertion below
# sees the copy that the path order now selects.
for _name in [m for m in sys.modules if m == "MomentEmu" or m.startswith("MomentEmu.")]:
    del sys.modules[_name]

_pkg = importlib.import_module("MomentEmu")
if _pkg.__file__ is None:
    raise RuntimeError("MomentEmu resolved to a namespace package (no __init__.py); src/MomentEmu is incomplete")
_where = Path(_pkg.__file__).resolve()
_expected = (SRC / "MomentEmu" / "__init__.py").resolve()  # resolve() so a symlinked src/ also passes
if _where != _expected:
    raise RuntimeError(
        f"MomentEmu imported from {_where}, not from {_expected}. A stale installed copy is "
        f"shadowing the repository; run pip uninstall MomentEmu and pip install -e ."
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "slow: takes more than ~5 s; deselect with -m 'not slow'")
    config.addinivalue_line("markers", "backend(name): requires the optional jax/torch extra")


@pytest.fixture(scope="session")
def repo_src() -> Path:
    return SRC


@pytest.fixture
def rng() -> np.random.Generator:
    """Fixed-seed generator so every test that draws data is reproducible."""
    return np.random.default_rng(20260910)
