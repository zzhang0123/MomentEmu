"""P4.4: packaging metadata, LICENSE and version."""
from __future__ import annotations

import tomllib
from pathlib import Path

import MomentEmu

ROOT = Path(__file__).resolve().parents[2]


def _project():
    with open(ROOT / "pyproject.toml", "rb") as f:
        return tomllib.load(f)["project"]


def test_version_is_2_0_0():
    assert _project()["version"] == "2.0.0"
    assert MomentEmu.__version__ == "2.0.0"


def test_requires_python_and_floors():
    p = _project()
    assert p["requires-python"] == ">=3.10"
    deps = " ".join(p["dependencies"])
    assert "numpy>=1.24" in deps
    assert "scipy>=1.10" in deps
    assert "sympy>=1.13" in deps
    assert "scikit-learn>=1.3" in deps


def test_pep639_license_expression():
    p = _project()
    assert p["license"] == "MIT"
    assert "License :: OSI Approved :: MIT License" not in p["classifiers"]
    assert "MIT" not in " ".join(p["classifiers"])


def test_license_has_mit_attribution_clause():
    text = (ROOT / "LICENSE").read_text()
    assert "shall be included in all copies or substantial portions" in text
    assert "THE SOFTWARE IS PROVIDED" in text


def test_extras_present():
    p = _project()
    for extra in ("jax", "torch", "all", "dev", "docs"):
        assert extra in p["optional-dependencies"]
