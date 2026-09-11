"""P4.7: every public symbol has a docstring; the docs site is configured."""
from __future__ import annotations

import inspect
from pathlib import Path

import MomentEmu
from MomentEmu import core, emulator, guards, io, monomials

MODULES = [MomentEmu, emulator, core, io, monomials, guards]


def _missing():
    missing = set()
    for mod in MODULES:
        for name, obj in vars(mod).items():
            if name.startswith("_") or inspect.ismodule(obj):
                continue
            if (inspect.isfunction(obj) or inspect.isclass(obj)) and obj.__module__ == mod.__name__:
                if not inspect.getdoc(obj):
                    missing.add(f"{mod.__name__}.{name}")
                if inspect.isclass(obj):
                    for mname, mobj in vars(obj).items():
                        if mname.startswith("_") or not inspect.isfunction(mobj):
                            continue
                        if not inspect.getdoc(mobj):
                            missing.add(f"{mod.__name__}.{name}.{mname}")
    return missing


def test_no_missing_public_docstrings():
    assert _missing() == set()


def test_docs_site_config_present():
    root = Path(__file__).resolve().parents[2]
    assert (root / "mkdocs.yml").exists()
    assert (root / "docs" / "index.md").exists()
    text = (root / "mkdocs.yml").read_text()
    assert "mkdocstrings" in text


def test_predictive_rename_alias():
    assert core.predictive_rmse_aic_bic is not None
    assert core.predictive_mse_aic_bic is core.predictive_rmse_aic_bic
