"""P4.6: README and autodiff-guide code blocks and corrected claims."""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _python_blocks(path: Path):
    text = path.read_text()
    return re.findall(r"```python\n(.*?)```", text, flags=re.DOTALL)


def test_all_python_blocks_compile():
    for name in ("README.md", "autodiff-guide.md"):
        blocks = _python_blocks(ROOT / name)
        assert blocks, name
        for i, block in enumerate(blocks):
            compile(block, f"{name}#{i}", "exec")


def test_no_zero_numerical_error_claim():
    for name in ("README.md", "autodiff-guide.md"):
        assert "zero numerical error" not in (ROOT / name).read_text().lower()


def test_readme_jax_import_is_correct():
    text = (ROOT / "README.md").read_text()
    assert "from jax_momentemu import" not in text
    assert "from MomentEmu.jax_momentemu import create_jax_emulator" in text


def test_guide_install_forms_are_pep508():
    text = (ROOT / "autodiff-guide.md").read_text()
    assert "git+https://github.com/zzhang0123/MomentEmu.git[jax]" not in text
    assert "MomentEmu[jax] @ git+" in text


def test_readme_states_estimator_equivalence_and_x64():
    text = (ROOT / "README.md").read_text()
    assert "same least-squares problem as" in text
    assert "jax_enable_x64" in text
    assert "cobaya / emcee" in text


def test_readme_quickstart_runs():
    blocks = _python_blocks(ROOT / "README.md")
    quickstart = blocks[0]
    env = dict(os.environ)
    env["MOMENTEMU_IGNORE_STALE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", quickstart],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
        cwd=ROOT,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
