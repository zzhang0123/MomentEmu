"""Guards for the integration tests that read the 2026-09 review archive.

The archive's ``bayes_common`` pins the checkout it was recorded against and
asserts that ``MomentEmu`` resolves under that path, so these tests only mean
something when the suite runs from that checkout. Anywhere else, most often a
git worktree, they are skipped with the mismatch spelled out instead of failing
as though the code were broken.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

#: The r2 archive holding the cached emulators and bayes_common.
ARCHIVE = Path("/Users/zzhang/Workspace/MomentEmu-review-2026-09/r2")

#: Presence marker: the cached emulators every archive test loads.
_CACHE = ARCHIVE / "bayes_cache_emus.pkl"


def _pinned_src() -> str | None:
    """The src path bayes_common pins, read as text.

    Importing bayes_common runs its own assert against this value, so it has
    to be read without importing the module.
    """
    f = ARCHIVE / "bayes_common.py"
    if not f.exists():
        return None
    m = re.search(r"""^REPO_SRC\s*=\s*["'](.+?)["']""", f.read_text(), re.M)
    return m.group(1) if m else None


@pytest.fixture
def review_archive() -> Path:
    """The archive directory; skips when it is not on this machine."""
    if not _CACHE.exists():
        pytest.skip(f"review archive not present at {ARCHIVE}")
    return ARCHIVE


@pytest.fixture
def archive_bayes_common(review_archive: Path):
    """The archive's bayes_common module.

    Skips when MomentEmu does not resolve under the path bayes_common pins.
    Importing it then trips its own assert, which reports a path mismatch as a
    test failure.
    """
    import MomentEmu

    pinned = _pinned_src()
    if pinned is not None and not MomentEmu.__file__.startswith(pinned):
        pytest.skip(
            f"the review archive pins MomentEmu at {pinned}, but this run "
            f"resolves it to {MomentEmu.__file__}; run the suite from that "
            f"checkout rather than from a worktree"
        )
    if str(review_archive) not in sys.path:
        sys.path.insert(0, str(review_archive))
    import bayes_common

    return bayes_common
