"""Anisotropic / structured index sets (P5.3).

A Basis composes constraints by intersection: a total-degree bound, a
maximum interaction order, a weighted q-norm, per-parameter degree caps and
per-group degree limits.  Basis.total_degree().build(names, d) is row-
identical to generate_multi_indices.
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator
from typing import Any

import numpy as np

from MomentEmu.emulator import generate_multi_indices


def validate_blocks(blocks: Any, n: int) -> None:
    """Raise unless ``blocks`` partitions ``range(n)``.

    Kept separate from the index search in :meth:`Basis.build` so the
    partition is rejected before any enumeration starts, rather than part-way
    through it.
    """
    if not blocks:
        raise ValueError("blocks must contain at least one block")
    seen: set[int] = set()
    for position, block in enumerate(blocks):
        if len(tuple(block)) == 0:
            raise ValueError(
                f"block {position} is empty; an empty block contributes no "
                "terms and usually means a block lost its contents"
            )
        for i in block:
            if not 0 <= i < n:
                raise ValueError(
                    f"block index {i} is out of range for {n} parameters"
                )
            if i in seen:
                raise ValueError(
                    f"blocks must be disjoint; parameter {i} appears twice"
                )
            seen.add(i)
    missing = sorted(set(range(n)) - seen)
    if missing:
        raise ValueError(
            f"blocks must cover every parameter; {missing} unassigned. A "
            "parameter left out has no defined coupling, so list it in its own "
            "block to declare it independent, or merge it into the block it "
            "interacts with."
        )


def _block_candidates(blocks: Any, n: int, d: int) -> Iterator[tuple[int, ...]]:
    """Yield every multi-index of degree <= d whose support is in one block.

    ``blocks`` must already have passed :func:`validate_blocks`; the constant
    row is yielded once.
    """
    yield (0,) * n
    for block in blocks:
        for sub in generate_multi_indices(len(block), d):
            if not sub.any():
                continue
            alpha = [0] * n
            for pos, var in enumerate(block):
                alpha[var] = int(sub[pos])
            yield tuple(alpha)


def _bounded_candidates(
    bounds: Any,
    d: int,
    parity: Any,
    max_interaction: int | None,
    groups: Any,
    q: float | None = None,
    weights: Any = None,
) -> Iterator[tuple[int, ...]]:
    """Yield every alpha with ``alpha[i] <= bounds[i]`` and ``sum(alpha) <= d``.

    The Cartesian product of the per-parameter ranges has prod(bounds_i + 1)
    entries and ignores the degree bound, so it enumerates 8**9 = 134,217,728
    tuples to keep the C(16, 7) = 11,440 of them that a degree-7 basis over 9
    parameters admits. Carrying the remaining degree down the descent
    enumerates the admissible set directly.

    Parity, ``max_interaction``, the group limits and the q-norm are monotone
    along a prefix -- extending it can only keep or raise the power, the
    support size, every group sum and every q-norm term -- so all four prune
    the descent instead of filtering its output.

    The q-norm prune carries a relative slack of 1e-9 and the caller still
    applies the exact test. Its threshold is ``d + 1e-12``, a tie-break that a
    running sum in a different order can land on the other side of; the slack
    keeps the prune from dropping a row the exact test would keep.
    """
    n = len(bounds)
    # Admissible powers per position. An "odd" position never takes 0, so it
    # also spends at least 1 of the budget, which `owed` below accounts for.
    steps: list[tuple[int, ...]] = []
    for i in range(n):
        p_i = None if parity is None else parity[i]
        if p_i is None:
            steps.append(tuple(range(bounds[i] + 1)))
        elif p_i == "even":
            steps.append(tuple(range(0, bounds[i] + 1, 2)))
        else:
            steps.append(tuple(range(1, bounds[i] + 1, 2)))
    if any(not step for step in steps):
        # Some position admits no power at all (e.g. parity "odd" under a cap
        # of 0), so no complete index exists.
        return
    # (v * w_i) ** q is increasing in v for q > 0, so a prefix that is already
    # over the cap stays over and the ascending steps can be cut short.
    q_cap = None if q is None else (d + 1e-12) ** q * (1.0 + 1e-9)
    # Concrete values for the descent below: an Optional captured by a nested
    # function cannot be narrowed at its use site, and q_cap stays the live
    # flag. build() already substitutes ones for a missing weight vector, so
    # the None branch here cannot be reached from the public API and guards
    # only against a future second caller.
    q_exp: float = 1.0 if q is None else float(q)
    q_w: list[float] = [1.0] * n if weights is None else [float(x) for x in weights]
    # owed[i]: the least degree positions i..n-1 must still consume.
    owed = [0] * (n + 1)
    for i in range(n - 1, -1, -1):
        owed[i] = owed[i + 1] + steps[i][0]
    group_of: list[list[int]] = [[] for _ in range(n)]
    limits: list[int] = []
    for g, (idx, limit) in enumerate(groups):
        limits.append(int(limit))
        for i in idx:
            group_of[i].append(g)

    alpha = [0] * n
    gsum = [0] * len(limits)

    def descend(
        i: int, remaining: int, support: int, acc: float
    ) -> Iterator[tuple[int, ...]]:
        if i == n:
            yield tuple(alpha)
            return
        if owed[i] > remaining:
            return
        at_support_cap = max_interaction is not None and support >= max_interaction
        for v in steps[i]:
            if v > remaining:
                break
            if v and at_support_cap:
                # steps[i] ascends, so every later power is nonzero too.
                break
            acc_v = acc if (q_cap is None or not v) else acc + (v * q_w[i]) ** q_exp
            if q_cap is not None and acc_v > q_cap:
                break
            for g in group_of[i]:
                gsum[g] += v
            if all(gsum[g] <= limits[g] for g in group_of[i]):
                alpha[i] = v
                yield from descend(
                    i + 1, remaining - v, support + (1 if v else 0), acc_v
                )
                alpha[i] = 0
            for g in group_of[i]:
                gsum[g] -= v

    yield from descend(0, d, 0, 0.0)


class Basis:
    """Composable index-set specification (P5.3).

    All constraints are applied together.  weights scales each parameter
    inside the q-norm; groups is a tuple of (indices, degree limit);
    per_parameter caps each parameter power (in addition to the D16 cap).
    The class method total_degree() builds the default isotropic basis.
    """

    def __init__(
        self,
        degree: int | None = None,
        max_interaction: int | None = None,
        q: float | None = None,
        weights: Any | None = None,
        groups: Any = (),
        per_parameter: Any | None = None,
        blocks: Any | None = None,
        parity: Any | None = None,
    ) -> None:
        if degree is not None and int(degree) < 0:
            raise ValueError("degree must be >= 0")
        if max_interaction is not None and int(max_interaction) < 1:
            raise ValueError("max_interaction must be >= 1")
        if q is not None and float(q) <= 0.0:
            raise ValueError("q must be > 0")
        self.degree = None if degree is None else int(degree)
        self.max_interaction = None if max_interaction is None else int(max_interaction)
        self.q = None if q is None else float(q)
        self.weights = None if weights is None else tuple(float(w) for w in weights)
        self.groups = tuple((tuple(int(i) for i in idx), int(lim)) for idx, lim in groups)
        self.per_parameter = (
            None if per_parameter is None else tuple(int(c) for c in per_parameter)
        )
        # blocks=None means no separability constraint; an empty tuple is an
        # error, caught in build() where the parameter count is known.
        self.blocks = (
            None if blocks is None
            else tuple(tuple(int(i) for i in block) for block in blocks)
        )
        # parity[i] constrains the power of parameter i: "even" allows
        # 0, 2, 4, ... and "odd" allows 1, 3, 5, ... The length is checked in
        # build(), where the parameter count is known.
        self.parity = None if parity is None else tuple(parity)

    #: Field defaults, used to backfill pickles written before a field existed.
    _DEFAULTS = {
        "degree": None,
        "max_interaction": None,
        "q": None,
        "weights": None,
        "groups": (),
        "per_parameter": None,
        "blocks": None,
        "parity": None,
    }

    def __setstate__(self, state: dict) -> None:
        """Restore a pickle written before a field existed.

        A Basis stored inside a PolyEmu pickle from an earlier release has no
        entry for a field added since, and the default unpickler restores
        __dict__ without calling __init__, so every missing field is backfilled
        to its default here. Same failure class as the B1 guard in
        PolyEmu._transforms.
        """
        merged = dict(Basis._DEFAULTS)
        merged.update(state)
        self.__dict__.update(merged)

    @classmethod
    def total_degree(cls, degree: int | None = None) -> Basis:
        """The default isotropic total-degree basis."""
        return cls(degree=degree)

    @classmethod
    def separable(cls, blocks: Any, degree: int | None = None, **kwargs: Any) -> Basis:
        """Block-separable basis: every support lies inside one block (T-001).

        For f(theta) = sum_k f_k(theta_{B_k}) the expansion carries only
        monomials whose support sits in a single block, so the index set drops
        from C(p+d, d) to 1 + sum_k [C(p_k+d, d) - 1] -- 230,230 to 1,845 at
        p=20, d=6 with four blocks of five. ``blocks`` must partition the
        parameters; :meth:`PolyEmu.interaction_graph` returns exactly that.

        The reduction is exactly lossless when f really is block separable and
        the design measure is a product measure, which a Latin hypercube or a
        uniform box satisfies. A coupling across blocks is NOT recoverable
        afterwards, so confirm the blocks before relying on them.
        """
        return cls(degree=degree, blocks=blocks, **kwargs)

    @classmethod
    def q_norm(
        cls,
        q: float,
        weights: Any | None = None,
        degree: int | None = None,
        **kwargs: Any,
    ) -> Basis:
        """A q-norm-weighted basis (weights default to ones)."""
        return cls(degree=degree, q=q, weights=weights, **kwargs)

    def build(self, names: Any, degree: int | None = None) -> np.ndarray:
        """Return the multi-indices satisfying every constraint (rows sorted)."""
        n = len(names)
        d = self.degree if degree is None else int(degree)
        if d is None:
            raise ValueError("a degree is required (pass it to build or set a degree)")
        if (
            self.max_interaction is None
            and self.q is None
            and self.per_parameter is None
            and not self.groups
            and self.blocks is None
            and self.parity is None
        ):
            return generate_multi_indices(n, d)

        bounds = (
            [d] * n
            if self.per_parameter is None
            else [min(d, int(c)) for c in self.per_parameter]
        )
        w = None
        if self.q is not None:
            w = np.ones(n) if self.weights is None else np.asarray(self.weights, float)
            if w.shape != (n,):
                raise ValueError(f"weights has shape {w.shape}, expected ({n},)")
            for i in range(n):
                bounds[i] = min(bounds[i], int(np.floor(d / w[i])))
        parity = self.parity
        if parity is not None:
            if len(parity) != n:
                raise ValueError(f"parity has length {len(parity)}, expected {n}")
            bad = [v for v in parity if v not in (None, "even", "odd")]
            if bad:
                raise ValueError(
                    f"parity entries must be None, 'even' or 'odd'; got {bad[0]!r}"
                )
        # _block_candidates enumerates per block and embeds, so its rows still
        # have to be checked against the bounds, parity, max_interaction and
        # the group limits. _bounded_candidates enforces all four as it builds
        # them, so those checks are skipped on that path.
        blocks_path = self.blocks is not None
        candidates: Iterable[tuple[int, ...]]
        if blocks_path:
            validate_blocks(self.blocks, n)
            candidates = _block_candidates(self.blocks, n, d)
        else:
            candidates = _bounded_candidates(
                bounds, d, parity, self.max_interaction, self.groups, self.q, w
            )
        # One numpy array per candidate costs more than every test in this
        # loop put together, so the checks stay on plain ints. per_parameter is
        # already folded into bounds, which both candidate sources respect.
        rows = []
        for alpha in candidates:
            if sum(alpha) > d:
                continue
            if blocks_path:
                if any(a > b for a, b in zip(alpha, bounds)):
                    continue
                if parity is not None and any(
                    (alpha[i] % 2 == 1) if v == "even" else (alpha[i] % 2 == 0)
                    for i, v in enumerate(parity) if v is not None
                ):
                    continue
                if (
                    self.max_interaction is not None
                    and sum(1 for a in alpha if a) > self.max_interaction
                ):
                    continue
                if any(
                    sum(alpha[i] for i in idx) > limit for idx, limit in self.groups
                ):
                    continue
            if self.q is not None and w is not None:
                # Kept in numpy, and kept last: the +1e-12 makes this a
                # tie-break, and a hand-rolled sum can land on the other side.
                arr = np.asarray(alpha, dtype=np.int64)
                if float(np.sum((arr * w) ** self.q) ** (1.0 / self.q)) > d + 1e-12:
                    continue
            rows.append(tuple(int(v) for v in alpha))
        if not rows:
            return np.zeros((0, n), dtype=np.int64)
        rows.sort(key=lambda a: (sum(a), a))
        return np.array(rows, dtype=np.int64)

    def spec(self) -> str:
        """A copy-pasteable constructor spec."""
        parts = [f"degree={self.degree}"]
        if self.max_interaction is not None:
            parts.append(f"max_interaction={self.max_interaction}")
        if self.q is not None:
            parts.append(f"q={self.q}")
        if self.weights is not None:
            parts.append(f"weights={list(self.weights)}")
        if self.per_parameter is not None:
            parts.append(f"per_parameter={list(self.per_parameter)}")
        if self.groups:
            parts.append(f"groups={list(self.groups)}")
        if self.blocks is not None:
            parts.append(f"blocks={[list(b) for b in self.blocks]}")
        if self.parity is not None:
            parts.append(f"parity={list(self.parity)}")
        return "Basis(" + ", ".join(parts) + ")"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Basis):
            return NotImplemented
        return (
            self.degree == other.degree
            and self.max_interaction == other.max_interaction
            and self.q == other.q
            and self.weights == other.weights
            and self.groups == other.groups
            and self.per_parameter == other.per_parameter
            and self.blocks == other.blocks
            and self.parity == other.parity
        )

    def __repr__(self) -> str:
        return self.spec()
