"""PyTorch backend for MomentEmu (P0.8 stopgap; contract-matched in P2.3).

The wrapper now builds tensors in the target dtype (float64 by default),
raises for a log_Y fit, treats a missing output scale as ones, checks the
trailing axis, and evaluates monomials functionally (no in-place Phi writes),
so torch.func.vmap can transform forward().
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from MomentEmu.emulator import _transform_codes
from MomentEmu.guards import output_scale
from MomentEmu.monomials import MonomialPlan


class TorchMomentEmu(nn.Module):
    """PyTorch module for a fitted forward MomentEmu."""

    def __init__(self, trained_emulator, dtype=torch.float64):
        super().__init__()
        if hasattr(trained_emulator, "export_payload"):
            self.dtype = dtype
            self._init_from_compressed(trained_emulator, dtype)
            return
        sparse = hasattr(trained_emulator, "candidate_indices") and not hasattr(
            trained_emulator, "forward_multi_indices"
        )
        if not sparse and not hasattr(trained_emulator, "forward_coeffs"):
            raise ValueError("the Torch backend needs a forward emulator")
        self.dtype = dtype
        if sparse:
            self._init_from_sparse(trained_emulator, dtype)
            return
        self.log_Y = bool(getattr(trained_emulator, "log_Y", False))
        transform = getattr(trained_emulator, "transform", None)
        if transform is None:
            transform = tuple(
                "log" if self.log_Y else "linear"
                for _ in range(trained_emulator.n_outputs)
            )
        self.transform_codes = _transform_codes(transform)
        self.n_params = int(trained_emulator.n_params)
        self.n_outputs = int(trained_emulator.n_outputs)
        self.multi_indices = np.asarray(trained_emulator.forward_multi_indices)
        # T-007: this used to be MonomialPlan.build() whatever the fit used, so
        # a Legendre or Chebyshev emulator was evaluated in the WRONG basis and
        # answered without raising: 4.8e-01 and 3.7e-01 relative on a degree-5
        # fit. Take the plan the emulator was actually fitted with.
        fitted_plan = getattr(trained_emulator, "forward_plan", None)
        self._plan = (
            fitted_plan if fitted_plan is not None
            else MonomialPlan.build(self.multi_indices)
        )
        self.basis_family = str(getattr(self._plan, "family", "monomial"))
        self.table_degree = (
            int(self.multi_indices.max()) if self.multi_indices.size else 0
        )
        self.register_buffer(
            "coeffs",
            torch.tensor(trained_emulator.forward_coeffs, dtype=dtype),
        )
        self.register_buffer(
            "input_mean", torch.tensor(trained_emulator.scaler_X.mean_, dtype=dtype)
        )
        self.register_buffer(
            "input_scale", torch.tensor(trained_emulator.scaler_X.scale_, dtype=dtype)
        )
        self.register_buffer(
            "output_mean", torch.tensor(trained_emulator.scaler_Y.mean_, dtype=dtype)
        )
        self.register_buffer(
            "output_scale",
            torch.tensor(
                output_scale(trained_emulator.scaler_Y, trained_emulator.n_outputs),
                dtype=dtype,
            ),
        )

    def _init_from_compressed(self, emulator, dtype) -> None:
        """Take a :class:`MomentEmu.compress.CompressedEmu`.

        The inner model supplies every stage but the last; the payload
        supplies the (D, k) coefficients and the (k, m) map, derived in
        CompressedEmu.export_payload so all three backends read one algebra.
        """
        pay = emulator.export_payload()
        TorchMomentEmu.__init__(self, emulator.model, dtype=dtype)
        self.n_outputs = int(pay.modes.shape[1])
        self.transform_codes = _transform_codes(
            tuple("linear" for _ in range(self.n_outputs))
        )
        # The payload already folds the inner model's output affine in.
        for name, value in (
            ("coeffs", pay.coeffs),
            ("output_mean", np.zeros(int(pay.coeffs.shape[1]))),
            ("output_scale", np.ones(int(pay.coeffs.shape[1]))),
            ("output_modes", pay.modes),
            ("output_offset", pay.offset),
        ):
            if hasattr(self, name):
                delattr(self, name)
            self.register_buffer(name, torch.tensor(np.asarray(value), dtype=dtype))

    def _init_from_sparse(self, emulator, dtype) -> None:
        """Take a :class:`MomentEmu.sparse.SparseEmu`, which is a forward fit.

        The closure walk needs a DOWNWARD CLOSED index set; a selected one is
        not, which is what sparsity means, so ``_plan`` is left unset and every
        basis goes through the multi-index path.
        """
        from MomentEmu.emulator import BASIS_PLANS

        self.log_Y = False
        mi = np.asarray(emulator.multi_indices)
        self.multi_indices = mi
        self.n_params = int(mi.shape[1])
        self.n_outputs = int(np.asarray(emulator.coefficients).shape[1])
        self.transform_codes = _transform_codes(
            tuple("linear" for _ in range(self.n_outputs))
        )
        self._plan = None
        self.basis_family = str(getattr(emulator, "basis_kind", "legendre"))
        if self.basis_family not in BASIS_PLANS:
            raise ValueError(f"unknown basis_kind {self.basis_family!r}")
        self.table_degree = int(mi.max()) if mi.size else 0
        # SparseEmu maps its box onto [-1, 1] as 2 (X - lo) / span - 1, which
        # is (X - mean) / scale with these two.
        span = np.asarray(emulator.span_, dtype=np.float64)
        scale = 0.5 * span
        self.register_buffer(
            "coeffs", torch.tensor(np.asarray(emulator.coefficients), dtype=dtype)
        )
        self.register_buffer(
            "input_mean",
            torch.tensor(np.asarray(emulator.lo_, dtype=np.float64) + scale,
                         dtype=dtype),
        )
        self.register_buffer("input_scale", torch.tensor(scale, dtype=dtype))
        self.register_buffer(
            "output_mean", torch.tensor(np.asarray(emulator.mean_Y_), dtype=dtype)
        )
        self.register_buffer(
            "output_scale", torch.tensor(np.asarray(emulator.scale_Y_), dtype=dtype)
        )

    def evaluate_basis(self, X_scaled: torch.Tensor) -> torch.Tensor:
        """(N, D) design in the basis the emulator was fitted with.

        The closure walk is taken only when a plan carrying closure tables is
        present, which a dense fit has and a sparse one does not; the family
        alone cannot decide it.
        """
        if self.basis_family == "monomial" and self._plan is not None:
            return self.evaluate_monomials(X_scaled)
        return self.evaluate_tensor_basis(X_scaled)

    def evaluate_tensor_basis(self, X_scaled: torch.Tensor) -> torch.Tensor:
        """(N, D) tensor-product design, ``prod_i p_{alpha_i}(z_i)``.

        Mirrors ``monomials._TensorPlan.evaluate``, clip included: outside the
        training box the tensor families saturate rather than diverging, and
        the caller's extrapolation guard still reports the excursion.
        Monomials are NOT clipped -- ``MonomialPlan`` does not, and a monomial
        fit describes a diverging model out there, so clipping would agree
        inside the box and describe a different model outside it.
        """
        z_all = (
            X_scaled if self.basis_family == "monomial"
            else torch.clamp(X_scaled, -1.0, 1.0)
        )
        idx = torch.as_tensor(
            self.multi_indices, dtype=torch.long, device=X_scaled.device
        )
        out = torch.ones(
            X_scaled.shape[0], self.multi_indices.shape[0],
            dtype=X_scaled.dtype, device=X_scaled.device,
        )
        for i in range(self.n_params):
            z = z_all[:, i]
            columns = [torch.ones_like(z)]
            if self.table_degree >= 1:
                columns.append(z)
            for k in range(1, self.table_degree):
                if self.basis_family == "legendre":
                    # (k+1) P_{k+1} = (2k+1) z P_k - k P_{k-1}
                    nxt = ((2 * k + 1) * z * columns[k] - k * columns[k - 1]) / (k + 1)
                elif self.basis_family == "monomial":
                    # z^{k+1} = z * z^k. Named rather than left to the else,
                    # which would evaluate monomials with Chebyshev's
                    # recurrence and return plausible numbers.
                    nxt = z * columns[k]
                else:
                    # T_{k+1} = 2 z T_k - T_{k-1}
                    nxt = 2.0 * z * columns[k] - columns[k - 1]
                columns.append(nxt)
            table = torch.stack(columns, dim=1)
            if self.basis_family == "legendre":
                degrees = torch.arange(
                    self.table_degree + 1, dtype=table.dtype, device=table.device
                )
                table = table * torch.sqrt(2.0 * degrees + 1.0)
            out = out * table[:, idx[:, i]]
        return out

    def evaluate_monomials(self, X_scaled: torch.Tensor) -> torch.Tensor:
        """Functional (N, D) monomial build from the P0.7 plan.

        No in-place writes, so torch.func.vmap can trace it.
        """
        plan = self._plan
        rows = [None] * plan.n_closure
        rows[0] = torch.ones(
            X_scaled.shape[0], dtype=X_scaled.dtype, device=X_scaled.device
        )
        xT = X_scaled.transpose(0, 1)
        for level in plan.levels:
            for j in level:
                rows[j] = rows[plan.parent[j]] * xT[plan.var[j]]
        buf = torch.stack(rows, dim=1)  # (N, D_closure)
        return buf[:, torch.as_tensor(plan.select, dtype=torch.long, device=X_scaled.device)]

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        if X.dim() == 0:
            if self.n_params != 1:
                raise ValueError(
                    f"scalar input given but the emulator has n_params = {self.n_params}"
                )
            X = X.reshape(1, 1)
        elif X.dim() == 1:
            if X.shape[0] != self.n_params:
                raise ValueError(
                    f"1-D input has {X.shape[0]} elements; expected n_params = "
                    f"{self.n_params}. Pass a 2-D (N, n_params) array otherwise."
                )
            X = X.unsqueeze(0)
        elif X.shape[-1] != self.n_params:
            raise ValueError(
                f"input has {X.shape[-1]} elements along its last axis; expected "
                f"n_params = {self.n_params}"
            )
        X = X.to(self.dtype)
        X_scaled = (X - self.input_mean) / self.input_scale
        Phi = self.evaluate_basis(X_scaled)
        Y = Phi @ self.coeffs * self.output_scale + self.output_mean
        modes = getattr(self, "output_modes", None)
        if modes is not None:
            # T-011: the fit carries k output modes; expand to m.
            Y = Y @ modes + self.output_offset
        if any(self.transform_codes):
            Y = self._apply_inverse(Y)
        return Y

    def _apply_inverse(self, Y: torch.Tensor) -> torch.Tensor:
        """Inverse per-column transform (1 exp, 2 sinh)."""
        codes = self.transform_codes
        if len(set(codes)) == 1:
            c = codes[0]
            return torch.exp(Y) if c == 1 else torch.sinh(Y) if c == 2 else Y
        cols = [Y[:, j] for j in range(Y.shape[1])]
        for j, c in enumerate(codes):
            if c == 1:
                cols[j] = torch.exp(cols[j])
            elif c == 2:
                cols[j] = torch.sinh(cols[j])
        return torch.stack(cols, dim=1)


# Backwards-compatible aliases.
MomentEmuModule = TorchMomentEmu


def create_torch_emulator(trained_emulator, dtype=torch.float64):
    """Create a PyTorch emulator from a trained MomentEmu."""
    return TorchMomentEmu(trained_emulator, dtype=dtype)


def demo_torch_autodiff():
    """Demonstrate PyTorch gradients on a small quadratic emulator."""
    from MomentEmu.emulator import PolyEmu

    rng = np.random.default_rng(42)
    X = rng.uniform(-1, 1, (200, 3))
    Y = (X[:, 0] ** 2 + X[:, 1] * X[:, 2]).reshape(-1, 1)
    emu = PolyEmu(X, Y, cross_validation=False, max_degree_forward=2, dim_reduction=False)
    torch_emu = TorchMomentEmu(emu)
    x = torch.tensor([0.5, 0.3, 0.2], requires_grad=True, dtype=torch.float64)
    y = torch_emu(x)
    y.sum().backward()
    print("prediction:", y.item())
    print("gradient:", x.grad)


if __name__ == "__main__":
    demo_torch_autodiff()
