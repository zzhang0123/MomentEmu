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

from MomentEmu.guards import check_backend_supports, output_scale
from MomentEmu.monomials import MonomialPlan


class TorchMomentEmu(nn.Module):
    """PyTorch module for a fitted forward MomentEmu."""

    def __init__(self, trained_emulator, dtype=torch.float64):
        super().__init__()
        check_backend_supports(trained_emulator, "torch")
        self.dtype = dtype
        self.n_params = int(trained_emulator.n_params)
        self.n_outputs = int(trained_emulator.n_outputs)
        self.multi_indices = np.asarray(trained_emulator.forward_multi_indices)
        self._plan = MonomialPlan.build(self.multi_indices)
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
        Phi = self.evaluate_monomials(X_scaled)
        Y = Phi @ self.coeffs * self.output_scale + self.output_mean
        return Y


# Backwards-compatible aliases.
MomentEmuModule = TorchMomentEmu


def create_torch_emulator(trained_emulator, dtype=torch.float64):
    """Create a PyTorch emulator from a trained MomentEmu."""
    return TorchMomentEmu(trained_emulator, dtype=dtype)


def demo_torch_autodiff():
    """Demonstrate PyTorch gradients on a small quadratic emulator."""
    from MomentEmu.PolyEmu import PolyEmu

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
