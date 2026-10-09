# ode_rk.py
from __future__ import annotations
import torch
from typing import Callable

def rk4_step(
    t: torch.Tensor,                 # scalar
    x: torch.Tensor,                 # (G,)
    h: torch.Tensor,                 # scalar
    f: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],  # returns (G,)
) -> torch.Tensor:
    """
    Classic RK4.
    """
    k1 = f(t, x)
    k2 = f(t + 0.5 * h, x + 0.5 * h * k1)
    k3 = f(t + 0.5 * h, x + 0.5 * h * k2)
    k4 = f(t + h,       x + h * k3)
    return x + (h / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
