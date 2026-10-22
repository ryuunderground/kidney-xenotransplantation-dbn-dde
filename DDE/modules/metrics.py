# metrics.py
from __future__ import annotations
import torch

# NOTE: there is intentionally no pooled/global rmse() or r2_score() here.
# A metric computed by flattening (time x gene) together lets a handful of
# high-variance genes dominate the score, which can look good even when most
# individual genes are poorly predicted. Everything downstream should be
# built from per-gene metrics (per_gene_metrics / summarize_per_gene).


def per_gene_metrics(y_true: torch.Tensor, y_pred: torch.Tensor):
    """
    y_true, y_pred: (N,G)
    returns per-gene rmse/r2 tensors, each shape (G,)
    """
    resid2 = (y_true - y_pred) ** 2
    rmse_g = torch.sqrt(resid2.mean(dim=0))  # (G,)

    y_mean = y_true.mean(dim=0, keepdim=True)
    ss_res = resid2.sum(dim=0)
    ss_tot = ((y_true - y_mean) ** 2).sum(dim=0) + 1e-12
    r2_g = 1.0 - ss_res / ss_tot
    return rmse_g, r2_g


def summarize_per_gene(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    """
    Per-gene rmse/r2, summarized to scalars via mean/median across genes.
    This is the model-quality number that should be reported/selected on --
    never a pooled (time x gene) metric.
    """
    rmse_g, r2_g = per_gene_metrics(y_true, y_pred)
    return {
        "rmse": float(rmse_g.mean().item()),
        "rmse_median": float(rmse_g.median().item()),
        "r2": float(r2_g.mean().item()),
        "r2_median": float(r2_g.median().item()),
    }
