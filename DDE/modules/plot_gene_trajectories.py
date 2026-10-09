# plot_gene_trajectories.py

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch


def gene_metrics(y_true, y_pred, eps: float = 1e-12):
    """
    Returns:
      rmse: root mean squared error
      r2:   coefficient of determination
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))

    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    r2 = 1.0 - ss_res / (ss_tot + eps)

    return rmse, r2


def plot_single_gene(
    gene_name: str,
    samples: dict,
    preds: dict,
    gene_idx: int,
    out_path: Path,
    K_fit: int | None = None,
    future: dict | None = None,   # {sample_name: (t_future, x_future)}
    test_sample_name: str | None = None,   # exact sample-dict key that is the held-out test sample
    show_legend: bool = True,
):
    """
    future:
      dict[sample_name] = (t_future, x_future[:, gene_idx])
    test_sample_name:
      key in `samples`/`preds` for the held-out test sample. Its "forecast"
      metric (R2/RMSE) is computed only on points after K_fit (i.e. the
      timepoints not used to fit history). If omitted, no sample is
      treated as the forecast/test sample.
    """

    n = len(samples)
    fig, axes = plt.subplots(n, 1, figsize=(6, 3 * n), sharex=False)
    if n == 1:
        axes = [axes]

    for ax, (name, sample) in zip(axes, samples.items()):
        t = sample.t.detach().cpu().numpy()
        y_true = sample.X[:, gene_idx].detach().cpu().numpy()
        y_pred = preds[name][:, gene_idx].detach().cpu().numpy()

        is_test = test_sample_name is not None and name == test_sample_name
        split = K_fit is not None and is_test and 0 < K_fit < len(t)

        if split:
            # calibration dates (used to fit history) = red, forecast-only dates (never seen) = blue
            ax.scatter(t[:K_fit], y_true[:K_fit], color="red", label="Observed (fit)", zorder=3)
            ax.scatter(t[K_fit:], y_true[K_fit:], color="blue", label="Observed (forecast)", zorder=3)
            ax.plot(t[:K_fit], y_pred[:K_fit], color="red", linewidth=2, label="Predicted (fit)")
            # start the forecast segment at K_fit-1 so the line is continuous, no gap
            ax.plot(t[K_fit - 1:], y_pred[K_fit - 1:], color="blue", linewidth=2, label="Predicted (forecast)")
        else:
            ax.scatter(t, y_true, color="black", label="Observed", zorder=3)
            ax.plot(t, y_pred, color="red", linewidth=2, label="Predicted")

        # Metrics
        if K_fit is not None and is_test and K_fit < len(t):
            rmse_f, r2_f = gene_metrics(y_true[K_fit:], y_pred[K_fit:])
            title_metric = f"Forecast R²={r2_f:.2f}, RMSE={rmse_f:.2f}"
        else:
            rmse_v, r2 = gene_metrics(y_true, y_pred)
            title_metric = f"R²={r2:.2f}, RMSE={rmse_v:.2f}"

        # forward simulation (future) -- extends past the last observed date
        if future is not None and name in future:
            t_fut, x_fut = future[name]

            # 안전하게 numpy 변환
            t_fut = np.asarray(t_fut)
            x_fut = np.asarray(x_fut)

            # gene_idx 슬라이싱 안정화
            if x_fut.ndim == 2:
                y_fut = x_fut[:, gene_idx]
            else:
                y_fut = x_fut

            ax.plot(
                np.concatenate([t[-1:], t_fut]),
                np.concatenate([[y_pred[-1]], y_fut]),
                color="green",
                linewidth=2,
                label="Forward (7d)"
            )

        if show_legend:
            ax.set_title(f"{name} | {title_metric}", fontsize=11)
        ax.set_ylabel("log2(TMM)")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("Time (days)")
    if show_legend:
        fig.suptitle(f"Gene: {gene_name}", fontsize=14, y=0.99)
        # gather from all subplots (not just axes[0]) since the fit/forecast
        # color split only appears on the test-sample subplot, which may not
        # be the first one
        seen = {}
        for ax in axes:
            h, l = ax.get_legend_handles_labels()
            for hi, li in zip(h, l):
                seen.setdefault(li, hi)
        fig.legend(seen.values(), seen.keys(), loc="upper center",
                   bbox_to_anchor=(0.5, 0.955), ncol=3, fontsize=8)

    fig.tight_layout(rect=[0, 0, 1.0, 0.87])
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


def plot_all_genes(
    samples: dict,
    preds: dict,
    gene_names: list,
    out_dir: str,
    max_genes: int | None = None,
    K_fit: int | None = None,
    future: dict | None = None,
    test_sample_name: str | None = None,
    show_legend: bool = True,
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    G = len(gene_names)
    genes_to_plot = range(G) if max_genes is None else range(min(G, max_genes))

    for i in genes_to_plot:
        gene = gene_names[i]
        fname = f"gene_{i:04d}_{gene}.png"
        plot_single_gene(
            gene_name=gene,
            samples=samples,
            preds=preds,
            gene_idx=i,
            out_path=out_dir / fname,
            K_fit=K_fit,
            future=future,
            test_sample_name=test_sample_name,
            show_legend=show_legend,
        )
