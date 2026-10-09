from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from scipy.optimize import curve_fit, OptimizeWarning
import warnings

# ── helper ────────────────────────────────────────────────────────────────────

def _pod_cutoff(time_points: np.ndarray, cutoff: float = 14.0):
    """Return boolean mask: True where time <= cutoff."""
    return time_points <= cutoff


def _latex_eq(slope: float, intercept: float, gene: str,
               rmse: float, r2: float) -> str:
    sign = "+" if intercept >= 0 else "-"
    return (
        rf"\text{{{gene}}}: & \quad y = {slope:.4f}\,t \;{sign}\; "
        rf"{abs(intercept):.4f}"
        rf"\quad \text{{RMSE}}={rmse:.4f},\; R^2={r2:.4f} \\"
    )


# ── core class ────────────────────────────────────────────────────────────────

class GeneLinearRegressor:
    """Per-gene linear regression of expression over time."""

    def __init__(self, csv_path: str | Path):
        df = pd.read_csv(csv_path, index_col=0)
        self.time_points = np.array(
            [int(col.replace("POD", "")) for col in df.columns], dtype=float
        )
        self.genes: list[str] = df.index.tolist()
        self.expression: np.ndarray = df.values      # (n_genes, n_times)
        self._models: list[LinearRegression] = []
        self._rmse: np.ndarray | None = None
        self._r2: np.ndarray | None = None

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self) -> "GeneLinearRegressor":
        X = self.time_points.reshape(-1, 1)
        self._models, rmse_list, r2_list = [], [], []
        for expr in self.expression:
            m = LinearRegression().fit(X, expr)
            y_pred = m.predict(X)
            rmse_list.append(np.sqrt(mean_squared_error(expr, y_pred)))
            r2_list.append(r2_score(expr, y_pred))
            self._models.append(m)
        self._rmse = np.array(rmse_list)
        self._r2 = np.array(r2_list)
        return self

    def predict(self, time_points: np.ndarray) -> np.ndarray:
        """Returns (n_genes, len(time_points))."""
        return np.stack(
            [m.predict(time_points.reshape(-1, 1)) for m in self._models], axis=0
        )

    # ── LaTeX export ──────────────────────────────────────────────────────────

    def save_latex(self, save_path: str | Path) -> None:
        lines = [
            r"\documentclass{article}",
            r"\usepackage{amsmath}",
            r"\begin{document}",
            r"\section*{Linear Regression Equations}",
            r"\begin{align*}",
        ]
        for i, gene in enumerate(self.genes):
            slope = float(self._models[i].coef_[0])
            intercept = float(self._models[i].intercept_)
            lines.append(_latex_eq(slope, intercept, gene,
                                   self._rmse[i], self._r2[i]))
        lines += [r"\end{align*}", r"\end{document}"]
        Path(save_path).write_text("\n".join(lines), encoding="utf-8")
        print(f"  saved → {save_path}")

    # ── grid plot (all genes) ─────────────────────────────────────────────────

    def plot_fits(
        self,
        save_path: str | Path,
        title: str = "",
        t_range: tuple[float, float] | None = None,
    ) -> None:
        """Multi-panel grid: one panel per gene, training data + fit line."""
        n = len(self.genes)
        ncols = 5
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
        axes = axes.flatten()

        t_lo = t_range[0] if t_range else self.time_points.min()
        t_hi = t_range[1] if t_range else self.time_points.max()
        t_dense = np.linspace(t_lo, t_hi, 300)

        for i, (gene, expr) in enumerate(zip(self.genes, self.expression)):
            ax = axes[i]
            y_line = self._models[i].predict(t_dense.reshape(-1, 1))
            ax.scatter(self.time_points, expr, color="tomato",
                       s=22, zorder=4, label="data")
            ax.plot(t_dense, y_line, color="black", lw=1.2, label="fit")
            ax.set_title(gene, fontsize=8, pad=2)
            ax.tick_params(labelsize=6)
            ax.annotate(
                f"RMSE={self._rmse[i]:.3f}\n$R^2$={self._r2[i]:.3f}",
                xy=(0.04, 0.96), xycoords="axes fraction", va="top", fontsize=6.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8),
            )

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right", fontsize=8,
                   bbox_to_anchor=(0.98, 0.01))
        if title:
            fig.suptitle(title, fontsize=12, y=1.005)
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved → {save_path}")

    # ── individual PDFs ───────────────────────────────────────────────────────

    def plot_individual_pdfs(
        self,
        out_dir: str | Path,
        t_range: tuple[float, float] | None = None,
    ) -> None:
        """Save one PDF per gene: {gene}_lr.pdf"""
        out_dir = Path(out_dir)
        t_lo = t_range[0] if t_range else self.time_points.min()
        t_hi = t_range[1] if t_range else self.time_points.max()
        t_dense = np.linspace(t_lo, t_hi, 300)

        for i, (gene, expr) in enumerate(zip(self.genes, self.expression)):
            fig, ax = plt.subplots(figsize=(5, 3.5))
            y_line = self._models[i].predict(t_dense.reshape(-1, 1))
            ax.scatter(self.time_points, expr, color="tomato",
                       s=30, zorder=4, label="data")
            ax.plot(t_dense, y_line, color="black", lw=1.4, label="fit")
            ax.set_xlabel("POD", fontsize=10)
            ax.set_ylabel("Expression", fontsize=10)
            ax.set_title(gene, fontsize=11)
            ax.legend(fontsize=8)

            slope = float(self._models[i].coef_[0])
            intercept = float(self._models[i].intercept_)
            sign = "+" if intercept >= 0 else "-"
            eq_str = (f"$y = {slope:.4f}\\,t \\;{sign}\\; {abs(intercept):.4f}$\n"
                      f"RMSE = {self._rmse[i]:.4f},  $R^2$ = {self._r2[i]:.4f}")
            ax.annotate(eq_str, xy=(0.04, 0.96), xycoords="axes fraction",
                        va="top", fontsize=8,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85))

            fig.tight_layout()
            fig.savefig(out_dir / f"{gene}_lr.pdf", bbox_inches="tight")
            plt.close(fig)
        print(f"  saved {len(self.genes)} individual PDFs → {out_dir}")

    # ── summary bar chart ─────────────────────────────────────────────────────

    def plot_summary(self, save_path: str | Path, title: str = "") -> None:
        x = np.arange(len(self.genes))
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(max(10, len(self.genes) * 0.45), 6), sharex=True
        )
        ax1.bar(x, self._rmse, color="tomato", alpha=0.85)
        ax1.set_ylabel("RMSE", fontsize=10)
        ax1.axhline(self._rmse.mean(), color="black", lw=1.2, ls="--",
                    label=f"mean = {self._rmse.mean():.3f}")
        ax1.legend(fontsize=8)

        ax2.bar(x, self._r2, color="steelblue", alpha=0.85)
        ax2.set_ylabel("$R^2$", fontsize=10)
        ax2.axhline(self._r2.mean(), color="black", lw=1.2, ls="--",
                    label=f"mean = {self._r2.mean():.3f}")
        ax2.axhline(0, color="gray", lw=0.8, ls=":")
        ax2.legend(fontsize=8)
        ax2.set_xticks(x)
        ax2.set_xticklabels(self.genes, rotation=45, ha="right", fontsize=7)

        if title:
            fig.suptitle(title, fontsize=12)
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved → {save_path}")

    def save_metrics(self, save_path: str | Path) -> pd.DataFrame:
        df = pd.DataFrame({"gene": self.genes, "rmse": self._rmse, "r2": self._r2})
        df.to_csv(save_path, index=False)
        print(f"  saved → {save_path}")
        return df


# ── missing-data forecaster ───────────────────────────────────────────────────

class MissingDataForecaster:
    """
    Fit line: trained on missing (early timepoints).
    Data points: from full inter_input.csv, colored by POD <= 14 vs > 14.
    """

    POD_CUTOFF = 14.0

    def __init__(self, missing_csv: str | Path, full_csv: str | Path):
        self._train = GeneLinearRegressor(missing_csv)
        self._full = GeneLinearRegressor(full_csv)
        assert self._train.genes == self._full.genes, "Gene lists must match."
        self.genes = self._train.genes

        seen = set(self._train.time_points)
        extra_idx = [i for i, t in enumerate(self._full.time_points) if t not in seen]
        self.extra_time = self._full.time_points[extra_idx]
        self.extra_expr = self._full.expression[:, extra_idx]

        self._rmse_extrap: np.ndarray | None = None
        self._r2_extrap: np.ndarray | None = None

    def fit(self) -> "MissingDataForecaster":
        self._train.fit()
        self._full.fit()   # NEW: also fit on full data for comparison line
        return self

    def evaluate_extrapolation(self) -> None:
        """RMSE/R² of missing model on unseen timepoints only."""
        pred = self._train.predict(self.extra_time)
        self._rmse_extrap = np.array([
            np.sqrt(mean_squared_error(self.extra_expr[i], pred[i]))
            for i in range(len(self.genes))
        ])
        self._r2_extrap = np.array([
            r2_score(self.extra_expr[i], pred[i])
            for i in range(len(self.genes))
        ])

    # ── LaTeX export ──────────────────────────────────────────────────────────

    def save_latex(self, save_path: str | Path) -> None:
        lines = [
            r"\documentclass{article}",
            r"\usepackage{amsmath}",
            r"\begin{document}",
            r"\section*{Linear Regression Equations (trained on missing data)}",
            r"\begin{align*}",
        ]
        for i, gene in enumerate(self.genes):
            slope = float(self._train._models[i].coef_[0])
            intercept = float(self._train._models[i].intercept_)
            lines.append(_latex_eq(slope, intercept, gene,
                                   self._train._rmse[i], self._train._r2[i]))
        lines += [r"\end{align*}", r"\end{document}"]
        Path(save_path).write_text("\n".join(lines), encoding="utf-8")
        print(f"  saved → {save_path}")

    # ── shared panel drawing ──────────────────────────────────────────────────

    def _draw_panel(
        self,
        ax: plt.Axes,
        gene_idx: int,
        t_dense: np.ndarray,
        show_legend: bool = False,
        show_full_fit: bool = True,
        fit_which: str = "both",   # "both" | "missing" | "full"
        show_annot: bool = True,
    ) -> None:
        gene = self.genes[gene_idx]
        full_t = self._full.time_points
        full_e = self._full.expression[gene_idx]

        mask_before = _pod_cutoff(full_t, self.POD_CUTOFF)
        mask_after = ~mask_before

        y_miss_line = self._train._models[gene_idx].predict(t_dense.reshape(-1, 1))
        y_full_line = self._full._models[gene_idx].predict(t_dense.reshape(-1, 1))

        ax.scatter(full_t[mask_before], full_e[mask_before],
                   color="blue", s=22, zorder=4,
                   label=f"$\\leq$POD{int(self.POD_CUTOFF)}")
        ax.scatter(full_t[mask_after], full_e[mask_after],
                   color="black", s=22, zorder=4,
                   label=f">POD{int(self.POD_CUTOFF)}")

        if fit_which in ("both", "missing"):
            ax.plot(t_dense, y_miss_line, color="red", lw=1.2, label="fit (missing)")
        if fit_which in ("both", "full") and show_full_fit:
            ax.plot(t_dense, y_full_line, color="gray", lw=1.0, ls="--", label="fit (full)")

        ax.set_title(gene, fontsize=8, pad=2)
        ax.tick_params(labelsize=6)

        slope_m = float(self._train._models[gene_idx].coef_[0])
        int_m   = float(self._train._models[gene_idx].intercept_)
        slope_f = float(self._full._models[gene_idx].coef_[0])
        int_f   = float(self._full._models[gene_idx].intercept_)
        sign_m  = "+" if int_m >= 0 else "-"
        sign_f  = "+" if int_f >= 0 else "-"
        if fit_which == "missing":
            eq_str = f"$y={slope_m:.3f}t{sign_m}{abs(int_m):.3f}$"
        elif fit_which == "full":
            eq_str = f"$y={slope_f:.3f}t{sign_f}{abs(int_f):.3f}$"
        else:
            eq_str = (f"miss: $y={slope_m:.3f}t{sign_m}{abs(int_m):.3f}$\n"
                      f"full: $y={slope_f:.3f}t{sign_f}{abs(int_f):.3f}$")

        # forecast metric: R2/RMSE on unseen future points only (POD > cutoff)
        if fit_which in ("both", "missing") and self._r2_extrap is not None:
            gi = gene_idx
            eq_str += (f"\nForecast(>POD{int(self.POD_CUTOFF)}): "
                       f"RMSE={self._rmse_extrap[gi]:.3f}, $R^2$={self._r2_extrap[gi]:.3f}")

        if show_annot:
            ax.annotate(eq_str, xy=(0.04, 0.96), xycoords="axes fraction",
                        va="top", fontsize=6,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.85))

        if show_legend:
            ax.legend(fontsize=6, loc="lower right")

    # ── grid plot ─────────────────────────────────────────────────────────────

    def plot_fits(self, save_path: str | Path, show_full_fit: bool = True) -> None:
        n = len(self.genes)
        ncols = 5
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.5, nrows * 2.8))
        axes = axes.flatten()

        t_lo = self._full.time_points.min()
        t_hi = self._full.time_points.max()
        t_dense = np.linspace(t_lo, t_hi, 300)

        for i in range(n):
            self._draw_panel(axes[i], i, t_dense, show_legend=(i == 0),
                             show_full_fit=show_full_fit)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right", fontsize=8,
                   bbox_to_anchor=(0.98, 0.01))
        fig.suptitle(
            "Linear Regression — miss fit (red) vs full fit (gray--), points from full data\n"
            f"(blue $\\leq$POD{int(self.POD_CUTOFF)}, black >POD{int(self.POD_CUTOFF)})",
            fontsize=11, y=1.01,
        )
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        fig.savefig(Path(save_path).with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved → {save_path}")

    # ── individual PDFs ───────────────────────────────────────────────────────

    def plot_individual_pdfs(self, out_dir: str | Path,
                              show_legend: bool = True,
                              show_full_fit: bool = True,
                              fit_which: str = "both",
                              show_annot: bool = True) -> None:
        """Save one PDF + PNG per gene: {gene}_lr.pdf / {gene}_lr.png"""
        out_dir = Path(out_dir)
        full_t = self._full.time_points
        t_dense = np.linspace(full_t.min(), full_t.max(), 300)

        for i, gene in enumerate(self.genes):
            fig, ax = plt.subplots(figsize=(5, 3.5))
            self._draw_panel(ax, i, t_dense, show_legend=show_legend,
                             show_full_fit=show_full_fit, fit_which=fit_which,
                             show_annot=show_annot)
            ax.set_xlabel("POD", fontsize=10)
            ax.set_ylabel("Expression", fontsize=10)
            fig.tight_layout()
            fig.savefig(out_dir / f"{gene}_lr.pdf", bbox_inches="tight")
            fig.savefig(out_dir / f"{gene}_lr.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
        print(f"  saved {len(self.genes)} individual PDF+PNG pairs → {out_dir}")

    def save_metrics(self, save_path: str | Path) -> pd.DataFrame:
        """
        Comparison group:
          - missing_vs_missing : fit on missing data, evaluated on missing data
        """
        df = pd.DataFrame({
            "gene": self.genes,
            "missing_vs_missing_rmse": self._train._rmse,
            "missing_vs_missing_r2":   self._train._r2,
            f"forecast_gt_POD{int(self.POD_CUTOFF)}_rmse": self._rmse_extrap,
            f"forecast_gt_POD{int(self.POD_CUTOFF)}_r2":   self._r2_extrap,
        })
        df.to_csv(save_path, index=False)
        print(f"  saved → {save_path}")
        return df



# ── exponential curve fit  y = a·e^(bx) + c  (a < 0, b < 0) ─────────────────

def _exp_model(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    return a * np.exp(b * x) + c


class ExpCurveRegressor:
    """
    Nonlinear least-squares fit of  y = a·exp(b·t) + c
    with constraints  a < 0,  b < 0  (saturating growth toward asymptote c).

    Uses scipy.optimize.curve_fit with multiple random initialisations
    and picks the run with lowest RMSE.
    """

    # candidate initial (a, b) pairs; c is always initialised from data
    _A0 = [-1.0, -5.0, -10.0, -0.5]
    _B0 = [-0.05, -0.1, -0.2, -0.5]

    def __init__(self, csv_path: str | Path, genes: list[str] | None = None):
        df = pd.read_csv(csv_path, index_col=0)
        if genes:
            df = df.loc[genes]
        self.time_points = np.array(
            [int(c.replace("POD", "")) for c in df.columns], dtype=float
        )
        self.genes: list[str] = df.index.tolist()
        self.expression: np.ndarray = df.values       # (n_genes, n_times)
        # per-gene result dict: a, b, c, rmse, r2, valid
        self._results: list[dict] = []

    # ── fit ───────────────────────────────────────────────────────────────────

    def fit(self) -> "ExpCurveRegressor":
        t = self.time_points
        bounds = ([-np.inf, -np.inf, -np.inf], [0.0, 0.0, np.inf])

        for expr in self.expression:
            c0 = float(np.max(expr))
            best: dict | None = None

            for a0 in self._A0:
                for b0 in self._B0:
                    try:
                        with warnings.catch_warnings():
                            warnings.simplefilter("ignore", OptimizeWarning)
                            popt, _ = curve_fit(
                                _exp_model, t, expr,
                                p0=[a0, b0, c0],
                                bounds=bounds,
                                maxfev=20000,
                            )
                        a, b, c = popt
                        y_pred = _exp_model(t, a, b, c)
                        rmse = float(np.sqrt(mean_squared_error(expr, y_pred)))
                        r2   = float(r2_score(expr, y_pred))
                        if best is None or rmse < best["rmse"]:
                            best = {"valid": True, "a": a, "b": b, "c": c,
                                    "rmse": rmse, "r2": r2}
                    except Exception:
                        continue

            if best is None:
                best = {"valid": False, "a": np.nan, "b": np.nan, "c": np.nan,
                        "rmse": np.nan, "r2": np.nan}
            self._results.append(best)
        return self

    # ── single-gene figure ────────────────────────────────────────────────────

    def _make_gene_fig(self, gene_idx: int) -> plt.Figure:
        gene = self.genes[gene_idx]
        expr = self.expression[gene_idx]
        t    = self.time_points
        res  = self._results[gene_idx]
        t_dense = np.linspace(t.min(), t.max(), 300)

        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.scatter(t, expr, color="tomato", s=35, zorder=4, label="data")

        if res["valid"]:
            a, b, c = res["a"], res["b"], res["c"]
            ax.plot(t_dense, _exp_model(t_dense, a, b, c),
                    color="black", lw=1.6, label="fit")
            # asymptote
            ax.axhline(c, color="steelblue", lw=1.0, ls="--",
                       label=f"asymptote $c={c:.3f}$")

            s_a = f"{a:.4f}"
            s_b = f"{b:.4f}"
            s_c = f"+{c:.4f}" if c >= 0 else f"{c:.4f}"
            eq  = f"$y = {s_a}\\,e^{{{s_b}\\,t}} {s_c}$"
            annot = f"{eq}\nRMSE = {res['rmse']:.4f},  $R^2$ = {res['r2']:.4f}"
        else:
            annot = "Fit failed\n(curve_fit did not converge)"

        ax.set_title(gene, fontsize=12, fontweight="bold")
        ax.set_xlabel("POD", fontsize=10)
        ax.set_ylabel("Expression", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.annotate(annot, xy=(0.04, 0.96), xycoords="axes fraction",
                    va="top", fontsize=8.5,
                    bbox=dict(boxstyle="round,pad=0.35", fc="white", alpha=0.9))
        ax.legend(fontsize=8)
        fig.tight_layout()
        return fig

    # ── individual PDFs ───────────────────────────────────────────────────────

    def plot_individual_pdfs(self, out_dir: str | Path) -> None:
        out_dir = Path(out_dir)
        for i, gene in enumerate(self.genes):
            fig = self._make_gene_fig(i)
            fig.savefig(out_dir / f"{gene}_exp.pdf", bbox_inches="tight")
            plt.close(fig)
        print(f"  saved {len(self.genes)} individual PDFs → {out_dir}")

    # ── grid overview ─────────────────────────────────────────────────────────

    def plot_grid(self, save_path: str | Path) -> None:
        n = len(self.genes)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(ncols * 4.5, nrows * 3.5))
        axes = np.array(axes).flatten()
        t = self.time_points
        t_dense = np.linspace(t.min(), t.max(), 300)

        for i, (gene, expr) in enumerate(zip(self.genes, self.expression)):
            ax  = axes[i]
            res = self._results[i]
            ax.scatter(t, expr, color="tomato", s=25, zorder=4, label="data")

            if res["valid"]:
                a, b, c = res["a"], res["b"], res["c"]
                ax.plot(t_dense, _exp_model(t_dense, a, b, c),
                        color="black", lw=1.4, label="fit")
                ax.axhline(c, color="steelblue", lw=0.9, ls="--",
                           label=f"asymptote={c:.2f}")
                s_a = f"{a:.3f}"
                s_b = f"{b:.3f}"
                s_c = f"+{c:.3f}" if c >= 0 else f"{c:.3f}"
                annot = (f"$y={s_a}e^{{{s_b}t}}{s_c}$\n"
                         f"RMSE={res['rmse']:.3f}  $R^2$={res['r2']:.3f}")
            else:
                annot = "Fit failed"

            ax.set_title(gene, fontsize=10, fontweight="bold")
            ax.set_xlabel("POD", fontsize=8)
            ax.set_ylabel("Expression", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.annotate(annot, xy=(0.04, 0.96), xycoords="axes fraction",
                        va="top", fontsize=7.5,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.88))
            ax.legend(fontsize=7)

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        fig.suptitle(r"Exponential fit: $y = a\,e^{bt} + c$  ($a<0,\,b<0$)",
                     fontsize=12)
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved → {save_path}")

    # ── LaTeX ─────────────────────────────────────────────────────────────────

    def save_latex(self, save_path: str | Path) -> None:
        lines = [
            r"\documentclass{article}",
            r"\usepackage{amsmath}",
            r"\begin{document}",
            r"\section*{Exponential Fit: $y = a\,e^{bt} + c$ \quad ($a<0,\;b<0$)}",
            r"\begin{align*}",
        ]
        for i, gene in enumerate(self.genes):
            res = self._results[i]
            if res["valid"]:
                a, b, c = res["a"], res["b"], res["c"]
                s_c = f"+{c:.4f}" if c >= 0 else f"{c:.4f}"
                lines.append(
                    rf"\text{{{gene}}}: & \quad y = {a:.4f}\,e^{{{b:.4f}\,t}} {s_c}"
                    rf"\quad \text{{RMSE}}={res['rmse']:.4f},\; R^2={res['r2']:.4f} \\"
                )
            else:
                lines.append(rf"\text{{{gene}}}: & \quad \text{{Fit failed}} \\")
        lines += [r"\end{align*}", r"\end{document}"]
        Path(save_path).write_text("\n".join(lines), encoding="utf-8")
        print(f"  saved → {save_path}")

    # ── metrics CSV ───────────────────────────────────────────────────────────

    def save_metrics(self, save_path: str | Path) -> pd.DataFrame:
        df = pd.DataFrame([
            {"gene": g,
             "a": r["a"], "b": r["b"], "c": r["c"],
             "rmse": r["rmse"], "r2": r["r2"]}
            for g, r in zip(self.genes, self._results)
        ])
        df.to_csv(save_path, index=False)
        print(f"  saved → {save_path}")
        return df


# ── exponential fit on missing data, evaluated on full ───────────────────────

class ExpMissingForecaster:
    """
    Fit  y = a·exp(b·t) + c  (a<0, b<0) using missing (early timepoints) only.
    Plot shows all inter_input.csv points coloured by POD <= 14 vs > 14.
    Metrics are computed for two comparisons:
      - missing_vs_missing : fit evaluated on the training (missing) timepoints
      - missing_vs_full    : fit evaluated ONLY on unseen future timepoints
                              (POD > cutoff, i.e. the forecast region)
    """

    POD_CUTOFF = 14.0

    def __init__(self, missing_csv: str | Path, full_csv: str | Path,
                 genes: list[str] | None = None):
        self._train = ExpCurveRegressor(missing_csv, genes=genes)
        # load full data for the same genes
        full_df = pd.read_csv(full_csv, index_col=0)
        if genes:
            full_df = full_df.loc[genes]
        self.genes = self._train.genes
        self.full_time = np.array(
            [int(c.replace("POD", "")) for c in full_df.columns], dtype=float
        )
        self.full_expr = full_df.values          # (n_genes, n_full_times)

        self._rmse_miss: np.ndarray | None = None
        self._r2_miss:   np.ndarray | None = None
        self._rmse_full: np.ndarray | None = None
        self._r2_full:   np.ndarray | None = None

    # ── fit & evaluate ────────────────────────────────────────────────────────

    def fit(self) -> "ExpMissingForecaster":
        self._train.fit()
        # NEW: also fit on full data for comparison line
        import warnings
        self._results_full = []
        for expr in self.full_expr:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    popt, _ = curve_fit(
                        _exp_model, self.full_time, expr,
                        bounds=([-np.inf, -np.inf, -np.inf], [0.0, 0.0, np.inf]),
                        p0=[-1.0, -0.05, float(expr[-1])], maxfev=20000,
                    )
                self._results_full.append({"valid": True, "a": popt[0], "b": popt[1], "c": popt[2]})
            except Exception:
                self._results_full.append({"valid": False})
        return self

    def evaluate(self) -> None:
        t_miss = self._train.time_points
        rmse_m, r2_m, rmse_f, r2_f = [], [], [], []

        for i, res in enumerate(self._train._results):
            if not res["valid"]:
                rmse_m.append(np.nan); r2_m.append(np.nan)
                rmse_f.append(np.nan); r2_f.append(np.nan)
                continue
            a, b, c = res["a"], res["b"], res["c"]

            # vs missing (training) timepoints
            y_miss_pred = _exp_model(t_miss, a, b, c)
            y_miss_true = self._train.expression[i]
            rmse_m.append(float(np.sqrt(mean_squared_error(y_miss_true, y_miss_pred))))
            r2_m.append(float(r2_score(y_miss_true, y_miss_pred)))

            # vs unseen future timepoints only (POD > cutoff) -- forecast metric
            mask_fc = self.full_time > self.POD_CUTOFF
            y_fc_pred = _exp_model(self.full_time[mask_fc], a, b, c)
            y_fc_true = self.full_expr[i, mask_fc]
            rmse_f.append(float(np.sqrt(mean_squared_error(y_fc_true, y_fc_pred))))
            r2_f.append(float(r2_score(y_fc_true, y_fc_pred)))

        self._rmse_miss = np.array(rmse_m)
        self._r2_miss   = np.array(r2_m)
        self._rmse_full = np.array(rmse_f)
        self._r2_full   = np.array(r2_f)

    # ── shared panel drawing ──────────────────────────────────────────────────

    def _draw_panel(self, ax: plt.Axes, gene_idx: int,
                    t_dense: np.ndarray, show_annot: bool = True,
                    show_full_fit: bool = True, fit_which: str = "both") -> None:
        res  = self._train._results[gene_idx]
        gene = self.genes[gene_idx]

        mask_before = self.full_time <= self.POD_CUTOFF
        mask_after  = ~mask_before
        expr_full   = self.full_expr[gene_idx]

        ax.scatter(self.full_time[mask_before], expr_full[mask_before],
                   color="blue", s=28, zorder=4,
                   label=f"$\\leq$POD{int(self.POD_CUTOFF)}")
        ax.scatter(self.full_time[mask_after],  expr_full[mask_after],
                   color="black", s=28, zorder=4,
                   label=f">POD{int(self.POD_CUTOFF)}")

        # y-axis limits from data (set before axhline to avoid axis expansion)
        y_lo = expr_full.min()
        y_hi = expr_full.max()
        pad  = (y_hi - y_lo) * 0.18
        ax.set_ylim(y_lo - pad, y_hi + pad)

        miss_valid = res["valid"] and fit_which in ("both", "missing")
        if miss_valid:
            a, b, c = res["a"], res["b"], res["c"]
            ax.plot(t_dense, _exp_model(t_dense, a, b, c),
                    color="red", lw=1.5, label="fit (missing)")

        if fit_which in ("both", "full") and show_full_fit and hasattr(self, "_results_full"):
            res_f = self._results_full[gene_idx]
            if res_f["valid"]:
                y_f = _exp_model(t_dense, res_f["a"], res_f["b"], res_f["c"])
                y_f_clipped = np.clip(y_f, y_lo - pad, y_hi + pad)
                ax.plot(t_dense, y_f_clipped, color="gray", lw=1.0, ls="--",
                        label="fit (full)")

        if miss_valid:
            if y_lo - pad <= c <= y_hi + pad:
                ax.axhline(c, color="steelblue", lw=0.9, ls="--",
                           label=f"asymptote={c:.2f}")
            else:
                ax.axhline(y_hi + pad, color="none")

            if show_annot:
                s_a = f"{a:.3f}"
                s_b = f"{b:.3f}"
                s_c = f"+{c:.3f}" if c >= 0 else f"{c:.3f}"
                gi  = gene_idx
                annot = (
                    f"$y={s_a}e^{{{s_b}t}}{s_c}$\n"
                    f"miss: RMSE={self._rmse_miss[gi]:.3f}, $R^2$={self._r2_miss[gi]:.3f}\n"
                    f"forecast(>POD{int(self.POD_CUTOFF)}): "
                    f"RMSE={self._rmse_full[gi]:.3f}, $R^2$={self._r2_full[gi]:.3f}"
                )
                ax.annotate(annot, xy=(0.04, 0.96), xycoords="axes fraction",
                            va="top", fontsize=7.5,
                            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))
        elif show_annot and not res["valid"]:
            ax.annotate("Fit failed", xy=(0.04, 0.96), xycoords="axes fraction",
                        va="top", fontsize=7.5,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))

        ax.set_title(gene, fontsize=10, fontweight="bold")
        ax.set_xlabel("POD", fontsize=8)
        ax.set_ylabel("Expression", fontsize=8)
        ax.tick_params(labelsize=7)

    # ── grid plot ─────────────────────────────────────────────────────────────

    def plot_grid(self, save_path: str | Path, show_annot: bool = True) -> None:
        n = len(self.genes)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(ncols * 4.5, nrows * 3.8))
        axes = np.array(axes).flatten()
        t_dense = np.linspace(self.full_time.min(), self.full_time.max(), 300)

        for i in range(n):
            self._draw_panel(axes[i], i, t_dense, show_annot=show_annot, fit_which="missing")

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", fontsize=9,
                   bbox_to_anchor=(0.5, 0.0), ncol=len(handles),
                   frameon=True, borderaxespad=0.5)
        fig.suptitle(
            r"Exp fit $y=ae^{bt}+c$ ($a<0,b<0$) — trained on missing data"
            f"\n(blue $\\leq$POD{int(self.POD_CUTOFF)}, orange >POD{int(self.POD_CUTOFF)})",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        fig.savefig(Path(save_path).with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved → {save_path}")

    # ── individual PDFs ───────────────────────────────────────────────────────

    def plot_individual_pdfs(self, out_dir: str | Path,
                              show_annot: bool = True,
                              show_legend: bool = True,
                              show_full_fit: bool = True,
                              fit_which: str = "both") -> None:
        out_dir = Path(out_dir)
        t_dense = np.linspace(self.full_time.min(), self.full_time.max(), 300)
        for i, gene in enumerate(self.genes):
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            self._draw_panel(ax, i, t_dense, show_annot=show_annot,
                             show_full_fit=show_full_fit, fit_which=fit_which)
            if show_legend:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, fontsize=8,
                          loc="upper left", bbox_to_anchor=(1.02, 1.0),
                          borderaxespad=0, frameon=True)
            fig.tight_layout()
            fig.savefig(out_dir / f"{gene}_exp_missing.pdf", bbox_inches="tight")
            fig.savefig(out_dir / f"{gene}_exp_missing.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
        print(f"  saved {len(self.genes)} individual PDF+PNG pairs → {out_dir}")

    # ── LaTeX ─────────────────────────────────────────────────────────────────

    def save_latex(self, save_path: str | Path) -> None:
        lines = [
            r"\documentclass{article}",
            r"\usepackage{amsmath}",
            r"\begin{document}",
            r"\section*{Exp Fit (trained on missing): $y = a\,e^{bt} + c$ \quad ($a<0,\;b<0$)}",
            r"\begin{align*}",
        ]
        for i, gene in enumerate(self.genes):
            res = self._train._results[i]
            if res["valid"]:
                a, b, c = res["a"], res["b"], res["c"]
                s_c = f"+{c:.4f}" if c >= 0 else f"{c:.4f}"
                lines.append(
                    rf"\text{{{gene}}}: & \quad y = {a:.4f}\,e^{{{b:.4f}\,t}} {s_c} \\"
                )
            else:
                lines.append(rf"\text{{{gene}}}: & \quad \text{{Fit failed}} \\")
        lines += [r"\end{align*}", r"\end{document}"]
        Path(save_path).write_text("\n".join(lines), encoding="utf-8")
        print(f"  saved → {save_path}")

    # ── metrics CSV ───────────────────────────────────────────────────────────

    def save_metrics(self, save_path: str | Path) -> pd.DataFrame:
        df = pd.DataFrame([
            {"gene":                   g,
             "a":                      self._train._results[i]["a"],
             "b":                      self._train._results[i]["b"],
             "c":                      self._train._results[i]["c"],
             "missing_vs_missing_rmse": self._rmse_miss[i],
             "missing_vs_missing_r2":   self._r2_miss[i],
             "missing_vs_full_rmse":    self._rmse_full[i],
             "missing_vs_full_r2":      self._r2_full[i]}
            for i, g in enumerate(self.genes)
        ])
        df.to_csv(save_path, index=False)
        print(f"  saved → {save_path}")
        return df


# ── log-x fit on missing data, evaluated on full ──────────────────────────────

class LogXMissingForecaster:
    """
    Fit  y = a·ln(t) + b  using missing (early timepoints) only.
    Plot shows all inter_input.csv points coloured by POD <= 14 vs > 14.
    Metrics:
      - missing_vs_missing : fit evaluated on the training (missing) timepoints
      - missing_vs_full    : fit evaluated ONLY on unseen future timepoints
                              (POD > cutoff, i.e. the forecast region)
    """

    POD_CUTOFF = 14.0

    def __init__(self, missing_csv: str | Path, full_csv: str | Path,
                 genes: list[str] | None = None):
        miss_df = pd.read_csv(missing_csv, index_col=0)
        full_df = pd.read_csv(full_csv,    index_col=0)
        if genes:
            miss_df = miss_df.loc[genes]
            full_df = full_df.loc[genes]
        assert list(miss_df.index) == list(full_df.index), "Gene lists must match."

        self.genes     = miss_df.index.tolist()
        self.miss_time = np.array(
            [int(c.replace("POD", "")) for c in miss_df.columns], dtype=float
        )
        self.full_time = np.array(
            [int(c.replace("POD", "")) for c in full_df.columns], dtype=float
        )
        self.miss_expr = miss_df.values   # (n_genes, n_miss)
        self.full_expr = full_df.values   # (n_genes, n_full)

        self._models:    list[LinearRegression] = []
        self._rmse_miss: np.ndarray | None = None
        self._r2_miss:   np.ndarray | None = None
        self._rmse_full: np.ndarray | None = None
        self._r2_full:   np.ndarray | None = None

    # ── fit & evaluate ────────────────────────────────────────────────────────

    def fit(self) -> "LogXMissingForecaster":
        X = np.log(self.miss_time).reshape(-1, 1)
        self._models = []
        for expr in self.miss_expr:
            self._models.append(LinearRegression().fit(X, expr))
        # NEW: also fit on full data for comparison line
        X_full = np.log(self.full_time).reshape(-1, 1)
        self._models_full = []
        for expr in self.full_expr:
            self._models_full.append(LinearRegression().fit(X_full, expr))
        return self

    def evaluate(self) -> None:
        mask_fc = self.full_time > self.POD_CUTOFF
        rmse_m, r2_m, rmse_f, r2_f = [], [], [], []
        for i, m in enumerate(self._models):
            # vs missing (training) timepoints
            y_miss_pred = m.predict(np.log(self.miss_time).reshape(-1, 1))
            rmse_m.append(float(np.sqrt(mean_squared_error(self.miss_expr[i], y_miss_pred))))
            r2_m.append(float(r2_score(self.miss_expr[i], y_miss_pred)))
            # vs unseen future timepoints only (POD > cutoff) -- forecast metric
            y_fc_pred = m.predict(np.log(self.full_time[mask_fc]).reshape(-1, 1))
            y_fc_true = self.full_expr[i, mask_fc]
            rmse_f.append(float(np.sqrt(mean_squared_error(y_fc_true, y_fc_pred))))
            r2_f.append(float(r2_score(y_fc_true, y_fc_pred)))

        self._rmse_miss = np.array(rmse_m)
        self._r2_miss   = np.array(r2_m)
        self._rmse_full = np.array(rmse_f)
        self._r2_full   = np.array(r2_f)

    # ── shared panel drawing ──────────────────────────────────────────────────

    def _draw_panel(self, ax: plt.Axes, gene_idx: int,
                    t_dense: np.ndarray, show_full_fit: bool = True,
                    fit_which: str = "both", show_annot: bool = True) -> None:
        m    = self._models[gene_idx]
        gene = self.genes[gene_idx]

        mask_before = self.full_time <= self.POD_CUTOFF
        mask_after  = ~mask_before
        expr_full   = self.full_expr[gene_idx]

        ax.scatter(self.full_time[mask_before], expr_full[mask_before],
                   color="blue", s=28, zorder=4,
                   label=f"$\\leq$POD{int(self.POD_CUTOFF)}")
        ax.scatter(self.full_time[mask_after],  expr_full[mask_after],
                   color="black", s=28, zorder=4,
                   label=f">POD{int(self.POD_CUTOFF)}")

        if fit_which in ("both", "missing"):
            y_curve = m.predict(np.log(t_dense).reshape(-1, 1))
            ax.plot(t_dense, y_curve, color="red", lw=1.5, label="fit (missing)")

        if fit_which in ("both", "full") and show_full_fit and hasattr(self, "_models_full"):
            m_full = self._models_full[gene_idx]
            y_full_curve = m_full.predict(np.log(t_dense).reshape(-1, 1))
            ax.plot(t_dense, y_full_curve, color="gray", lw=1.0, ls="--", label="fit (full)")

        a   = float(m.coef_[0])
        b   = float(m.intercept_)
        s_b = f"+{b:.3f}" if b >= 0 else f"{b:.3f}"
        a_f = float(self._models_full[gene_idx].coef_[0]) if hasattr(self, "_models_full") else None
        b_f = float(self._models_full[gene_idx].intercept_) if hasattr(self, "_models_full") else None
        s_b_f = (f"+{b_f:.3f}" if b_f >= 0 else f"{b_f:.3f}") if b_f is not None else ""
        i   = gene_idx
        if fit_which == "missing":
            annot = (f"$y={a:.3f}\\ln t{s_b}$\n"
                      f"RMSE(miss)={self._rmse_miss[i]:.3f}, $R^2$(miss)={self._r2_miss[i]:.3f}\n"
                      f"forecast(>POD{int(self.POD_CUTOFF)}): "
                      f"RMSE={self._rmse_full[i]:.3f}, $R^2$={self._r2_full[i]:.3f}")
        elif fit_which == "full" and a_f is not None:
            annot = f"$y={a_f:.3f}\\ln t{s_b_f}$\nRMSE(full)={self._rmse_full[i]:.3f}"
        else:
            annot = (
                f"miss: $y={a:.3f}\\ln t{s_b}$\n"
                + (f"full:  $y={a_f:.3f}\\ln t{s_b_f}$\n" if a_f is not None else "")
                + f"RMSE(miss)={self._rmse_miss[i]:.3f}, $R^2$(miss)={self._r2_miss[i]:.3f}\n"
                + f"forecast(>POD{int(self.POD_CUTOFF)}): "
                + f"RMSE={self._rmse_full[i]:.3f}, $R^2$={self._r2_full[i]:.3f}"
            )
        ax.set_title(gene, fontsize=10, fontweight="bold")
        ax.set_xlabel("POD", fontsize=8)
        ax.set_ylabel("Expression", fontsize=8)
        ax.tick_params(labelsize=7)
        if show_annot:
            ax.annotate(annot, xy=(0.04, 0.96), xycoords="axes fraction",
                        va="top", fontsize=7.5,
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.9))

    # ── grid plot ─────────────────────────────────────────────────────────────

    def plot_grid(self, save_path: str | Path) -> None:
        n = len(self.genes)
        ncols = min(n, 3)
        nrows = (n + ncols - 1) // ncols
        fig, axes = plt.subplots(nrows, ncols,
                                  figsize=(ncols * 4.5, nrows * 3.8))
        axes = np.array(axes).flatten()
        t_dense = np.linspace(self.full_time.min(), self.full_time.max(), 300)

        for i in range(n):
            self._draw_panel(axes[i], i, t_dense, fit_which="missing")

        for j in range(n, len(axes)):
            axes[j].set_visible(False)

        handles, labels = axes[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="lower right", fontsize=8,
                   bbox_to_anchor=(0.98, 0.01))
        fig.suptitle(
            r"Log-x fit $y = a\ln t + b$ — trained on missing data"
            f"\n(blue $\\leq$POD{int(self.POD_CUTOFF)}, orange >POD{int(self.POD_CUTOFF)})",
            fontsize=11,
        )
        fig.tight_layout()
        fig.savefig(save_path, bbox_inches="tight")
        fig.savefig(Path(save_path).with_suffix(".png"), dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"  saved → {save_path}")

    # ── individual PDFs ───────────────────────────────────────────────────────

    def plot_individual_pdfs(self, out_dir: str | Path,
                              show_legend: bool = True,
                              show_full_fit: bool = True,
                              fit_which: str = "both",
                              show_annot: bool = True) -> None:
        out_dir = Path(out_dir)
        t_dense = np.linspace(self.full_time.min(), self.full_time.max(), 300)
        for i, gene in enumerate(self.genes):
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            self._draw_panel(ax, i, t_dense, show_full_fit=show_full_fit,
                             fit_which=fit_which, show_annot=show_annot)
            if show_legend:
                handles, labels = ax.get_legend_handles_labels()
                ax.legend(handles, labels, fontsize=8)
            fig.tight_layout()
            fig.savefig(out_dir / f"{gene}_logx_missing.pdf", bbox_inches="tight")
            fig.savefig(out_dir / f"{gene}_logx_missing.png", dpi=300, bbox_inches="tight")
            plt.close(fig)
        print(f"  saved {len(self.genes)} individual PDF+PNG pairs → {out_dir}")

    # ── LaTeX ─────────────────────────────────────────────────────────────────

    def save_latex(self, save_path: str | Path) -> None:
        lines = [
            r"\documentclass{article}",
            r"\usepackage{amsmath}",
            r"\begin{document}",
            r"\section*{Log-x Fit (trained on missing): $y = a\,\ln t + b$}",
            r"\begin{align*}",
        ]
        for i, gene in enumerate(self.genes):
            a   = float(self._models[i].coef_[0])
            b   = float(self._models[i].intercept_)
            s_b = f"+{b:.4f}" if b >= 0 else f"{b:.4f}"
            lines.append(
                rf"\text{{{gene}}}: & \quad y = {a:.4f}\,\ln t {s_b}"
                rf"\quad \text{{RMSE(miss)}}={self._rmse_miss[i]:.4f},"
                rf"\; R^2\text{{(miss)}}={self._r2_miss[i]:.4f} \\"
            )
        lines += [r"\end{align*}", r"\end{document}"]
        Path(save_path).write_text("\n".join(lines), encoding="utf-8")
        print(f"  saved → {save_path}")

    # ── metrics CSV ───────────────────────────────────────────────────────────

    def save_metrics(self, save_path: str | Path) -> pd.DataFrame:
        df = pd.DataFrame([
            {"gene":                   g,
             "slope_a":                float(self._models[i].coef_[0]),
             "intercept_b":            float(self._models[i].intercept_),
             "missing_vs_missing_rmse": self._rmse_miss[i],
             "missing_vs_missing_r2":   self._r2_miss[i],
             "missing_vs_full_rmse":    self._rmse_full[i],
             "missing_vs_full_r2":      self._r2_full[i]}
            for i, g in enumerate(self.genes)
        ])
        df.to_csv(save_path, index=False)
        print(f"  saved → {save_path}")
        return df
