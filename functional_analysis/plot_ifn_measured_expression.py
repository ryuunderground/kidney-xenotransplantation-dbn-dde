"""
Plot measured (non-fitted) expression of the IFN/antiviral genes for each
sample (short/inter/long), one figure per sample with all genes overlaid.

Input: DBN-DDE_inputs/{short,inter,long}_input_ifn.csv (GeneName x POD, logTMM)
Output: functional_analysis/ifn_measured_expression/{sample}_ifn_measured_expression.{png,pdf}
"""

import re
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "DBN-DDE_inputs"
OUT_DIR = Path(__file__).resolve().parent / "ifn_measured_expression"

SAMPLES = {
    "short": {"csv": DATA_DIR / "short_input.csv", "color": "red"},
    "inter": {"csv": DATA_DIR / "inter_input.csv", "color": "green"},
    "long":  {"csv": DATA_DIR / "long_input.csv",  "color": "blue"},
}

ALPHA = 0.35

# Fixed inches-per-POD-day scale so the three figures are directly comparable:
# short (shortest POD span) is narrowest, long (longest span) is widest, and
# 1 POD unit occupies the same physical width in all three plots.
INCHES_PER_POD = 0.09
HEIGHT_IN = 4.5
LEFT_IN, RIGHT_IN, TOP_IN, BOTTOM_IN = 0.9, 0.3, 0.55, 0.7


def load_pod_expression(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, index_col=0)
    pod_cols = [c for c in df.columns if re.match(r"^POD\d+$", c, flags=re.IGNORECASE)]
    df = df[pod_cols]
    df.columns = [int(re.findall(r"\d+", c)[0]) for c in pod_cols]
    return df.sort_index(axis=1)


def plot_sample(sample: str, csv_path: Path, color: str, out_dir: Path,
                ylim: tuple[float, float], canvas_width: float) -> None:
    expr = load_pod_expression(csv_path)
    pod = expr.columns.to_numpy()
    pod_span = pod.max() - pod.min()

    # Every sample is rendered on the SAME total canvas size (canvas_width x
    # HEIGHT_IN, sized to fit the longest POD span) so the exported images
    # are pixel-identical in width and height — no aspect-ratio drift when
    # viewed at a common display width. Shorter samples simply leave blank
    # space on the right; the axes width itself still scales with pod_span
    # so 1 POD unit stays the same physical size across all three plots.
    fig, ax = plt.subplots(figsize=(canvas_width, HEIGHT_IN))
    axes_width_in = pod_span * INCHES_PER_POD
    ax.set_position([
        LEFT_IN / canvas_width,
        BOTTOM_IN / HEIGHT_IN,
        axes_width_in / canvas_width,
        (HEIGHT_IN - TOP_IN - BOTTOM_IN) / HEIGHT_IN,
    ])

    for gene, row in expr.iterrows():
        ax.plot(pod, row.values, color=color, alpha=ALPHA, lw=1.3,
                marker="o", markersize=3)

    ax.set_xlim(pod.min(), pod.max())
    # Shared y-axis range (fixed across short/inter/long, same figure height)
    # so logTMM-units-per-inch is identical too — a spike looks equally steep
    # in every plot, not just equally wide.
    ax.set_ylim(*ylim)
    ax.set_xlabel("POD", fontsize=11)
    ax.set_ylabel("Measured expression (logTMM)", fontsize=11)
    ax.set_title(f"{sample} — measured IFN gene expression (n={len(expr)} genes)",
                 fontsize=12)
    ax.set_xticks(pod)
    ax.grid(True, alpha=0.25)

    out_dir.mkdir(parents=True, exist_ok=True)
    png_path = out_dir / f"{sample}_ifn_measured_expression.png"
    pdf_path = out_dir / f"{sample}_ifn_measured_expression.pdf"
    # NOTE: no bbox_inches="tight" here — it would crop to content and undo
    # the fixed inches-per-POD-day / logTMM-per-inch scale set up above.
    fig.savefig(png_path, dpi=300)
    fig.savefig(pdf_path)
    plt.close(fig)

    print(f"[Saved] {png_path}  (canvas={canvas_width:.2f}in, axes={axes_width_in:.2f}in, POD span={pod_span})")
    print(f"[Saved] {pdf_path}")


def main():
    exprs = {s: load_pod_expression(cfg["csv"]) for s, cfg in SAMPLES.items()}
    y_min = min(e.values.min() for e in exprs.values())
    y_max = max(e.values.max() for e in exprs.values())
    pad = 0.05 * (y_max - y_min)
    ylim = (y_min - pad, y_max + pad)

    max_pod_span = max(e.columns.max() - e.columns.min() for e in exprs.values())
    canvas_width = LEFT_IN + max_pod_span * INCHES_PER_POD + RIGHT_IN

    for sample, cfg in SAMPLES.items():
        plot_sample(sample, cfg["csv"], cfg["color"], OUT_DIR, ylim, canvas_width)


if __name__ == "__main__":
    main()
