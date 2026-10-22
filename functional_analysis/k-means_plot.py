import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def collect_cluster_data(cluster_dir: str):
    """
    Collect per-cluster parsed DataFrame objects from a directory.
    - Skip files that do not contain POD columns (e.g. gene-only lists)
    - If same cluster id appears multiple times, prefer text/tsv over csv
    """
    cluster_dir = Path(cluster_dir)
    files = sorted([f for f in cluster_dir.iterdir() if f.is_file() and "cluster" in f.stem.lower()])
    if not files:
        raise ValueError(f"No cluster files found in {cluster_dir}")

    # Lower score = higher priority
    ext_priority = {".txt": 0, ".tsv": 1, ".csv": 2}
    picked = {}

    for fp in files:
        try:
            cluster_id = extract_cluster_id(str(fp))
        except ValueError:
            continue

        try:
            df = read_cluster_file(str(fp))
        except Exception:
            # Not a trajectory table; likely a gene-only list.
            continue

        prev = picked.get(cluster_id)
        curr_rank = ext_priority.get(fp.suffix.lower(), 9)
        if prev is None or curr_rank < prev["rank"]:
            picked[cluster_id] = {"file": fp, "df": df, "rank": curr_rank}

    if not picked:
        raise ValueError(
            f"No valid trajectory files with POD columns found in {cluster_dir}. "
            "Check that files contain GeneName and POD* columns."
        )

    return [(cid, picked[cid]["file"], picked[cid]["df"]) for cid in sorted(picked.keys())]


def read_cluster_file(file_path: str) -> pd.DataFrame:
    """
    Read a cluster file containing:
    Original row, GeneName, POD1, POD3, ...
    Handles:
      1) normal CSV/TSV
      2) malformed one-column CSV where the whole row is stored in a single column
    """
    import csv
    import pandas as pd
    import re

    def finalize_df(df: pd.DataFrame, file_path: str) -> pd.DataFrame:
        df = df.dropna(how="all")
        df.columns = [str(c).strip() for c in df.columns]

        # Gene column 찾기
        gene_col = None
        for c in df.columns:
            if c.lower() == "genename":
                gene_col = c
                break
        if gene_col is None:
            raise ValueError(f"'GeneName' column not found in {file_path}. Found: {list(df.columns)}")

        # POD 컬럼 찾기
        time_cols = [c for c in df.columns if re.match(r"^POD\d+$", str(c), flags=re.IGNORECASE)]
        if not time_cols:
            raise ValueError(f"No POD columns found in {file_path}. Found: {list(df.columns)}")

        time_cols = sorted(time_cols, key=lambda x: int(re.findall(r"\d+", x)[0]))

        out = df[[gene_col] + time_cols].copy()
        out = out.rename(columns={gene_col: "GeneName"})

        out["GeneName"] = out["GeneName"].astype(str).str.strip()
        out = out[out["GeneName"].notna()]
        out = out[out["GeneName"] != ""]
        out = out[out["GeneName"].str.lower() != "nan"]

        for c in time_cols:
            out[c] = pd.to_numeric(out[c], errors="coerce")

        out = out.dropna(subset=time_cols, how="all")
        return out

    # 1) 일반적인 방식들 먼저 시도
    tried = []
    for sep, name in [(",", "comma"), ("\t", "tab"), (r"\s+", "whitespace")]:
        try:
            if name == "whitespace":
                df = pd.read_csv(file_path, sep=sep, engine="python")
            else:
                df = pd.read_csv(file_path, sep=sep)

            tried.append((name, list(df.columns)))

            # 정상 파싱되면 바로 반환
            if len(df.columns) > 1:
                return finalize_df(df, file_path)

            # 한 컬럼만 읽힌 경우 fallback으로 넘김
        except Exception:
            pass

    # 2) fallback: 파일을 직접 읽어서 csv sniff + 한 컬럼 문자열 분해
    with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
        lines = [line.rstrip("\n\r") for line in f if line.strip()]

    if not lines:
        raise ValueError(f"Empty file: {file_path}")

    # 첫 줄 헤더 추정
    header_line = lines[0]

    # 구분자 추정
    if header_line.count(",") >= 2:
        delim = ","
    elif header_line.count("\t") >= 2:
        delim = "\t"
    else:
        # 마지막 fallback: 연속 공백
        delim = None

    rows = []
    if delim is not None:
        for line in lines:
            rows.append([x.strip().strip('"') for x in line.split(delim)])
    else:
        for line in lines:
            rows.append(re.split(r"\s+", line.strip()))

    # 행 길이 맞추기
    max_len = max(len(r) for r in rows)
    rows = [r + [""] * (max_len - len(r)) for r in rows]

    df = pd.DataFrame(rows[1:], columns=[c.strip().strip('"') for c in rows[0]])

    # 만약 여전히 한 컬럼이면, 그 컬럼 안을 다시 comma로 분해
    if len(df.columns) == 1:
        only_col = df.columns[0]
        split_rows = df[only_col].astype(str).str.split(",", expand=True)
        header_split = only_col.split(",")
        if len(header_split) == split_rows.shape[1]:
            split_rows.columns = [h.strip().strip('"') for h in header_split]
            df = split_rows
        else:
            raise ValueError(
                f"Failed to parse {file_path}. "
                f"Tried delimiters: {tried}. "
                f"Still got a single-column table with header: {list(df.columns)}"
            )

    return finalize_df(df, file_path)


def extract_cluster_id(file_path: str) -> int:
    """
    Extract cluster number from filename like:
    short_gene_list_cluster-1.txt
    """
    m = re.search(r"cluster[-_]?(\d+)", Path(file_path).stem, flags=re.IGNORECASE)
    if not m:
        raise ValueError(f"Cannot extract cluster number from: {file_path}")
    return int(m.group(1))


def plot_sample_clusters(cluster_dir: str, sample_name: str, outdir: str):
    """
    Draw one figure per sample:
    - gray thin lines = each gene trajectory
    - magenta thick line = centroid
    - y-axis = logTMM
    """
    cluster_data = collect_cluster_data(cluster_dir)
    n_clusters = len(cluster_data)

    ncols = min(3, n_clusters)
    nrows = int(np.ceil(n_clusters / ncols))

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(5 * ncols, 3.8 * nrows), squeeze=False)

    for ax in axes.flat:
        ax.set_visible(False)

    for idx, (cluster_id, fp, df) in enumerate(cluster_data):
        ax = axes.flat[idx]
        ax.set_visible(True)

        time_cols = [c for c in df.columns if c != "GeneName"]
        x = np.arange(len(time_cols))

        expr = df[time_cols].copy()

        # Plot individual genes
        for _, row in expr.iterrows():
            ax.plot(x, row.values, color="gray", alpha=0.18, lw=0.8)

        # Centroid = mean across genes at each time point
        centroid = expr.mean(axis=0)
        ax.plot(x, centroid.values, color="magenta", lw=2.8)

        ax.set_title(f"Cluster {cluster_id} (n={len(df)})")
        ax.set_xticks(x)
        ax.set_xticklabels(time_cols, rotation=45, ha="right")
        ax.set_ylabel("logTMM")
        ax.grid(True, alpha=0.25)

    fig.suptitle(f"{sample_name} trajectory clusters", fontsize=15, y=1.02)
    fig.tight_layout()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    png_path = outdir / f"{sample_name}_clusters_logTMM.png"
    pdf_path = outdir / f"{sample_name}_clusters_logTMM.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {png_path}")
    print(f"[Saved] {pdf_path}")


def plot_combined_summary(short_dir: str, inter_dir: str, long_dir: str, outdir: str):
    """
    Combined figure:
    rows = short / inter / long
    columns = cluster index
    each panel:
      gray lines = genes
      magenta = centroid
    """
    sample_dirs = {
        "short": short_dir,
        "inter": inter_dir,
        "long": long_dir,
    }

    sample_clusters = {}
    max_clusters = 0

    for sample, d in sample_dirs.items():
        clusters = collect_cluster_data(d)
        sample_clusters[sample] = clusters
        max_clusters = max(max_clusters, len(clusters))

    fig, axes = plt.subplots(nrows=3, ncols=max_clusters, figsize=(4.5 * max_clusters, 10), squeeze=False)

    for ax in axes.flat:
        ax.set_visible(False)

    row_order = ["short", "inter", "long"]

    for i, sample in enumerate(row_order):
        for j, (cluster_id, fp, df) in enumerate(sample_clusters[sample]):
            ax = axes[i, j]
            ax.set_visible(True)

            time_cols = [c for c in df.columns if c != "GeneName"]
            x = np.arange(len(time_cols))
            expr = df[time_cols].copy()

            for _, row in expr.iterrows():
                ax.plot(x, row.values, color="gray", alpha=0.15, lw=0.7)

            centroid = expr.mean(axis=0)
            ax.plot(x, centroid.values, color="magenta", lw=2.5)

            ax.set_title(f"{sample} / C{cluster_id} (n={len(df)})")
            ax.set_xticks(x)
            ax.set_xticklabels(time_cols, rotation=45, ha="right")
            ax.set_ylabel("logTMM")
            ax.grid(True, alpha=0.25)

    fig.suptitle("Trajectory cluster summary (logTMM)", fontsize=16, y=1.01)
    fig.tight_layout()

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    png_path = outdir / "combined_cluster_summary_logTMM.png"
    pdf_path = outdir / "combined_cluster_summary_logTMM.pdf"

    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    plt.close(fig)

    print(f"[Saved] {png_path}")
    print(f"[Saved] {pdf_path}")


def main():
    parser = argparse.ArgumentParser(description="Replot cluster trajectories from per-cluster gene expression files.")
    parser.add_argument("--short_dir", required=True, help="e.g. short/cluster_lists")
    parser.add_argument("--inter_dir", required=True, help="e.g. inter/cluster_lists")
    parser.add_argument("--long_dir", required=True, help="e.g. long/cluster_lists")
    parser.add_argument("--outdir", default="cluster_plots_logTMM")

    args = parser.parse_args()

    plot_sample_clusters(args.short_dir, "short", args.outdir)
    plot_sample_clusters(args.inter_dir, "inter", args.outdir)
    plot_sample_clusters(args.long_dir, "long", args.outdir)

    plot_combined_summary(args.short_dir, args.inter_dir, args.long_dir, args.outdir)


if __name__ == "__main__":
    main()