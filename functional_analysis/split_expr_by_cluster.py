"""Split each timepoint's DEG_expr.csv into per-cluster expr csv files."""

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
EXPR_DIR = BASE_DIR / "DEG_expr"

# number of clusters per timepoint
N_CLUSTERS = {"short": 3, "inter": 4, "long": 3}


def split_group(group: str, n_clusters: int) -> None:
    expr_path = EXPR_DIR / f"{group}_expr.csv"
    expr_df = pd.read_csv(expr_path)

    group_dir = BASE_DIR / group
    out_dir = group_dir / "cluster_lists"
    out_dir.mkdir(exist_ok=True)

    for c in range(1, n_clusters + 1):
        cluster_list_path = group_dir / f"{group}_gene_list_cluster-{c}.csv"
        cluster_genes = pd.read_csv(cluster_list_path)["GeneName"]

        cluster_expr = expr_df[expr_df["GeneName"].isin(cluster_genes)]

        out_path = out_dir / f"{group}_cluster{c}_expr.csv"
        cluster_expr.to_csv(out_path, index=False)
        print(f"{out_path.name}: {len(cluster_expr)} genes")


def main() -> None:
    for group, n_clusters in N_CLUSTERS.items():
        split_group(group, n_clusters)


if __name__ == "__main__":
    main()
