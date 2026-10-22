# replot.py
"""
Regenerate plots (and only plots) from an already-trained results_dde/<run>
directory, without repeating the expensive multistart/permutation pipeline.
Reads everything it needs (paths, K_fit, solver config) from best_checkpoint.pt.

Usage (run from DDE/, as a module so `modules.xxx` imports resolve):
  uv run python -m modules.replot results_dde/kfit2
  uv run python -m modules.replot results_dde/kfit6
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

from modules.data_io import load_expression_csv, intersect_genes, subset_sample_to_genes
from modules.dde_rhs import SharedParams
from modules.history import HistoryParam, HistoryGrid
from modules.dde_solver import solve_dde_at_observation_times, SolverConfig
from modules.plot_gene_trajectories import plot_all_genes


def main(out_dir: str):
    out_dir = Path(out_dir)
    device = torch.device("cpu")

    ckpt = torch.load(out_dir / "best_checkpoint.pt", map_location=device, weights_only=False)
    args = ckpt["args"]
    common_genes = ckpt["common_genes"]
    tau_by_lag = ckpt["tau_by_lag"]
    tau_max = ckpt["tau_max"]

    train_a = load_expression_csv(args["train_a_expr"], name=Path(args["train_a_expr"]).stem, device=device)
    train_b = load_expression_csv(args["train_b_expr"], name=Path(args["train_b_expr"]).stem, device=device)
    test_s  = load_expression_csv(args["test_expr"],   name=Path(args["test_expr"]).stem,   device=device)

    train_a = subset_sample_to_genes(train_a, common_genes)
    train_b = subset_sample_to_genes(train_b, common_genes)
    test_s  = subset_sample_to_genes(test_s, common_genes)

    # edge src/dst indices aren't stored in shared_state (they're constructor
    # args, not learnable params) -- rebuild them the same way main.py does.
    from modules.data_io import load_edge_csv, merge_edge_tables, filter_edges_to_genes, edges_to_index_by_lag
    edge_tables = [load_edge_csv(p) for p in args["edge_csvs"]]
    edges_merged = merge_edge_tables(edge_tables)
    edges_filt = filter_edges_to_genes(edges_merged, set(common_genes))
    edge_idx_by_lag = edges_to_index_by_lag(edges_filt, train_a.gene_to_idx)
    edge_idx_local = {k: {"src": d["src"].to(device), "dst": d["dst"].to(device)} for k, d in edge_idx_by_lag.items()}

    shared = SharedParams(G=len(common_genes), edge_idx_by_lag=edge_idx_local).to(device)
    shared.load_state_dict(ckpt["shared_state"])

    grid = HistoryGrid(M=args["M_hist"])
    hist_a = HistoryParam(G=len(common_genes), grid=grid, init_from_first_obs=train_a.X[0]).to(device)
    hist_b = HistoryParam(G=len(common_genes), grid=grid, init_from_first_obs=train_b.X[0]).to(device)
    hist_a.load_state_dict(ckpt["hist_train_state"][train_a.name])
    hist_b.load_state_dict(ckpt["hist_train_state"][train_b.name])
    hist_test = HistoryParam(G=len(common_genes), grid=grid, init_from_first_obs=test_s.X[0]).to(device)
    hist_test.load_state_dict(ckpt["hist_test_state"])

    solver_cfg = SolverConfig(
        dt_frac_of_median_gap=args["dt_frac"],
        dt_max=args["dt_max"],
        dt_min=args["dt_min"],
        activation=args["activation"],
        store_every_step=True,
    )
    solver_cfg.fixed_dt = ckpt["fixed_dt"]

    with torch.no_grad():
        X_train_a = solve_dde_at_observation_times(train_a.t, shared, hist_a, tau_by_lag, tau_max, solver_cfg)
        X_train_b = solve_dde_at_observation_times(train_b.t, shared, hist_b, tau_by_lag, tau_max, solver_cfg)
        X_test    = solve_dde_at_observation_times(test_s.t,  shared, hist_test, tau_by_lag, tau_max, solver_cfg)

    samples = {train_a.name: train_a, train_b.name: train_b, test_s.name: test_s}
    preds = {train_a.name: X_train_a, train_b.name: X_train_b, test_s.name: X_test}

    FUTURE_DAYS = 7.0

    def make_future(sample, history):
        dt_internal = solver_cfg.fixed_dt
        t_last = sample.t[-1].item()
        t_future = torch.arange(t_last + dt_internal, t_last + FUTURE_DAYS + 1e-8, dt_internal, device=device)
        t_extended = torch.cat([sample.t, t_future], dim=0)
        with torch.no_grad():
            X_extended = solve_dde_at_observation_times(t_extended, shared, history, tau_by_lag, tau_max, solver_cfg)
        X_future = X_extended[sample.t.shape[0]:]
        return t_future.detach().cpu().numpy(), X_future.detach().cpu().numpy()

    future_preds = {
        train_a.name: make_future(train_a, hist_a),
        train_b.name: make_future(train_b, hist_b),
        test_s.name:  make_future(test_s, hist_test),
    }

    plot_all_genes(
        samples=samples, preds=preds, gene_names=common_genes,
        out_dir=out_dir / "plots" / "with_legend", max_genes=None,
        K_fit=args["K_fit"], future=future_preds,
        test_sample_name=test_s.name, show_legend=True,
    )
    plot_all_genes(
        samples=samples, preds=preds, gene_names=common_genes,
        out_dir=out_dir / "plots" / "no_legend", max_genes=None,
        K_fit=args["K_fit"], future=future_preds,
        test_sample_name=test_s.name, show_legend=False,
    )
    print(f"[Done] Replotted -> {out_dir / 'plots'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("usage: uv run python -m modules.replot <results_dde/run_dir>")
        sys.exit(1)
    main(sys.argv[1])
