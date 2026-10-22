"""
Re-run ONLY the edge-rewiring permutation test for an existing results_dde/kfitN
run, using the exact data/config recorded in that run's best_checkpoint.pt
(args = vars(argparse.Namespace) at original train time), but overriding
n_perm_edge.

Skips the expensive multistart training + time-permutation test entirely --
those are untouched. Deletes the old edge_rewire_* files in validation/ before
writing the new ones.

Usage (run from DDE/, since checkpoint args store paths relative to DDE/):
  uv run modules/rerun_edge_rewiring.py --out_dir results_dde/kfit2 --n_perm_edge 100
  uv run modules/rerun_edge_rewiring.py --out_dir results_dde/kfit6 --n_perm_edge 100
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch

from modules.data_io import (
    load_expression_csv, intersect_genes, subset_sample_to_genes,
    load_edge_csv, merge_edge_tables, filter_edges_to_genes,
    edges_to_index_by_lag,
)
from modules.time_delay import choose_delta_from_samples, DelayConfig
from modules.dde_solver import SolverConfig, choose_internal_dt
from modules.train_joint import TrainConfig, train_joint_two_samples
from modules.test_forecast import TestConfig, fit_history_and_forecast
from modules.dde_validation import edge_rewiring_permutation_test
from modules.analyze_edge_perm import summarize_edge_permutation


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", type=str, required=True, help="e.g. results_dde/kfit2")
    p.add_argument("--n_perm_edge", type=int, required=True)
    cli = p.parse_args()

    out_dir = Path(cli.out_dir)
    ckpt = torch.load(out_dir / "best_checkpoint.pt", map_location="cpu", weights_only=False)
    args = argparse.Namespace(**ckpt["args"])

    print(f"[Info] Reloaded original run args from {out_dir / 'best_checkpoint.pt'} "
          f"(K_fit={args.K_fit}, original n_perm_edge={args.n_perm_edge})")
    print(f"[Info] Overriding n_perm_edge -> {cli.n_perm_edge}")

    device = torch.device(args.device) if args.device != "auto" else torch.device("cpu")

    # ---- STEP A: Load expression (same as main.py) ----
    train_a = load_expression_csv(args.train_a_expr, name=Path(args.train_a_expr).stem, device=device)
    train_b = load_expression_csv(args.train_b_expr, name=Path(args.train_b_expr).stem, device=device)
    test_s  = load_expression_csv(args.test_expr,   name=Path(args.test_expr).stem,   device=device)

    # ---- STEP B: Gene intersection ----
    common_genes = intersect_genes([train_a, train_b, test_s])
    train_a = subset_sample_to_genes(train_a, common_genes)
    train_b = subset_sample_to_genes(train_b, common_genes)
    test_s  = subset_sample_to_genes(test_s,  common_genes)

    # ---- STEP C: Edges ----
    edge_tables = [load_edge_csv(p) for p in args.edge_csvs]
    edges_merged = merge_edge_tables(edge_tables)
    edges_filt = filter_edges_to_genes(edges_merged, set(common_genes))
    edge_idx_by_lag = edges_to_index_by_lag(edges_filt, train_a.gene_to_idx)
    lags = sorted(edge_idx_by_lag.keys())
    edge_idx_local = {k: {"src": d["src"].to(device), "dst": d["dst"].to(device)}
                       for k, d in edge_idx_by_lag.items()}

    # ---- STEP D: Delta / tau ----
    delta = choose_delta_from_samples([train_a.t, train_b.t], mode=args.delta_mode)
    delay_cfg = DelayConfig(delta=delta, lags=lags)
    tau_by_lag = delay_cfg.tau_by_lag()
    tau_max = delay_cfg.tau_max()

    # ---- STEP E: Configs ----
    solver_cfg = SolverConfig(
        dt_frac_of_median_gap=args.dt_frac,
        dt_max=args.dt_max,
        dt_min=args.dt_min,
        activation=args.activation,
        store_every_step=True,
    )
    solver_cfg.fixed_dt = choose_internal_dt(train_a.t, solver_cfg)

    train_cfg = TrainConfig(
        lr_shared=args.lr_shared, lr_hist=args.lr_hist, epochs=args.epochs,
        sigma=args.sigma, M_hist=args.M_hist, solver_cfg=solver_cfg,
        grad_clip=args.grad_clip, patience=args.patience,
    )
    test_cfg = TestConfig(
        K_fit=args.K_fit, iters_hist=args.iters_hist, lr_hist=args.lr_hist_test,
        sigma=args.sigma, M_hist=args.M_hist, solver_cfg=solver_cfg,
        grad_clip=args.grad_clip,
    )

    validation_dir = out_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    # ---- delete old edge_rewire_* results ----
    removed = 0
    for f in validation_dir.glob("edge_rewire_*"):
        f.unlink()
        removed += 1
    print(f"[Info] Removed {removed} old edge_rewire_* files from {validation_dir}")

    # ---- re-run edge rewiring per seed ----
    all_edge = []
    for s in args.seeds:
        print(f"\n[Validation] Edge rewiring (seed={s}, n_perm={cli.n_perm_edge})")
        res = edge_rewiring_permutation_test(
            train_joint_fn=train_joint_two_samples,
            forecast_fn=fit_history_and_forecast,
            train_a=train_a,
            train_b=train_b,
            test_sample=test_s,
            edge_idx_by_lag=edge_idx_local,
            tau_by_lag=tau_by_lag,
            tau_max=tau_max,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            n_perm=cli.n_perm_edge,
            seed=s,
            out_prefix=str(validation_dir / f"edge_rewire_seed{s}"),
            plot=True,
            perm_workers=args.perm_workers,
            device_type=device.type,
        )
        (validation_dir / f"edge_rewire_seed{s}.json").write_text(
            json.dumps(res, indent=2), encoding="utf-8"
        )
        print("[Validation] p(R2):", res["p_value_r2"], " p(RMSE):", res["p_value_rmse"])
        all_edge.append(res)

    (validation_dir / "edge_rewire_permutation_all.json").write_text(
        json.dumps(all_edge, indent=2), encoding="utf-8"
    )

    summarize_edge_permutation(validation_dir)
    print(f"\n[Done] Edge rewiring re-run (n_perm={cli.n_perm_edge}) saved in: {validation_dir}")


if __name__ == "__main__":
    main()
