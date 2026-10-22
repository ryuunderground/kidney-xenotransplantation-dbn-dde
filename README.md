# kidney-xenotransplantation-dbn-dde

Pipeline for modeling interferon/antiviral gene expression dynamics in kidney
xenotransplantation, combining Dynamic Bayesian Network (DBN) structure
learning with a Delay Differential Equation (DDE) dynamics model.

## Pipeline order

```
preprocessing -> functional_analysis (incl. ClueGO) -> DBN-DDE_inputs -> DBN -> DDE
```

Each stage's outputs feed the next stage's inputs. Steps below are listed in
run order; scripts are R (preprocessing) or Python (everything else, managed
via `uv` / `pyproject.toml`).

---

## 1. `preprocessing/`

Normalizes raw counts and selects differentially expressed genes (DEGs) per
survival group, then filters those DEGs down to a candidate gene set for the
DBN-DDE pipeline.

- `TMM_norm.Rmd` — TMM normalization.
- `DEG_Selection.R` — NOISeq-sim DEG selection (prob(q) ≥ 0.95, |log2FC| ≥ 3).
- `DEG_candidates_matrix.R` — builds the DEG candidate expression matrix per
  survival group from the NOISeq gene-list outputs.
- `DEG_expr.R` — filters DBN-DDE candidate genes independently per timescale
  (short/inter/long) by variance and lag-1 autocorrelation cutoffs.
- `diagnotics.R` — k-means cluster-count diagnostics (elbow + silhouette).
- `csv_to_tsv.R` — CSV → TSV conversion utility (e.g. for MeV).

Outputs feed into `functional_analysis/`.

## 2. `functional_analysis/`

Clusters the DEG expression trajectories, runs functional enrichment on each
cluster, and uses that enrichment to select the final interferon/antiviral
gene set.

- `split_expr_by_cluster.py` — splits each timescale's DEG expression table
  into per-k-means-cluster expression CSVs.
- `k-means_plot.py` (usage: `k-means_plot_cmd`) — plots per-cluster
  expression trajectories.
- **ClueGO (manual step)** — the per-cluster gene lists are run through
  ClueGO (Cytoscape) by hand; results are saved under
  `functional_analysis/cluego_result/` (`cluego_table_{short_c2,inter_c1,long_c2}.xls`
  plus per-cluster figures). This is the functional-enrichment step that
  identifies which cluster, in each of the three survival groups, is
  interferon/antiviral-related.
- `../extract_go_filtered_genes.py` (repo root) — reads the three ClueGO
  `.xls` tables, keeps GO terms common to all three clusters and
  interferon/antiviral-related, unions the genes tied to those terms per
  sample, intersects with each sample's expression data, then intersects
  across short/inter/long. Writes the final gene set to `DBN-DDE_inputs/`.
- `plot_ifn_measured_expression.py` — plots measured (non-fitted) expression
  trajectories of the selected genes; output under
  `functional_analysis/ifn_measured_expression/`.

## 3. `DBN-DDE_inputs/`

Not a code stage — this is the shared input directory written by
`extract_go_filtered_genes.py` and read by both `DBN/` and `DDE/`.

## 4. `DBN/`

Learns a Dynamic Bayesian Network structure (gene regulatory edges with lag)
per survival group via simulated annealing.

- `main.py` — entry point (usage: `main_cmd`); runs multi-seed simulated
  annealing structure search and exports consensus edge lists at given
  confidence thresholds.
- `data_loader.py` — loads expression data into the internal `Dataset`
  format.
- `pearson.py`, `lasso_topk.py` — parent-candidate preprocessing (Pearson
  correlation screening, Lasso top-k selection).
- `graph.py` — `DBNGraph` structure representation and operators.
- `score.py` — BIC-LP scoring of candidate structures.
- `anealing.py` — simulated-annealing search over graph structures.
- `export.py` — exports the learned graph (edge lists, consensus edges) via
  `networkx`.

Outputs (`results/{short,inter,long}/consensus_edges_thr*.csv`,
`results/figure/`) feed into `DDE/` as the edge-prior input.

## 5. `DDE/`

Fits a shared, gene-level Delay Differential Equation model on the
DBN-learned edges (joint training on two samples, history-fit forecasting on
a third), then validates it.

- `main.py` — entry point (example invocations in `order.txt`); loads
  expression + edge CSVs, jointly trains the DDE on two samples, fits
  history and forecasts on a held-out test sample given `--K_fit` observed
  points, then runs time-permutation and edge-rewiring validation tests.
- `modules/data_io.py` — expression/edge CSV loading, gene intersection.
- `modules/time_delay.py` — delay (`tau`) configuration from observation
  spacing.
- `modules/dde_rhs.py`, `modules/dde_solver.py`, `modules/ode_rk.py` — DDE
  right-hand-side parameterization and RK4 solver.
- `modules/history.py`, `modules/interp.py` — history-function
  parameterization/interpolation for the pre-observation window.
- `modules/train_joint.py` — joint maximum-likelihood training on two
  samples.
- `modules/test_forecast.py` — history-fit + forecast on the test sample.
- `modules/multistart.py` — multi-seed training and best-seed selection.
- `modules/dde_validation.py` — time-permutation and edge-rewiring
  permutation tests.
- `modules/analyze_time_perm.py`, `modules/analyze_edge_perm.py` —
  summarize permutation-test results (p-values, effect sizes).
- `modules/plot_gene_trajectories.py`, `modules/export_dde_latex.py` —
  per-gene trajectory plots and LaTeX export of the fitted DDE equations.
- `modules/replot.py` — regenerate plots from a saved checkpoint.
- `modules/rerun_edge_rewiring.py` — re-run only the edge-rewiring
  validation test (e.g. with a different permutation count) from a saved
  checkpoint's recorded run config, without repeating training.
- `modules/linear_regression.py`, `modules/run_linear_regression.py` —
  linear / log-x / exponential regression baselines for comparison against
  the DDE model, fit on the same POD1-14 training window and evaluated on
  the same held-out forecast window.

Outputs under `results_dde/kfit{2,6}/` (checkpoints, predictions, metrics,
validation, plots) and `results_dde/linear_regression_pod14/` (baseline
regression fits/metrics/plots).
