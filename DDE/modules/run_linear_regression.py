"""
Run linear regression analysis on inter_input.csv and inter_missing_input.csv.
Results → ./results_dde/linear_regression/
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.linear_regression import (
    MissingDataForecaster,
    ExpMissingForecaster, LogXMissingForecaster,
)
import numpy as np

DATA_DIR = Path(__file__).parent.parent.parent / "DBN-DDE_inputs"
BASE_OUT = Path(__file__).parent.parent / "results_dde" / "linear_regression_pod14"

FULL_CSV = DATA_DIR / "inter_input.csv"
MISSING_CSV = DATA_DIR / "inter_missing_input.csv"


def run_total_genes(full_csv: Path = FULL_CSV, missing_csv: Path = MISSING_CSV,
                     base_out: Path = None):
    base_out = base_out if base_out is not None else BASE_OUT
    base = base_out
    print(f"=== Total genes — Linear / LogX / Exp fit (trained on missing, plot comparison with full) ===")
    print(f"    full={full_csv}  missing={missing_csv}")
    FULL_CSV, MISSING_CSV = full_csv, missing_csv

    def _make_dirs(*dirs):
        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

    # ── linear: y = at + b ──────────────────────────────────────────────────
    out_lin = base / "linear"
    dirs_lin = {
        "miss_legend":    out_lin / "fit_missing" / "with_legend",
        "miss_nolegen":   out_lin / "fit_missing" / "no_legend",
        "miss_leg_nof":   out_lin / "fit_missing" / "with_legend" / "without_formula",
        "miss_noleg_nof": out_lin / "fit_missing" / "no_legend"   / "without_formula",
    }
    _make_dirs(*dirs_lin.values())

    print("  [linear] fitting all genes...")
    fc_lin = MissingDataForecaster(MISSING_CSV, FULL_CSV).fit()
    fc_lin.evaluate_extrapolation()
    fc_lin.save_latex(out_lin / "linear_total_equations.tex")
    fc_lin.plot_fits(out_lin / "linear_total_grid.pdf", show_full_fit=False)
    fc_lin.save_metrics(out_lin / "linear_total_metrics.csv")
    fc_lin.plot_individual_pdfs(dirs_lin["miss_legend"],    show_legend=True,  fit_which="missing")
    fc_lin.plot_individual_pdfs(dirs_lin["miss_nolegen"],   show_legend=False, fit_which="missing")
    fc_lin.plot_individual_pdfs(dirs_lin["miss_leg_nof"],   show_legend=True,  fit_which="missing", show_annot=False)
    fc_lin.plot_individual_pdfs(dirs_lin["miss_noleg_nof"], show_legend=False, fit_which="missing", show_annot=False)

    # ── log-x: y = a·ln(t) + b ──────────────────────────────────────────────
    out_logx = base / "logx"
    dirs_logx = {
        "miss_legend":    out_logx / "fit_missing" / "with_legend",
        "miss_nolegen":   out_logx / "fit_missing" / "no_legend",
        "miss_leg_nof":   out_logx / "fit_missing" / "with_legend" / "without_formula",
        "miss_noleg_nof": out_logx / "fit_missing" / "no_legend"   / "without_formula",
    }
    _make_dirs(*dirs_logx.values())

    print("  [logX] fitting all genes...")
    fc_logx = LogXMissingForecaster(MISSING_CSV, FULL_CSV).fit()
    fc_logx.evaluate()
    fc_logx.save_latex(out_logx / "logx_total_equations.tex")
    fc_logx.plot_grid(out_logx / "logx_total_grid.pdf")
    fc_logx.save_metrics(out_logx / "logx_total_metrics.csv")
    fc_logx.plot_individual_pdfs(dirs_logx["miss_legend"],    show_legend=True,  fit_which="missing")
    fc_logx.plot_individual_pdfs(dirs_logx["miss_nolegen"],   show_legend=False, fit_which="missing")
    fc_logx.plot_individual_pdfs(dirs_logx["miss_leg_nof"],   show_legend=True,  fit_which="missing", show_annot=False)
    fc_logx.plot_individual_pdfs(dirs_logx["miss_noleg_nof"], show_legend=False, fit_which="missing", show_annot=False)

    # ── exp: y = a·e^(bt) + c ───────────────────────────────────────────────
    out_exp = base / "exp"
    dirs_exp = {
        "miss_legend":    out_exp / "fit_missing" / "with_legend",
        "miss_nolegen":   out_exp / "fit_missing" / "no_legend",
        "miss_leg_nof":   out_exp / "fit_missing" / "with_legend" / "without_formula",
        "miss_noleg_nof": out_exp / "fit_missing" / "no_legend"   / "without_formula",
    }
    _make_dirs(*dirs_exp.values())

    print("  [exp] fitting all genes...")
    fc_exp = ExpMissingForecaster(MISSING_CSV, FULL_CSV).fit()
    fc_exp.evaluate()
    fc_exp.save_latex(out_exp / "exp_total_equations.tex")
    fc_exp.plot_grid(out_exp / "exp_total_grid.pdf")
    fc_exp.save_metrics(out_exp / "exp_total_metrics.csv")
    fc_exp.plot_individual_pdfs(dirs_exp["miss_legend"],    show_legend=True,  fit_which="missing")
    fc_exp.plot_individual_pdfs(dirs_exp["miss_nolegen"],   show_legend=False, fit_which="missing")
    fc_exp.plot_individual_pdfs(dirs_exp["miss_leg_nof"],   show_annot=False,  show_legend=True,  fit_which="missing")
    fc_exp.plot_individual_pdfs(dirs_exp["miss_noleg_nof"], show_annot=False,  show_legend=False, fit_which="missing")
    print()


if __name__ == "__main__":
    run_total_genes()
    print(f"Done. Results in: {BASE_OUT}")
