"""Write the 30-split Table 1 fragment from the saved comparison summary."""

from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "results" / "tables" / "model_comparison_summary.csv"
OUT = ROOT / "manuscript" / "tables" / "table1_model_quality.tex"
METRICS = ["RMSE", "MAE", "MSE", "R2", "Spearman", "CCC", "CI_0.80_size", "CI_0.90_size", "CI_0.95_size"]


def select(summary: pd.DataFrame, model: str, domain: str) -> pd.Series:
    rows = summary.loc[summary["model"].eq(model) & summary["evaluation_domain"].eq(domain)]
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row for {model}/{domain}; got {len(rows)}")
    return rows.iloc[0]


def mean_sd(row: pd.Series, metric: str) -> str:
    return f"${row[f'{metric}_mean']:.3f}\\,\\pm\\,{row[f'{metric}_std']:.3f}$"


def render() -> Path:
    summary = pd.read_csv(SUMMARY)
    full_models = ["Baseline", "PCA", "ADME", "Plain"]
    full_rows = [select(summary, model, "full_test") for model in full_models]
    bbb_plain = select(summary, "Plain", "bbb_pass_test")
    bbb_pass = select(summary, "BBB pass", "bbb_pass_test")
    n_full = f"{full_rows[0]['n_test_mean']:.0f} ({int(full_rows[0]['n_test_min']):,}--{int(full_rows[0]['n_test_max']):,})"
    n_bbb = f"{bbb_plain['n_test_mean']:.0f} ({int(bbb_plain['n_test_min']):,}--{int(bbb_plain['n_test_max']):,})"
    lines = [
        r"\begin{sidewaystable}[!p]",
        r"\caption{Thirty-split comparison of regression models. Each value is the mean $\pm$ sample standard deviation across 30 predefined cluster-safe splits. Lower RMSE, MAE, MSE, and conformal prediction-interval (PI) length are better; higher $R^2$, Spearman's $\rho$, and CCC are better. Split-conformal residuals were calibrated on the validation subset and evaluated once on the held-out test subset.\label{tab:model_quality}}",
        r"\centering",
        r"\tiny",
        r"\setlength{\tabcolsep}{1.7pt}",
        r"\renewcommand{\arraystretch}{1.14}",
        r"\begin{tabular}{l *{6}{r} *{3}{r}}",
        r"\toprule",
        r"\textbf{Model} & \textbf{RMSE} & \textbf{MAE} & \textbf{MSE} & $\boldsymbol{R^2}$ & \textbf{Spearman} & \textbf{CCC} & \multicolumn{3}{c}{\textbf{Conformal PI length}} \\",
        r"\cmidrule(lr){8-10}",
        r"& & & & & & & \textbf{80\%} & \textbf{90\%} & \textbf{95\%} \\",
        r"\midrule",
        rf"\multicolumn{{10}}{{l}}{{\textit{{Full held-out test set: mean $n={n_full}$ molecules per split}}}} \\",
    ]
    for model, row in zip(full_models, full_rows):
        label = model.replace(" ", r"\ ")
        lines.append(label + " & " + " & ".join(mean_sd(row, metric) for metric in METRICS) + r" \\")
    lines.extend([
        r"\midrule",
        rf"\multicolumn{{10}}{{l}}{{\textit{{BBB-permeant held-out test subset: same molecules for both rows; mean $n={n_bbb}$ per split}}}} \\",
        "Plain (full train) & " + " & ".join(mean_sd(bbb_plain, metric) for metric in METRICS) + r" \\",
        "BBB pass & " + " & ".join(mean_sd(bbb_pass, metric) for metric in METRICS) + r" \\",
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{sidewaystable}",
    ])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n")
    return OUT


if __name__ == "__main__":
    print(f"Wrote: {render()}")
