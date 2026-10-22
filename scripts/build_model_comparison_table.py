"""Build the corrected 30-split Table 1 and paired model-comparison evidence."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
MODELS = ROOT / "results" / "models"
METRICS_PATH = MODELS / "model_comparison_metrics.csv"
SUMMARY_PATH = MODELS / "model_comparison_summary.csv"
PAIRED_PATH = TABLES / "model_comparison_paired_differences.csv"
TABLE1_PATH = TABLES / "table1_model_comparison.csv"

FULL_MODELS = ("Baseline", "PCA", "ADME", "Plain")
TABLE1_METRICS = ("RMSE", "MAE", "R2", "Spearman", "CCC", "CI_0.90_size", "CI_0.90_coverage")
PAIRED_METRICS = ("RMSE", "MAE", "R2", "Spearman", "CI_0.90_size", "CI_0.90_coverage")
COMPARISONS = (
    ("ADME", "Plain", "full_test"),
    ("PCA", "Plain", "full_test"),
    ("Baseline", "Plain", "full_test"),
    ("BBB pass", "Plain", "bbb_pass_test"),
)
BOOTSTRAP_SEED = 20261017
N_BOOTSTRAP = 10_000


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def select_summary(summary: pd.DataFrame, model: str, domain: str) -> pd.Series:
    rows = summary.loc[summary["model"].eq(model) & summary["evaluation_domain"].eq(domain)]
    if len(rows) != 1:
        raise ValueError(f"Expected one summary row for {model}/{domain}; got {len(rows)}")
    return rows.iloc[0]


def mean_sd(row: pd.Series, metric: str, *, percent: bool = False) -> str:
    scale = 100.0 if percent else 1.0
    suffix = r"\%" if percent else ""
    return f"${row[f'{metric}_mean'] * scale:.3f}\\,\\pm\\,{row[f'{metric}_std'] * scale:.3f}{suffix}$"


def bootstrap_mean_interval(differences: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    positions = rng.integers(0, len(differences), size=(N_BOOTSTRAP, len(differences)))
    bootstrap_means = differences[positions].mean(axis=1)
    return tuple(float(value) for value in np.quantile(bootstrap_means, (0.025, 0.975)))


def paired_differences(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    source_hash = sha256(METRICS_PATH)
    for comparison_index, (model, reference, domain) in enumerate(COMPARISONS):
        model_rows = metrics.loc[
            metrics["model"].eq(model) & metrics["evaluation_domain"].eq(domain)
        ].set_index("split_index").sort_index()
        reference_rows = metrics.loc[
            metrics["model"].eq(reference) & metrics["evaluation_domain"].eq(domain)
        ].set_index("split_index").sort_index()
        if not model_rows.index.equals(reference_rows.index) or len(model_rows) != 30:
            raise ValueError(f"Incomplete paired comparison for {model} versus {reference} in {domain}")
        for metric_index, metric in enumerate(PAIRED_METRICS):
            differences = (model_rows[metric] - reference_rows[metric]).to_numpy(float)
            rng = np.random.default_rng(BOOTSTRAP_SEED + comparison_index * 100 + metric_index)
            ci_lower, ci_upper = bootstrap_mean_interval(differences, rng)
            rows.append(
                {
                    "evaluation_domain": domain,
                    "model": model,
                    "reference_model": reference,
                    "metric": metric,
                    "difference_definition": "model - reference_model",
                    "n_splits": int(len(differences)),
                    "mean_difference": float(differences.mean()),
                    "sd_of_split_differences": float(differences.std(ddof=1)),
                    "bootstrap_ci_95_lower": ci_lower,
                    "bootstrap_ci_95_upper": ci_upper,
                    "bootstrap_method": "paired percentile bootstrap of the mean split-wise difference",
                    "bootstrap_replicates": N_BOOTSTRAP,
                    "bootstrap_seed": BOOTSTRAP_SEED + comparison_index * 100 + metric_index,
                    "source_metrics_sha256": source_hash,
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    summary = pd.read_csv(SUMMARY_PATH)
    metrics = pd.read_csv(METRICS_PATH)
    if len(metrics) != 180 or set(summary["n_splits"]) != {30}:
        raise ValueError("Expected completed corrected 30-split model artifacts")
    paired = paired_differences(metrics)
    PAIRED_PATH.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(TABLE1_PATH, index=False)
    paired.to_csv(PAIRED_PATH, index=False)
    print(f"Wrote: {TABLE1_PATH.relative_to(ROOT)}")
    print(f"Wrote: {PAIRED_PATH.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
