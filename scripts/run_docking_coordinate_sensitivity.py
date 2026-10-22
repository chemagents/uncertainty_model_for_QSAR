#!/usr/bin/env python3
"""Recompute the current 30-split docking-coordinate sensitivity analysis.

This program uses only the saved Plain-model predictions and calibration
residuals.  It does not fit or tune a point-prediction model.  For each
registered split, it compares the current raw mean docking-score coordinate
with a target-wise standardized mean and the first principal component (PC1)
of the standardized 44-score panel.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data/df_final.csv"
SPLIT_PATH = ROOT / "data/splits/split_indices.csv"
MODEL_ROOT = ROOT / "results/models"
TABLE_PATH = ROOT / "results/tables/tableS6_docking_coordinate.csv"

PROTEINS = [
    "1m2z", "1pbq", "1xoq", "2rh1", "2vt4", "2ydo", "2z5x", "3b66", "3kk6", "3ln1",
    "3rze", "4djh", "4ey7", "4iar", "4mqs", "4n6h", "5cxv", "5i71", "5tvn", "5u09",
    "5va1", "6cm4", "6kpf", "6kux", "6lqa", "6pdj", "6x3x", "6y1z", "7f8y", "7kwe",
    "7ljd", "7wc9", "7xnk", "7ym8", "8e9y", "8ef6", "8fhs", "8pjk", "8st0", "8wty",
    "8xvk", "8yn3", "9eo4", "V1A",
]
BTOX_COLUMNS = ["hERG", "Respiratory", "Neurotoxicity-DI"]
ALPHA = 0.10
LOCAL_BANDWIDTH = 0.90
NEFF_FALLBACK = 80


def weighted_quantiles_and_neff(
    query_features: np.ndarray,
    calibration_features: np.ndarray,
    calibration_residuals: np.ndarray,
    quantile_level: float,
    bandwidth: float,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Current local-conformal weighted quantile rule from notebook 07."""
    order = np.argsort(calibration_residuals)
    sorted_residuals = calibration_residuals[order]
    quantiles = np.empty(len(query_features), dtype=float)
    neff = np.empty(len(query_features), dtype=float)
    for start in range(0, len(query_features), batch_size):
        end = min(start + batch_size, len(query_features))
        query = query_features[start:end]
        difference = query[:, None, :] - calibration_features[None, :, :]
        weights = np.exp(-0.5 * np.sum(difference * difference, axis=2) / (bandwidth**2))
        weight_sum = weights.sum(axis=1)
        neff[start:end] = weight_sum**2 / np.sum(weights * weights, axis=1)
        cumulative = np.cumsum(weights[:, order], axis=1) / weight_sum[:, None]
        indices = np.argmax(cumulative >= quantile_level, axis=1)
        quantiles[start:end] = sorted_residuals[indices]
    return quantiles, neff


def standardize_from_calibration(
    calibration_raw: np.ndarray,
    query_raw: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    location = np.nanmean(calibration_raw, axis=0)
    scale = np.nanstd(calibration_raw, axis=0)
    scale[~np.isfinite(scale) | (scale == 0)] = 1.0
    return (calibration_raw - location) / scale, (query_raw - location) / scale


def pca_first_component(
    calibration_standardized: np.ndarray,
    query_standardized: np.ndarray,
    raw_calibration_emean: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit PC1 on calibration molecules and orient it towards raw Emean."""
    _, singular_values, right_vectors = np.linalg.svd(calibration_standardized, full_matrices=False)
    loading = right_vectors[0]
    calibration_pc1 = calibration_standardized @ loading
    query_pc1 = query_standardized @ loading
    if spearmanr(raw_calibration_emean, calibration_pc1).statistic < 0:
        calibration_pc1 *= -1
        query_pc1 *= -1
    explained_variance = float(singular_values[0] ** 2 / np.sum(singular_values**2))
    return calibration_pc1, query_pc1, explained_variance


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "rank_agreement_with_raw_emean_test",
        "pc1_explained_variance_calibration",
        "empirical_coverage_90",
        "mean_pi_width_90",
        "fallback_fraction",
    ]
    records: list[dict[str, object]] = []
    for coordinate, group in frame.groupby("coordinate", sort=False):
        row: dict[str, object] = {"coordinate": coordinate, "n_splits": len(group)}
        for metric in metrics:
            values = group[metric].dropna()
            row[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            row[f"{metric}_sd"] = float(values.std(ddof=1)) if len(values) > 1 else np.nan
            row[f"{metric}_min"] = float(values.min()) if len(values) else np.nan
            row[f"{metric}_max"] = float(values.max()) if len(values) else np.nan
        records.append(row)
    return pd.DataFrame(records)


def mean_pm(row: pd.Series, metric: str, decimals: int) -> str:
    return f"{row[f'{metric}_mean']:.{decimals}f} $\\pm$ {row[f'{metric}_sd']:.{decimals}f}"


def fallback_note(summary: pd.DataFrame, tracked: bool = False) -> str:
    labels = {
        "raw_emean": "raw mean",
        "targetwise_standardized_mean": "standardized mean",
        "targetwise_standardized_pc1": "PC1",
    }
    parts = []
    for coordinate, label in labels.items():
        row = summary.loc[summary.coordinate.eq(coordinate)].iloc[0]
        parts.append(f"{label}: {100 * row.fallback_fraction_mean:.2f} $\\pm$ {100 * row.fallback_fraction_sd:.2f}\\%")
    text = (
        r"The fixed global fallback ($n_{\mathrm{eff}}<80$) was used in "
        + "; ".join(parts)
        + " of held-out molecules."
    )
    if not tracked:
        return text
    return (
        r"\trackednew{Global fallback ($n_{\mathrm{eff}}<80$): " + parts[0] + ".}"
        + r"\par\trackednew{" + parts[1] + "; " + parts[2] + " of held-out molecules.}"
    )


HISTORICAL_S6 = {
    "raw_emean": ("1.000", r"0.892 $\pm$ 0.010", r"1.535 $\pm$ 0.051"),
    "targetwise_standardized_mean": (r"0.9993 $\pm$ 0.0001", r"0.892 $\pm$ 0.010", r"1.538 $\pm$ 0.051"),
    "targetwise_standardized_pc1": (r"0.9970 $\pm$ 0.0002", r"0.892 $\pm$ 0.010", r"1.537 $\pm$ 0.051"),
}


def tracked_cell(old: str, new: str) -> str:
    if old == new:
        return new
    return r"\shortstack[c]{\sout{" + old + r"}\\\trackednew{" + new + "}}"


def render_table(summary: pd.DataFrame, tracked: bool = False) -> str:
    labels = {
        "raw_emean": r"Raw $E_{\mathrm{mean}}$",
        "targetwise_standardized_mean": "Target-wise standardized mean",
        "targetwise_standardized_pc1": "Standardized PC1",
    }
    rows = []
    for coordinate in labels:
        row = summary.loc[summary.coordinate.eq(coordinate)].iloc[0]
        rank = "1.000" if coordinate == "raw_emean" else mean_pm(row, "rank_agreement_with_raw_emean_test", 4)
        coverage = mean_pm(row, "empirical_coverage_90", 3)
        width = mean_pm(row, "mean_pi_width_90", 3)
        if tracked:
            old_rank, old_coverage, old_width = HISTORICAL_S6[coordinate]
            rank = tracked_cell(old_rank, rank)
            coverage = tracked_cell(old_coverage, coverage)
            width = tracked_cell(old_width, width)
        rows.append(" & ".join((labels[coordinate], rank, coverage, width)) + " " + chr(92) * 2)
    pc1 = summary.loc[summary.coordinate.eq("targetwise_standardized_pc1")].iloc[0]
    return "\n".join([
        "% Generated by scripts/run_docking_coordinate_sensitivity.py; do not edit numbers by hand.",
        r"\begin{tabular}{lccc}", r"\toprule",
        r"Docking coordinate & Rank agreement with raw $E_{\mathrm{mean}}$ & 90\% empirical coverage & Mean PI width \\",
        r"\midrule", *rows, r"\bottomrule", r"\end{tabular}",
        r"\par\vspace{0.5em}", r"\begin{minipage}{0.94\textwidth}\footnotesize",
        (
            "PC1 explained " + mean_pm(pc1, "pc1_explained_variance_calibration", 3)
            + " of the calibration-panel variance across the 30 splits. "
            + "Each target-wise scaling and PC1 fit used calibration molecules only; rank agreement was evaluated on held-out test molecules. "
            + fallback_note(summary)
            if not tracked else
            "The first principal component explained "
            + r"\sout{$0.733\pm0.004$} \trackednew{$"
            + mean_pm(pc1, "pc1_explained_variance_calibration", 3).replace(" $\\pm$ ", r"\pm")
            + r"$} of the variance in the target-wise standardized 44-score panel. "
            + r"\par\trackednew{Target-wise scaling and PC1 use calibration molecules only.} "
            + r"\trackednew{Rank agreement is evaluated on held-out test molecules.}"
            + r"\par" + fallback_note(summary, tracked=True)
        ),
        r"\end{minipage}",
        "",
    ])


def main() -> None:
    TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    required = ["ligand_id", *PROTEINS, *BTOX_COLUMNS]
    missing = sorted(set(required).difference(df.columns))
    if missing:
        raise ValueError(f"Missing required input columns: {missing}")
    df[PROTEINS] = df[PROTEINS].apply(pd.to_numeric, errors="coerce")
    df[BTOX_COLUMNS] = (
        df[BTOX_COLUMNS].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(0.0, 1.0)
    )
    df = df.set_index("ligand_id")
    assert df.index.is_unique
    registry = pd.read_csv(SPLIT_PATH)
    rows: list[dict[str, object]] = []
    for split_index in sorted(registry.split_index.unique()):
        split = registry.loc[registry.split_index.eq(split_index)]
        seed = 42 + int(split_index)
        calibration_ids = split.loc[split["set"].eq("val"), "ligand_id"].astype(int).to_numpy()
        test_ids = split.loc[split["set"].eq("test"), "ligand_id"].astype(int).to_numpy()
        split_dir = MODEL_ROOT / f"split_{split_index:02d}" / "plain"
        calibration_path = split_dir / "calibration_predictions.csv"
        test_path = split_dir / "test_predictions_full_test.csv"
        conformal_path = split_dir / "conformal.json"
        calibration_pred = pd.read_csv(calibration_path).set_index("ligand_id")
        test_pred = pd.read_csv(test_path).set_index("ligand_id")
        assert calibration_pred.index.is_unique and test_pred.index.is_unique
        if set(calibration_ids) != set(calibration_pred.index.astype(int)):
            raise AssertionError(f"split {split_index}: registry/calibration prediction IDs differ")
        if set(test_ids) != set(test_pred.index.astype(int)):
            raise AssertionError(f"split {split_index}: registry/test prediction IDs differ")
        calibration = df.loc[calibration_pred.index]
        test = df.loc[test_pred.index]
        energy_cal = calibration[PROTEINS].to_numpy(float)
        energy_test = test[PROTEINS].to_numpy(float)
        if not np.isfinite(energy_cal).all() or not np.isfinite(energy_test).all():
            raise AssertionError(f"split {split_index}: non-finite docking score")
        raw_cal = energy_cal.mean(axis=1)
        raw_test = energy_test.mean(axis=1)
        z_cal, z_test = standardize_from_calibration(energy_cal, energy_test)
        standardized_mean_cal = z_cal.mean(axis=1)
        standardized_mean_test = z_test.mean(axis=1)
        pc1_cal, pc1_test, pc1_variance = pca_first_component(z_cal, z_test, raw_cal)
        coordinates = {
            "raw_emean": (raw_cal, raw_test, np.nan),
            "targetwise_standardized_mean": (standardized_mean_cal, standardized_mean_test, np.nan),
            "targetwise_standardized_pc1": (pc1_cal, pc1_test, pc1_variance),
        }
        with conformal_path.open(encoding="utf-8") as handle:
            global_half_width = float(json.load(handle)["half_width_quantiles"]["CI_0.90"])
        residuals = calibration_pred["abs_residual"].to_numpy(float)
        quantile_level = min(1.0, math.ceil((len(calibration_pred) + 1) * (1 - ALPHA)) / len(calibration_pred))
        y_true = test_pred["y_true"].to_numpy(float)
        y_pred = test_pred["y_pred"].to_numpy(float)
        for coordinate, (scalar_cal, scalar_test, explained_variance) in coordinates.items():
            feature_cal_raw = np.column_stack((scalar_cal, calibration[BTOX_COLUMNS].to_numpy(float)))
            feature_test_raw = np.column_stack((scalar_test, test[BTOX_COLUMNS].to_numpy(float)))
            feature_cal, feature_test = standardize_from_calibration(feature_cal_raw, feature_test_raw)
            half_width, neff = weighted_quantiles_and_neff(
                feature_test, feature_cal, residuals, quantile_level, LOCAL_BANDWIDTH
            )
            fallback = neff < NEFF_FALLBACK
            half_width[fallback] = global_half_width
            width = 2 * half_width
            coverage = np.abs(y_true - y_pred) <= half_width
            observed_rank = 1.0 if coordinate == "raw_emean" else float(spearmanr(raw_test, scalar_test).statistic)
            row = {
                "split_index": split_index,
                "seed": seed,
                "coordinate": coordinate,
                "n_calibration": len(calibration),
                "n_test": len(test),
                "rank_agreement_with_raw_emean_test": observed_rank,
                "pc1_explained_variance_calibration": explained_variance,
                "covered_count": int(coverage.sum()),
                "empirical_coverage_90": float(coverage.mean()),
                "mean_pi_width_90": float(width.mean()),
                "median_pi_width_90": float(np.median(width)),
                "fallback_count": int(fallback.sum()),
                "fallback_fraction": float(fallback.mean()),
            }
            rows.append(row)

    by_split = pd.DataFrame(rows)
    summary = summarize(by_split)
    summary.to_csv(TABLE_PATH, index=False)
    print("Docking-coordinate sensitivity: PASS")
    print(summary.to_string(index=False))
    print(f"Wrote {TABLE_PATH}")


if __name__ == "__main__":
    main()
