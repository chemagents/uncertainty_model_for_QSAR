"""Generate non-model reproducibility audits requested in Reviewer 1, Comment 8."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
TABLES = ROOT / "results" / "tables"
ID = "ligand_id"
COMPONENTS = ["hERG", "Respiratory", "Neurotoxicity-DI"]
PROTEINS = [
    "1m2z", "1pbq", "1xoq", "2rh1", "2vt4", "2ydo", "2z5x", "3b66", "3kk6", "3ln1",
    "3rze", "4djh", "4ey7", "4iar", "4mqs", "4n6h", "5cxv", "5i71", "5tvn", "5u09",
    "5va1", "6cm4", "6kpf", "6kux", "6lqa", "6pdj", "6x3x", "6y1z", "7f8y", "7kwe",
    "7ljd", "7wc9", "7xnk", "7ym8", "8e9y", "8ef6", "8fhs", "8pjk", "8st0", "8wty",
    "8xvk", "8yn3", "9eo4", "V1A",
]
BOOTSTRAP_RESAMPLES = 5000
BOOTSTRAP_SEED = 20260915
Z_95 = 1.959963984540054


def wilson_interval(successes: int, total: int) -> tuple[float, float]:
    proportion = successes / total
    denominator = 1 + Z_95**2 / total
    center = (proportion + Z_95**2 / (2 * total)) / denominator
    radius = Z_95 * np.sqrt(proportion * (1 - proportion) / total + Z_95**2 / (4 * total**2)) / denominator
    return float(center - radius), float(center + radius)


def main() -> None:
    raw = pd.read_csv(DATA / "df_final.csv")
    excluded = set(pd.read_csv(DATA / "excluded_molecules.csv").query("scope == 'all_model_splits'")[ID].astype(int))
    df = raw.loc[~raw[ID].astype(int).isin(excluded)].reset_index(drop=True)
    if len(df) != 12650:
        raise RuntimeError(f"Expected the active 12,650-molecule cohort, found {len(df)}")

    components = df[COMPONENTS].apply(pd.to_numeric, errors="coerce").fillna(0.0).clip(0, 1)
    btox = components.sum(axis=1).to_numpy(float)
    mean_energy = np.nanmean(df[PROTEINS].to_numpy(float), axis=1)
    observed_btox = float(np.median(btox))
    observed_energy = float(np.median(mean_energy))
    thresholds = pd.read_csv(TABLES / "conformal_quadrant_thresholds.csv").iloc[0]
    if not np.isclose(observed_btox, thresholds["btox_median"]) or not np.isclose(observed_energy, thresholds["mean_E_44_median"]):
        raise RuntimeError("Recomputed cohort medians do not match the stored quadrant thresholds")

    rng = np.random.default_rng(BOOTSTRAP_SEED)
    bootstrap = np.empty((BOOTSTRAP_RESAMPLES, 2), dtype=float)
    for index in range(BOOTSTRAP_RESAMPLES):
        sample = rng.integers(0, len(df), size=len(df))
        bootstrap[index, 0] = np.median(btox[sample])
        bootstrap[index, 1] = np.median(mean_energy[sample])
    replicates = pd.DataFrame({
        "replicate": np.arange(1, BOOTSTRAP_RESAMPLES + 1),
        "btox_median": bootstrap[:, 0],
        "mean_E_44_median": bootstrap[:, 1],
    })
    replicates.to_csv(TABLES / "conformal_quadrant_threshold_bootstrap_replicates.csv", index=False)

    summary = pd.DataFrame([
        {
            "coordinate": "B_Tox",
            "n_active_cohort": len(df),
            "observed_active_cohort_median": observed_btox,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_mean": float(bootstrap[:, 0].mean()),
            "bootstrap_sample_sd": float(bootstrap[:, 0].std(ddof=1)),
            "bootstrap_percentile95_low": float(np.quantile(bootstrap[:, 0], 0.025)),
            "bootstrap_percentile95_high": float(np.quantile(bootstrap[:, 0], 0.975)),
        },
        {
            "coordinate": "E_mean",
            "n_active_cohort": len(df),
            "observed_active_cohort_median": observed_energy,
            "bootstrap_resamples": BOOTSTRAP_RESAMPLES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_mean": float(bootstrap[:, 1].mean()),
            "bootstrap_sample_sd": float(bootstrap[:, 1].std(ddof=1)),
            "bootstrap_percentile95_low": float(np.quantile(bootstrap[:, 1], 0.025)),
            "bootstrap_percentile95_high": float(np.quantile(bootstrap[:, 1], 0.975)),
        },
    ])
    summary.to_csv(TABLES / "conformal_quadrant_threshold_bootstrap_summary.csv", index=False)

    figure5_grid = pd.DataFrame([
        {
            "coordinate": "E_mean",
            "grid_quantile_low": float(np.percentile(mean_energy, 0.5)),
            "grid_quantile_high": float(np.percentile(mean_energy, 99.5)),
            "display_axis_low": -9.0,
            "display_axis_high": -4.0,
            "grid_cells_per_axis": 80,
            "coordinate_rule": "active-cohort 0.5th to 99.5th percentile; displayed x-axis fixed to -9 to -4",
        },
        {
            "coordinate": "B_Tox",
            "grid_quantile_low": float(np.percentile(btox, 0.5)),
            "grid_quantile_high": float(np.percentile(btox, 99.5)),
            "display_axis_low": float(np.percentile(btox, 0.5)),
            "display_axis_high": float(np.percentile(btox, 99.5)),
            "grid_cells_per_axis": 80,
            "coordinate_rule": "active-cohort 0.5th to 99.5th percentile",
        },
    ])
    figure5_grid.to_csv(TABLES / "conformal_figure5_grid_metadata.csv", index=False)

    quadrants = pd.read_csv(TABLES / "conformal_quadrants.csv")
    rows = []
    for _, row in quadrants.iterrows():
        successes = int(row["Interval_lt_global_n"])
        total = int(row["Interval_lt_global_denom"])
        low, high = wilson_interval(successes, total)
        rows.append({
            "stratum": row["Quadrant"],
            "interval_lt_global_n": successes,
            "interval_lt_global_denom": total,
            "interval_lt_global_fraction": successes / total,
            "wilson95_low": low,
            "wilson95_high": high,
        })
    successes = int(quadrants["Interval_lt_global_n"].sum())
    total = int(quadrants["Interval_lt_global_denom"].sum())
    low, high = wilson_interval(successes, total)
    rows.append({
        "stratum": "Overall held-out test set",
        "interval_lt_global_n": successes,
        "interval_lt_global_denom": total,
        "interval_lt_global_fraction": successes / total,
        "wilson95_low": low,
        "wilson95_high": high,
    })
    pd.DataFrame(rows).to_csv(TABLES / "conformal_interval_lt_global_wilson.csv", index=False)
    print("Wrote Comment 8 bootstrap and Wilson audits.")


if __name__ == "__main__":
    main()
