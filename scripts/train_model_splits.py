"""Train and persist the five frozen model specifications on all 30 split-registry rows.

The registered train/validation/test roles are strict: preprocessing and model
fitting use train only, split-conformal residual calibration uses validation only,
and test labels are read only for final reporting.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import pickle
import platform
from typing import Any

import catboost
from catboost import CatBoostRegressor
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import sklearn
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
MODELS = RESULTS / "models"
SPLITS = RESULTS / "splits"
TABLES = RESULTS / "tables"
TARGET = "-lgLD50, mol/kg"
LIGAND_ID = "ligand_id"
BBB = "bbb_rule_pass"
PROTEINS = [
    "1m2z", "1pbq", "1xoq", "2rh1", "2vt4", "2ydo", "2z5x", "3b66", "3kk6", "3ln1",
    "3rze", "4djh", "4ey7", "4iar", "4mqs", "4n6h", "5cxv", "5i71", "5tvn", "5u09",
    "5va1", "6cm4", "6kpf", "6kux", "6lqa", "6pdj", "6x3x", "7f8y", "7kwe",
    "7ljd", "7wc9", "7xnk", "7ym8", "8e9y", "8ef6", "8fhs", "8pjk", "8st0", "8wty",
    "8xvk", "8yn3", "9eo4", "V1A",
]
ADMET_FEATURES = [
    "Vol", "Dense", "nHA", "nHD", "TPSA", "nRot", "nRing", "MaxRing", "nHet", "fChar", "nRig",
    "Flex", "nStereo", "gasa", "QED", "Synth", "Fsp3", "MCE-18", "Natural Product-likeness",
    "Alarm_NMR", "BMS", "Chelating", "PAINS", "Lipinski", "Pfizer", "GSK", "GoldenTriangle", "logS",
    "logD", "logP_admet", "mp", "bp", "pka_acidic", "pka_basic", "caco2", "MDCK", "PAMPA",
    "pgp_inh", "pgp_sub", "hia", "f20", "f30", "f50", "OATP1B1", "OATP1B3", "BCRP", "BSEP",
    "BBB", "MRP1", "PPB", "logVDss", "Fu", "CYP1A2-inh", "CYP1A2-sub", "CYP2C19-inh",
    "CYP2C19-sub", "CYP2C9-inh", "CYP2C9-sub", "CYP2D6-inh", "CYP2D6-sub", "CYP3A4-inh",
    "CYP3A4-sub", "CYP2B6-inh", "CYP2B6-sub", "CYP2C8-inh", "LM-human", "cl-plasma", "t0.5",
]
MODEL_ORDER = ["Baseline", "PCA", "ADME", "Plain", "BBB pass"]
CONFIDENCE_LEVELS = (0.80, 0.90, 0.95)

# These tree counts are part of the frozen reference configuration for the
# 30-split evaluation.
FROZEN_PARAMS = {
    "Baseline": {"iterations": 592, "depth": 6, "learning_rate": 0.22486425123692586, "l2_leaf_reg": 4.414780019047116, "random_strength": 1.2801770049351129, "bagging_temperature": 0.19362599980591533},
    "PCA": {"iterations": 594, "depth": 6, "learning_rate": 0.22486425123692586, "l2_leaf_reg": 4.414780019047116, "random_strength": 1.2801770049351129, "bagging_temperature": 0.19362599980591533},
    "ADME": {"iterations": 360, "depth": 8, "learning_rate": 0.1205712628744377, "l2_leaf_reg": 6.387926357773329, "random_strength": 0.31203728088487304, "bagging_temperature": 0.7799726016810132},
    "Plain": {"iterations": 690, "depth": 7, "learning_rate": 0.14074393812229913, "l2_leaf_reg": 3.536749818416572, "random_strength": 1.593003811844754, "bagging_temperature": 3.234900862052149},
    "BBB pass": {"iterations": 605, "depth": 7, "learning_rate": 0.14978277007160656, "l2_leaf_reg": 1.0200393001337464, "random_strength": 1.655136721037486, "bagging_temperature": 1.1852900897108616},
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=str) + "\n")


def slug(model_name: str) -> str:
    return model_name.lower().replace(" ", "_")


def feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    fp = sorted((column for column in df.columns if column.startswith("FP_")), key=lambda column: int(column.split("_")[1]))
    plain = ["MW, g/mol", "logP"] + fp
    return {
        "Baseline": plain + PROTEINS,
        "PCA": plain + PROTEINS,
        "ADME": plain + PROTEINS + ADMET_FEATURES,
        "Plain": plain,
        "BBB pass": plain + PROTEINS,
    }


def ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    mean_true, mean_pred = y_true.mean(), y_pred.mean()
    variance_true, variance_pred = y_true.var(), y_pred.var()
    covariance = np.mean((y_true - mean_true) * (y_pred - mean_pred))
    return float(2 * covariance / (variance_true + variance_pred + (mean_true - mean_pred) ** 2 + 1e-12))


def point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": mse,
        "R2": float(r2_score(y_true, y_pred)),
        "Spearman": float(spearmanr(y_true, y_pred).statistic),
        "CCC": ccc(y_true, y_pred),
    }


def split_quantile(residuals: np.ndarray, confidence_level: float) -> float:
    rank = min(1.0, np.ceil((len(residuals) + 1) * confidence_level) / len(residuals))
    return float(np.quantile(residuals, rank, method="higher"))


def conformal_metrics(y_val: np.ndarray, pred_val: np.ndarray, y_test: np.ndarray, pred_test: np.ndarray) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    residuals = np.abs(y_val - pred_val)
    metrics: dict[str, float] = {}
    intervals: dict[str, np.ndarray] = {}
    for confidence_level in CONFIDENCE_LEVELS:
        quantile = split_quantile(residuals, confidence_level)
        key = f"CI_{confidence_level:.2f}"
        lower, upper = pred_test - quantile, pred_test + quantile
        intervals[key] = np.column_stack([lower, upper])
        metrics[f"{key}_size"] = float(2 * quantile)
        metrics[f"{key}_coverage"] = float(((y_test >= lower) & (y_test <= upper)).mean())
        metrics[f"{key}_quantile"] = quantile
    return metrics, intervals


def load_frames(df: pd.DataFrame, registry: pd.DataFrame, split_index: int) -> dict[str, pd.DataFrame]:
    membership = registry.loc[registry["split_index"].eq(split_index)]
    indexed = df.set_index(LIGAND_ID, drop=False)
    frames = {}
    for role in ("train", "val", "test"):
        ids = membership.loc[membership["set"].eq(role), LIGAND_ID].astype(int).tolist()
        frames[role] = indexed.loc[ids].reset_index(drop=True)
    return frames


def transform_pca(train: pd.DataFrame, val: pd.DataFrame, test: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], PCA, pd.Series, list[str]]:
    fill_values = train[PROTEINS].mean(numeric_only=True).fillna(0.0)
    pca = PCA(n_components=3, random_state=42)
    train_components = pca.fit_transform(train[PROTEINS].fillna(fill_values))
    transformed = {}
    for role, frame, components in (
        ("train", train, train_components),
        ("val", val, pca.transform(val[PROTEINS].fillna(fill_values))),
        ("test", test, pca.transform(test[PROTEINS].fillna(fill_values))),
    ):
        result = frame.copy()
        for index in range(3):
            result[f"PC{index + 1}"] = components[:, index]
        transformed[role] = result
    fp = sorted((column for column in train.columns if column.startswith("FP_")), key=lambda column: int(column.split("_")[1]))
    return transformed, pca, fill_values, ["MW, g/mol", "logP"] + fp + ["PC1", "PC2", "PC3"]


def model_parameters(model_name: str) -> dict[str, Any]:
    params = dict(FROZEN_PARAMS[model_name])
    params.update({
        "loss_function": "RMSE",
        "verbose": False,
        "random_seed": 42,
        "allow_writing_files": False,
        "thread_count": -1,
    })
    return params


def prediction_frame(frame: pd.DataFrame, prediction: np.ndarray, intervals: dict[str, np.ndarray]) -> pd.DataFrame:
    result = pd.DataFrame({LIGAND_ID: frame[LIGAND_ID].astype(int), "y_true": frame[TARGET].astype(float), "y_pred": prediction})
    for key, values in intervals.items():
        level = key.replace("CI_", "")
        result[f"lower_{level}"] = values[:, 0]
        result[f"upper_{level}"] = values[:, 1]
        result[f"covered_{level}"] = ((result["y_true"] >= values[:, 0]) & (result["y_true"] <= values[:, 1])).astype(int)
    return result


def train_one(
    model_name: str,
    frames: dict[str, pd.DataFrame],
    all_features: dict[str, list[str]],
    model_dir: Path,
    evaluation_frames: dict[str, pd.DataFrame],
) -> list[dict[str, Any]]:
    if model_name == "BBB pass":
        source_frames = {role: frame.loc[frame[BBB].eq(1)].copy() for role, frame in frames.items()}
        features = all_features[model_name]
        pca = None
        fill_values = None
    elif model_name == "PCA":
        source_frames, pca, fill_values, features = transform_pca(frames["train"], frames["val"], frames["test"])
    else:
        source_frames = frames
        features = all_features[model_name]
        pca = None
        fill_values = None

    if min(len(source_frames[role]) for role in ("train", "val", "test")) == 0:
        raise RuntimeError(f"{model_name}: an active split role is empty")
    model = CatBoostRegressor(**model_parameters(model_name))
    model.fit(source_frames["train"][features], source_frames["train"][TARGET])
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_model(model_dir / "model.cbm")

    if pca is not None:
        with (model_dir / "pca.pkl").open("wb") as stream:
            pickle.dump(pca, stream)
    preprocessing = {
        "features": features,
        "pca": pca is not None,
        "protein_fill_values": None if fill_values is None else {key: float(value) for key, value in fill_values.items()},
        "train_only_preprocessing": True,
    }
    write_json(model_dir / "preprocess.json", preprocessing)

    pred_val = model.predict(source_frames["val"][features])
    calibration = pd.DataFrame({
        LIGAND_ID: source_frames["val"][LIGAND_ID].astype(int),
        "y_true": source_frames["val"][TARGET].astype(float),
        "y_pred": pred_val,
    })
    calibration["abs_residual"] = np.abs(calibration["y_true"] - calibration["y_pred"])
    calibration.to_csv(model_dir / "calibration_predictions.csv", index=False)

    quantiles = {f"CI_{level:.2f}": split_quantile(calibration["abs_residual"].to_numpy(), level) for level in CONFIDENCE_LEVELS}
    write_json(model_dir / "conformal.json", {
        "method": "split_conformal_absolute_residual",
        "calibration_role": "val",
        "confidence_levels": list(CONFIDENCE_LEVELS),
        "half_width_quantiles": quantiles,
        "n_calibration": int(len(calibration)),
    })

    rows = []
    for domain, test_frame in evaluation_frames.items():
        if domain == "bbb_pass_test" and model_name == "BBB pass":
            test_frame = source_frames["test"]
        elif model_name == "PCA":
            transformed, _, _, _ = transform_pca(frames["train"], frames["val"], test_frame)
            test_frame = transformed["test"]
        prediction = model.predict(test_frame[features])
        y_test = test_frame[TARGET].to_numpy(float)
        intervals = {
            key: np.column_stack([prediction - quantile, prediction + quantile])
            for key, quantile in quantiles.items()
        }
        prediction_frame(test_frame, prediction, intervals).to_csv(model_dir / f"test_predictions_{domain}.csv", index=False)
        row = {
            "model": model_name,
            "evaluation_domain": domain,
            "n_train": int(len(source_frames["train"])),
            "n_calibration": int(len(source_frames["val"])),
            "n_test": int(len(test_frame)),
            "n_features": int(len(features)),
            "model_file": str((model_dir / "model.cbm").relative_to(ROOT)),
            **point_metrics(y_test, prediction),
        }
        for key, interval in intervals.items():
            row[f"{key}_size"] = float((interval[:, 1] - interval[:, 0]).mean())
            row[f"{key}_coverage"] = float(((y_test >= interval[:, 0]) & (y_test <= interval[:, 1])).mean())
        rows.append(row)
    write_json(model_dir / "metrics.json", {"rows": rows, "tree_count": int(model.tree_count_)})
    return rows


def completed_rows(model_dir: Path) -> list[dict[str, Any]] | None:
    metrics_path = model_dir / "metrics.json"
    if not (metrics_path.is_file() and (model_dir / "model.cbm").is_file()):
        return None
    return json.loads(metrics_path.read_text())["rows"]


def summarize(metrics: pd.DataFrame) -> pd.DataFrame:
    metric_columns = [
        "RMSE", "MAE", "MSE", "R2", "Spearman", "CCC",
        "CI_0.80_size", "CI_0.90_size", "CI_0.95_size",
        "CI_0.80_coverage", "CI_0.90_coverage", "CI_0.95_coverage",
        "n_train", "n_calibration", "n_test",
    ]
    grouped = metrics.groupby(["model", "evaluation_domain"], sort=False)
    rows = []
    for (model, domain), group in grouped:
        row: dict[str, Any] = {"model": model, "evaluation_domain": domain, "n_splits": int(len(group))}
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_std"] = float(group[metric].std(ddof=1))
        row["n_test_min"] = int(group["n_test"].min())
        row["n_test_max"] = int(group["n_test"].max())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--resume", action="store_true", help="Reuse a completed model directory when present.")
    parser.add_argument("--split-indices", help="Comma-separated registered split indices; default: all 30.")
    args = parser.parse_args()

    raw = pd.read_csv(DATA / "df_final.csv")
    excluded = set(pd.read_csv(DATA / "excluded_molecules.csv").query("scope == 'all_model_splits'")[LIGAND_ID].astype(int))
    df = raw.loc[~raw[LIGAND_ID].astype(int).isin(excluded)].copy()
    registry_path = SPLITS / "split_indices.csv"
    registry = pd.read_csv(registry_path)
    if len(df) != 12650 or registry[LIGAND_ID].isin(excluded).any():
        raise RuntimeError("The active 12,650-molecule cohort or split registry is invalid")
    required = set(feature_sets(df)["ADME"]) | {TARGET, LIGAND_ID, BBB}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Missing required columns: {missing}")

    write_json(MODELS / "butina30_protocol.json", {
        "protocol": "butina30_train_val_split_conformal_v1",
        "source_data": "data/df_final.csv",
        "source_data_sha256": sha256(DATA / "df_final.csv"),
        "split_registry": "results/splits/split_indices.csv",
        "split_registry_sha256": sha256(registry_path),
        "raw_source_rows": int(len(raw)),
        "active_modeling_cohort": int(len(df)),
        "excluded_all_model_splits": sorted(excluded),
        "roles": {"train": "fit model and preprocessing", "val": "split-conformal calibration", "test": "one final evaluation"},
        "model_order": MODEL_ORDER,
        "frozen_parameters": FROZEN_PARAMS,
        "confidence_levels": list(CONFIDENCE_LEVELS),
        "environment": {"python": platform.python_version(), "catboost": catboost.__version__, "scikit_learn": sklearn.__version__},
    })

    all_features = feature_sets(df)
    all_rows = []
    pair_rows = []
    split_indices = sorted(registry["split_index"].unique())
    if args.split_indices:
        requested = {int(value) for value in args.split_indices.split(",") if value.strip()}
        unknown = requested - set(split_indices)
        if unknown:
            raise ValueError(f"Unknown split indices: {sorted(unknown)}")
        split_indices = [index for index in split_indices if index in requested]
    for split_index in split_indices:
        split_dir = MODELS / f"split_{split_index:02d}"
        split_dir.mkdir(parents=True, exist_ok=True)
        frames = load_frames(df, registry, int(split_index))
        bbb_test = frames["test"].loc[frames["test"][BBB].eq(1)].copy()
        if bbb_test.empty:
            raise RuntimeError(f"Split {split_index} has no BBB-pass test molecules")
        write_json(split_dir / "split_manifest.json", {
            "split_index": int(split_index),
            "seed": int(42 + split_index),
            "counts": {role: int(len(frame)) for role, frame in frames.items()},
            "bbb_pass_counts": {role: int(frame[BBB].eq(1).sum()) for role, frame in frames.items()},
            "id_sha256": {role: hashlib.sha256(",".join(map(str, frame[LIGAND_ID].astype(int))).encode()).hexdigest() for role, frame in frames.items()},
        })
        print(f"split_{split_index:02d}: train={len(frames['train'])}, val={len(frames['val'])}, test={len(frames['test'])}, bbb_test={len(bbb_test)}", flush=True)

        plain_bbb_ids: set[int] | None = None
        for model_name in MODEL_ORDER:
            model_dir = split_dir / slug(model_name)
            rows = completed_rows(model_dir) if args.resume else None
            if rows is None:
                domains = {"full_test": frames["test"]} if model_name != "BBB pass" else {}
                if model_name in {"Plain", "BBB pass"}:
                    domains["bbb_pass_test"] = bbb_test
                rows = train_one(model_name, frames, all_features, model_dir, domains)
            for row in rows:
                row["split_index"] = int(split_index)
                row["seed"] = int(42 + split_index)
                all_rows.append(row)
                if model_name == "Plain" and row["evaluation_domain"] == "bbb_pass_test":
                    plain_bbb_ids = set(pd.read_csv(model_dir / "test_predictions_bbb_pass_test.csv")[LIGAND_ID].astype(int))

        bbb_ids = set(pd.read_csv(split_dir / "bbb_pass" / "test_predictions_bbb_pass_test.csv")[LIGAND_ID].astype(int))
        pair_rows.append({
            "split_index": int(split_index),
            "seed": int(42 + split_index),
            "plain_bbb_test_n": int(len(plain_bbb_ids or set())),
            "bbb_pass_test_n": int(len(bbb_ids)),
            "same_test_ids": plain_bbb_ids == bbb_ids,
        })
        if plain_bbb_ids != bbb_ids:
            raise RuntimeError(f"Split {split_index}: Plain and BBB-pass test IDs differ")

    metrics = pd.DataFrame(all_rows).sort_values(["split_index", "model", "evaluation_domain"])
    metrics.to_csv(TABLES / "model_comparison_metrics.csv", index=False)
    summary = summarize(metrics)
    summary.to_csv(TABLES / "model_comparison_summary.csv", index=False)
    pairs = pd.DataFrame(pair_rows)
    pairs.to_csv(TABLES / "model_comparison_bbb_pair_audit.csv", index=False)
    if not pairs["same_test_ids"].all():
        raise RuntimeError("BBB pair audit failed")
    write_json(TABLES / "model_comparison_config.json", {
        "protocol": "butina30_train_val_split_conformal_v1",
        "metrics_file": "results/tables/model_comparison_metrics.csv",
        "summary_file": "results/tables/model_comparison_summary.csv",
        "bbb_pair_audit": "results/tables/model_comparison_bbb_pair_audit.csv",
        "model_root": "results/models/split_00 ... results/models/split_29",
        "figure_1_panel_a": ["Baseline", "PCA", "ADME", "Plain"],
        "figure_1_panel_b": ["Plain", "BBB pass"],
    })
    print(f"Saved model artifacts and metrics for {len(split_indices)} split(s).", flush=True)


if __name__ == "__main__":
    main()
