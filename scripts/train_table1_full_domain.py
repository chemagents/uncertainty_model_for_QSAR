#!/usr/bin/env python
# coding: utf-8

# # 02 Model Training
#
# Notebook-first article reproduction. Calculations are kept in notebook cells.

# ## Model Training
# Reproduce the submitted full-domain models from the shared table and frozen legacy assignments.

# In[1]:


import json
import pickle
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results" / "table1_full_domain"
TABLES = RESULTS / "tables"
MODELS_DIR = RESULTS / "models"
SPLITS_DIR = RESULTS / "splits"
SPLIT_REGISTRY = DATA / "table1_full_domain_splits" / "split_indices.csv"
CLUSTER_ASSIGNMENTS = DATA / "butina_clusters.csv"

for path in [TABLES, MODELS_DIR, SPLITS_DIR]:
    path.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(ROOT / "scripts"))
from cluster_assignments import attach_cluster_assignments

TARGET_COL = "-lgLD50, mol/kg"
CLUSTER_COL = "Butina_clusters"
BBB_FILTER_COL = "bbb_rule_pass"
LIGAND_ID_COL = "ligand_id"
FP_PREFIX = "FP_"
RANDOM_STATE = 42

PROTEINS = [
    "1m2z", "1pbq", "1xoq", "2rh1", "2vt4", "2ydo", "2z5x", "3b66", "3kk6", "3ln1",
    "3rze", "4djh", "4ey7", "4iar", "4mqs", "4n6h", "5cxv", "5i71", "5tvn", "5u09",
    "5va1", "6cm4", "6kpf", "6kux", "6lqa", "6pdj", "6x3x", "6y1z", "7f8y", "7kwe",
    "7ljd", "7wc9", "7xnk", "7ym8", "8e9y", "8ef6", "8fhs", "8pjk", "8st0", "8wty",
    "8xvk", "8yn3", "9eo4", "V1A",
]
MODEL_ORDER = ["Baseline", "PCA", "ADME", "Plain", "BBB pass"]

import matplotlib
matplotlib.use("Agg")

from dataclasses import dataclass
from collections import defaultdict

from catboost import CatBoostRegressor
from mapie.regression import SplitConformalRegressor
from scipy.stats import spearmanr
from sklearn.decomposition import PCA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

TEST_SIZE = 0.2
VAL_SIZE = 0.2
N_PCA_COMPONENTS = 3
CONFIDENCE_LEVELS = [0.8, 0.9, 0.95]
CONFORMAL_ITER = 30
CALIB_SIZE = 500
THREAD_COUNT = -1
MODEL_SPLIT_INDEX = 1

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

FIXED_PARAMS = {
    "Baseline": {"iterations": 762, "depth": 6, "learning_rate": 0.22486425123692586, "l2_leaf_reg": 4.414780019047116, "random_strength": 1.2801770049351129, "bagging_temperature": 0.19362599980591533},
    "PCA": {"iterations": 762, "depth": 6, "learning_rate": 0.22486425123692586, "l2_leaf_reg": 4.414780019047116, "random_strength": 1.2801770049351129, "bagging_temperature": 0.19362599980591533},
    "ADME": {"iterations": 362, "depth": 8, "learning_rate": 0.1205712628744377, "l2_leaf_reg": 6.387926357773329, "random_strength": 0.31203728088487304, "bagging_temperature": 0.7799726016810132},
    "Plain": {"iterations": 708, "depth": 7, "learning_rate": 0.14074393812229913, "l2_leaf_reg": 3.536749818416572, "random_strength": 1.593003811844754, "bagging_temperature": 3.234900862052149},
    "BBB pass": {"iterations": 614, "depth": 7, "learning_rate": 0.14978277007160656, "l2_leaf_reg": 1.0200393001337464, "random_strength": 1.655136721037486, "bagging_temperature": 1.1852900897108616},
}

@dataclass
class ModelArtifact:
    model: CatBoostRegressor
    train_data: pd.DataFrame
    test_data: pd.DataFrame
    features: list[str]
    pca: PCA | None = None
    protein_fill_values: pd.Series | None = None


# In[2]:


def prepare_feature_sets(df: pd.DataFrame) -> dict[str, list[str]]:
    fp_cols = sorted([col for col in df.columns if col.startswith(FP_PREFIX)], key=lambda x: int(x.split("_")[1]))
    return {
        "baseline": ["MW, g/mol", "logP"] + fp_cols + PROTEINS,
        "pca": ["MW, g/mol", "logP"] + fp_cols + PROTEINS,
        "adme": ["MW, g/mol", "logP"] + fp_cols + PROTEINS + ADMET_FEATURES,
        "plain": ["MW, g/mol", "logP"] + fp_cols,
    }


def validate_columns(df: pd.DataFrame, feature_sets: dict[str, list[str]]) -> None:
    required = {TARGET_COL, CLUSTER_COL, BBB_FILTER_COL, LIGAND_ID_COL}
    for cols in feature_sets.values():
        required.update(cols)
    missing = sorted(col for col in required if col not in df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_split_frames(df: pd.DataFrame, split_index: int) -> dict[str, pd.DataFrame]:
    registry = pd.read_csv(SPLIT_REGISTRY)
    active = registry.loc[registry["split_index"].eq(split_index)].copy()
    required = {"train", "val", "test"}
    present = set(active["set"].astype(str))
    missing_sets = sorted(required - present)
    if missing_sets:
        raise ValueError(f"Split {split_index} is missing sets: {missing_sets}")
    by_id = df.copy()
    by_id[LIGAND_ID_COL] = by_id[LIGAND_ID_COL].astype(int)
    by_id = by_id.set_index(LIGAND_ID_COL, drop=False)
    available = set(by_id.index.astype(int))
    splits = {}
    for split_name in ["train", "val", "test"]:
        ids = active.loc[active["set"].eq(split_name), "ligand_id"].astype(int).tolist()
        missing_ids = sorted(set(ids) - available)
        if missing_ids:
            raise ValueError(f"Split {split_index}/{split_name} has missing ligand IDs: {missing_ids[:5]}")
        splits[split_name] = by_id.loc[ids].reset_index(drop=True)
    return splits


def save_splits(splits: dict[str, pd.DataFrame]) -> None:
    for split_name, split_df in splits.items():
        split_df[[LIGAND_ID_COL]].to_csv(SPLITS_DIR / f"{split_name}.csv", index=False)


def to_ld50(y_neglog):
    return np.power(10.0, -np.asarray(y_neglog, dtype=float))


def concordance_ccc(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mu_a, mu_b = a.mean(), b.mean()
    va, vb = a.var(), b.var()
    cov = np.mean((a - mu_a) * (b - mu_b))
    return float(2 * cov / (va + vb + (mu_a - mu_b) ** 2 + 1e-12))


def compute_metrics_neglog(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    ld_true = to_ld50(y_true)
    ld_pred = to_ld50(y_pred)
    rho, _ = spearmanr(y_true, y_pred)
    return {
        "RMSE": float(np.sqrt(mse)),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mse),
        "R2": float(r2_score(y_true, y_pred)),
        "MAPE": float(np.mean(np.abs((ld_true - ld_pred) / ld_true)) * 100.0),
        "Spearman": float(rho),
        "CCC": concordance_ccc(y_true, y_pred),
    }


def catboost_params(params: dict) -> dict:
    out = dict(params)
    out.update({
        "loss_function": "RMSE",
        "verbose": False,
        "random_seed": RANDOM_STATE,
        "allow_writing_files": False,
        "thread_count": THREAD_COUNT,
    })
    return out


def train_final_model(X: pd.DataFrame, y: pd.Series, groups: pd.Series, params: dict, model_name: str):
    gss_es = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=RANDOM_STATE)
    train_es_idx, val_es_idx = next(gss_es.split(X, y, groups))
    model = CatBoostRegressor(**catboost_params(params))
    model.fit(
        X.iloc[train_es_idx],
        y.iloc[train_es_idx],
        eval_set=(X.iloc[val_es_idx], y.iloc[val_es_idx]),
        early_stopping_rounds=200,
        use_best_model=True,
        verbose=False,
    )
    es_pred = model.predict(X.iloc[val_es_idx])
    es_rmse = float(np.sqrt(mean_squared_error(y.iloc[val_es_idx], es_pred)))
    print(f"{model_name}: early-stop RMSE={es_rmse:.4f}; trees={model.tree_count_}")
    return model, es_rmse


def add_pca_features(train_data: pd.DataFrame, test_data: pd.DataFrame):
    fill_values = train_data[PROTEINS].mean(numeric_only=True)
    train_proteins = train_data[PROTEINS].fillna(fill_values)
    test_proteins = test_data[PROTEINS].fillna(fill_values)
    pca = PCA(n_components=N_PCA_COMPONENTS, random_state=RANDOM_STATE)
    train_pca = pca.fit_transform(train_proteins)
    test_pca = pca.transform(test_proteins)
    train_result = train_data.copy()
    test_result = test_data.copy()
    for i in range(N_PCA_COMPONENTS):
        train_result[f"PC{i + 1}"] = train_pca[:, i]
        test_result[f"PC{i + 1}"] = test_pca[:, i]
    return train_result, test_result, pca, fill_values


def build_model_data(model_name: str, feature_sets: dict[str, list[str]], splits: dict[str, pd.DataFrame]):
    if model_name == "BBB pass":
        train_data = pd.concat([
            splits["train"].query(f"{BBB_FILTER_COL} == 1"),
            splits["val"].query(f"{BBB_FILTER_COL} == 1"),
        ], ignore_index=True)
        test_data = splits["test"].query(f"{BBB_FILTER_COL} == 1").copy()
        return train_data, test_data, feature_sets["baseline"], None, None

    train_data = pd.concat([splits["train"], splits["val"]], ignore_index=True)
    test_data = splits["test"].copy()
    if model_name == "Baseline":
        return train_data, test_data, feature_sets["baseline"], None, None
    if model_name == "ADME":
        return train_data, test_data, feature_sets["adme"], None, None
    if model_name == "Plain":
        return train_data, test_data, feature_sets["plain"], None, None
    if model_name == "PCA":
        train_pca, test_pca, pca, fill_values = add_pca_features(train_data, test_data)
        features = [f for f in feature_sets["pca"] if f not in PROTEINS] + ["PC1", "PC2", "PC3"]
        return train_pca, test_pca, features, pca, fill_values
    raise ValueError(model_name)


# In[3]:


df = attach_cluster_assignments(pd.read_csv(DATA / "df_final.csv"), CLUSTER_ASSIGNMENTS)
feature_sets = prepare_feature_sets(df)
validate_columns(df, feature_sets)

splits = load_split_frames(df, MODEL_SPLIT_INDEX)
save_splits(splits)
print({name: len(frame) for name, frame in splits.items()})


# In[4]:


artifacts: dict[str, ModelArtifact] = {}
training_infos: dict[str, dict] = {}
holdout_metrics: dict[str, dict] = {}

for model_name in MODEL_ORDER:
    print(f"\n=== Training {model_name} ===")
    train_data, test_data, features, pca, fill_values = build_model_data(model_name, feature_sets, splits)
    if len(train_data) == 0 or len(test_data) == 0:
        raise ValueError(f"Model {model_name} has empty train/test data")

    X_train = train_data[features]
    y_train = train_data[TARGET_COL]
    groups = train_data[CLUSTER_COL]
    X_test = test_data[features]
    y_test = test_data[TARGET_COL]

    model, es_rmse = train_final_model(X_train, y_train, groups, FIXED_PARAMS[model_name], model_name)
    pred = model.predict(X_test)
    metrics = compute_metrics_neglog(y_test, pred)
    print(f"{model_name}: holdout RMSE={metrics['RMSE']:.4f}; R2={metrics['R2']:.4f}; Spearman={metrics['Spearman']:.4f}")

    artifacts[model_name] = ModelArtifact(model=model, train_data=train_data, test_data=test_data, features=features, pca=pca, protein_fill_values=fill_values)
    training_infos[model_name] = {
        "best_params": FIXED_PARAMS[model_name],
        "early_stop_rmse": es_rmse,
        "n_samples": int(len(train_data)),
        "n_test": int(len(test_data)),
        "n_features": int(len(features)),
        "training_mode": "fixed",
    }
    holdout_metrics[model_name] = metrics

    with open(MODELS_DIR / f"{model_name}.pkl", "wb") as f:
        pickle.dump({
            "model": model,
            "train_data": train_data,
            "test_data": test_data,
            "features": features,
            "pca": pca,
            "protein_fill_values": fill_values,
            "training_info": training_infos[model_name],
            "holdout_metrics": metrics,
        }, f)

    pd.DataFrame({
        "ligand_id": test_data[LIGAND_ID_COL].astype(int).to_numpy(),
        "model": model_name,
        "y_true": y_test.to_numpy(float),
        "y_pred": pred,
    }).to_csv(TABLES / f"predictions_{model_name.replace(' ', '_')}.csv", index=False)

with open(MODELS_DIR / "experiment_config.json", "w") as f:
    json.dump({
        "model_order": MODEL_ORDER,
        "fixed_params": FIXED_PARAMS,
        "random_state": RANDOM_STATE,
        "test_size": TEST_SIZE,
        "val_size": VAL_SIZE,
        "n_pca_components": N_PCA_COMPONENTS,
        "confidence_levels": CONFIDENCE_LEVELS,
        "conformal_iter": CONFORMAL_ITER,
        "calib_size": CALIB_SIZE,
        "model_split_index": MODEL_SPLIT_INDEX,
        "input": "data/df_final.csv",
        "butina_clusters": "data/butina_clusters.csv",
        "split_registry": "data/table1_full_domain_splits/split_indices.csv",
    }, f, indent=2)


# In[5]:


def compute_conformal_metrics(model: CatBoostRegressor, test_data: pd.DataFrame, calib_size: int, n_iter: int) -> dict[str, float]:
    features = list(model.feature_names_)
    intervals_sizes = {level: [] for level in CONFIDENCE_LEVELS}
    metrics = defaultdict(list)
    max_calib = max(20, len(test_data) // 2)
    actual_calib_size = min(calib_size, max_calib)
    if len(test_data) - actual_calib_size < 20:
        raise ValueError(f"Not enough test rows for conformal split: n={len(test_data)}")

    for i in range(n_iter):
        calib = test_data.sample(actual_calib_size, random_state=i)
        test_true = test_data.drop(calib.index)
        conformal_model = SplitConformalRegressor(
            estimator=model,
            confidence_level=CONFIDENCE_LEVELS,
            conformity_score="gamma",
            prefit=True,
            n_jobs=-1,
        )
        conformal_model.conformalize(calib[features], calib[TARGET_COL])
        pred, intervals = conformal_model.predict_interval(test_true[features])
        for key, value in compute_metrics_neglog(test_true[TARGET_COL], pred).items():
            metrics[key].append(value)
        for j, level in enumerate(CONFIDENCE_LEVELS):
            intervals_sizes[level].append(float((intervals[:, 1, j] - intervals[:, 0, j]).mean()))

    result = {key: float(np.mean(values)) for key, values in metrics.items()}
    for level in CONFIDENCE_LEVELS:
        result[f"CI_{level}_size"] = float(np.mean(intervals_sizes[level]))
    result["calib_size"] = float(actual_calib_size)
    result["n_iter"] = float(n_iter)
    return result

conformal_rows = []
for model_name in MODEL_ORDER:
    print("Conformal intervals:", model_name)
    art = artifacts[model_name]
    conf = compute_conformal_metrics(art.model, art.test_data, calib_size=CALIB_SIZE, n_iter=CONFORMAL_ITER)
    conf["model"] = model_name
    conformal_rows.append(conf)

conformal_df = pd.DataFrame(conformal_rows).set_index("model")
conformal_df.to_csv(TABLES / "conformal_metrics.csv")

summary_rows = []
for model_name in MODEL_ORDER:
    row = {"model": model_name}
    row.update(holdout_metrics[model_name])
    for level in CONFIDENCE_LEVELS:
        row[f"CI_{level}_size"] = conformal_df.loc[model_name, f"CI_{level}_size"]
    row["n_train"] = training_infos[model_name]["n_samples"]
    row["n_test"] = training_infos[model_name]["n_test"]
    row["n_features"] = training_infos[model_name]["n_features"]
    row["early_stop_rmse"] = training_infos[model_name]["early_stop_rmse"]
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(TABLES / "summary.csv", index=False)
with open(TABLES / "training_info.json", "w") as f:
    json.dump(training_infos, f, indent=2)

print(summary_df[["model", "RMSE", "MAE", "MSE", "R2", "Spearman", "CCC", "CI_0.8_size", "CI_0.9_size", "CI_0.95_size"]].to_string(index=False))
