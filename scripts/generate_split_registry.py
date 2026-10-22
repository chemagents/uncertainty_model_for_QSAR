#!/usr/bin/env python3
"""Create the documented 30-split Butina registry for the corrected cohort."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rdkit
import sklearn
from rdkit import Chem, DataStructs, rdBase
from rdkit.Chem import rdFingerprintGenerator
from rdkit.ML.Cluster import Butina
from sklearn.model_selection import GroupShuffleSplit


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
SPLITS = DATA / "splits"
BUTINA = SPLITS / "butina"

SOURCE = DATA / "df_final.csv"
EXCLUSIONS = DATA / "excluded_molecules.csv"
LIGAND_ID = "ligand_id"
SMILES = "Canonical SMILES"
CLUSTER = "Butina_clusters"

N_MOLECULES = 12_650
RADIUS = 2
FP_SIZE = 2_048
INCLUDE_CHIRALITY = False
COUNT_SIMULATION = False
TANIMOTO_SIMILARITY_THRESHOLD = 0.65
BUTINA_DISTANCE_CUTOFF = 1.0 - TANIMOTO_SIMILARITY_THRESHOLD
SEEDS = tuple(range(42, 72))
TEST_SIZE = 0.20
VALIDATION_SIZE_WITHIN_DEVELOPMENT = 0.20


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def split_signature(frame: pd.DataFrame) -> str:
    payload = frame[["set", LIGAND_ID]].to_csv(index=False).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def load_active_cohort() -> tuple[pd.DataFrame, list[int]]:
    frame = pd.read_csv(SOURCE, usecols=[LIGAND_ID, SMILES])
    exclusions = pd.read_csv(EXCLUSIONS)
    excluded_ids = sorted(
        exclusions.loc[exclusions["scope"].eq("all_model_splits"), LIGAND_ID].astype(int).tolist()
    )
    frame[LIGAND_ID] = frame[LIGAND_ID].astype(int)
    frame = frame.sort_values(LIGAND_ID, kind="stable").reset_index(drop=True)

    if len(frame) != N_MOLECULES:
        raise AssertionError(f"Expected {N_MOLECULES:,} active molecules, found {len(frame):,}")
    if frame[LIGAND_ID].duplicated().any():
        raise AssertionError("The active cohort contains duplicate ligand IDs")
    if frame[SMILES].isna().any():
        raise AssertionError(f"The active cohort contains missing {SMILES!r} values")
    overlap = sorted(set(frame[LIGAND_ID]) & set(excluded_ids))
    if overlap:
        raise AssertionError(f"Excluded ligand IDs are unexpectedly present: {overlap}")
    return frame, excluded_ids


def build_clusters(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=RADIUS,
        fpSize=FP_SIZE,
        includeChirality=INCLUDE_CHIRALITY,
        countSimulation=COUNT_SIMULATION,
    )
    fingerprints = []
    for row in frame.itertuples(index=False):
        molecule = Chem.MolFromSmiles(getattr(row, "_1"))
        if molecule is None:
            raise ValueError(f"Cannot parse {SMILES!r} for ligand_id={getattr(row, LIGAND_ID)}")
        fingerprints.append(generator.GetFingerprint(molecule))

    n_molecules = len(fingerprints)
    distances = np.empty(n_molecules * (n_molecules - 1) // 2, dtype=np.float64)
    cursor = 0
    for index in range(1, n_molecules):
        similarities = np.asarray(
            DataStructs.BulkTanimotoSimilarity(fingerprints[index], fingerprints[:index]), dtype=np.float64
        )
        next_cursor = cursor + len(similarities)
        distances[cursor:next_cursor] = 1.0 - similarities
        cursor = next_cursor
    if cursor != len(distances):
        raise AssertionError("The pairwise-distance array is incomplete")

    raw_clusters = Butina.ClusterData(
        distances,
        n_molecules,
        BUTINA_DISTANCE_CUTOFF,
        isDistData=True,
        reordering=False,
    )
    cluster_ids = np.empty(n_molecules, dtype=np.int32)
    cluster_sizes = np.empty(n_molecules, dtype=np.int32)
    cluster_centers = np.empty(n_molecules, dtype=np.int64)
    summary_rows = []
    for cluster_id, members in enumerate(raw_clusters, start=1):
        member_indices = np.asarray(members, dtype=int)
        center_ligand_id = int(frame.iloc[member_indices][LIGAND_ID].min())
        cluster_ids[member_indices] = cluster_id
        cluster_sizes[member_indices] = len(member_indices)
        cluster_centers[member_indices] = center_ligand_id
        summary_rows.append(
            {
                CLUSTER: cluster_id,
                "cluster_size": int(len(member_indices)),
                "cluster_center_ligand_id": center_ligand_id,
            }
        )

    clusters = frame.copy()
    clusters.insert(0, "input_order", np.arange(n_molecules, dtype=int))
    clusters[CLUSTER] = cluster_ids
    clusters["cluster_size"] = cluster_sizes
    clusters["cluster_center_ligand_id"] = cluster_centers
    summary = pd.DataFrame(summary_rows)
    if len(clusters) != N_MOLECULES or clusters[CLUSTER].nunique() != len(summary):
        raise AssertionError("Invalid cluster assignment output")
    return clusters, summary


def build_split_registry(clusters: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    all_indices = np.arange(len(clusters))
    group_values = clusters[CLUSTER].to_numpy()
    split_rows = []
    summary_rows = []
    for split_index, seed in enumerate(SEEDS):
        first_stage = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=seed)
        development_indices, test_indices = next(first_stage.split(all_indices, groups=group_values))
        second_stage = GroupShuffleSplit(
            n_splits=1,
            test_size=VALIDATION_SIZE_WITHIN_DEVELOPMENT,
            random_state=seed,
        )
        train_local, validation_local = next(
            second_stage.split(development_indices, groups=group_values[development_indices])
        )
        set_indices = {
            "train": development_indices[train_local],
            "val": development_indices[validation_local],
            "test": test_indices,
        }
        cluster_sets = {name: set(group_values[indices]) for name, indices in set_indices.items()}
        shared_train_val = len(cluster_sets["train"] & cluster_sets["val"])
        shared_train_test = len(cluster_sets["train"] & cluster_sets["test"])
        shared_val_test = len(cluster_sets["val"] & cluster_sets["test"])
        if shared_train_val or shared_train_test or shared_val_test:
            raise AssertionError(f"Split {split_index} shares one or more Butina clusters")

        split_frame = pd.concat(
            [
                pd.DataFrame(
                    {
                        "set": set_name,
                        LIGAND_ID: np.sort(clusters.iloc[indices][LIGAND_ID].to_numpy(dtype=int)),
                    }
                )
                for set_name, indices in set_indices.items()
            ],
            ignore_index=True,
        )
        if len(split_frame) != N_MOLECULES or split_frame[LIGAND_ID].nunique() != N_MOLECULES:
            raise AssertionError(f"Split {split_index} does not assign every active molecule exactly once")
        split_frame.insert(0, "split_index", split_index)
        split_rows.append(split_frame)
        summary_rows.append(
            {
                "split_index": split_index,
                "seed": seed,
                "n_molecules": N_MOLECULES,
                "n_clusters": int(clusters[CLUSTER].nunique()),
                "split_signature_sha256": split_signature(split_frame),
                "shared_clusters_train_val": shared_train_val,
                "shared_clusters_train_test": shared_train_test,
                "shared_clusters_val_test": shared_val_test,
                "n_train": int(len(set_indices["train"])),
                "n_clusters_train": int(len(cluster_sets["train"])),
                "n_val": int(len(set_indices["val"])),
                "n_clusters_val": int(len(cluster_sets["val"])),
                "n_test": int(len(set_indices["test"])),
                "n_clusters_test": int(len(cluster_sets["test"])),
            }
        )
    registry = pd.concat(split_rows, ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    if len(registry) != N_MOLECULES * len(SEEDS):
        raise AssertionError("Unexpected number of registry rows")
    if not summary["split_signature_sha256"].is_unique:
        raise AssertionError("The 30 split assignments are not distinct")
    return registry, summary


def main() -> None:
    BUTINA.mkdir(parents=True, exist_ok=True)

    cohort, excluded_ids = load_active_cohort()
    clusters, cluster_summary = build_clusters(cohort)
    clusters_path = BUTINA / "clusters.csv"
    cluster_summary_path = BUTINA / "cluster_summary.csv"
    clusters.to_csv(clusters_path, index=False)
    cluster_summary.to_csv(cluster_summary_path, index=False)

    cluster_validation = {
        "success": True,
        "active_molecules": int(len(cohort)),
        "excluded_ligand_ids": excluded_ids,
        "excluded_ids_absent_from_active_table": True,
        "n_clustered_molecules": int(len(clusters)),
        "n_clusters": int(clusters[CLUSTER].nunique()),
        "n_singleton_clusters": int((cluster_summary["cluster_size"] == 1).sum()),
        "largest_cluster_size": int(cluster_summary["cluster_size"].max()),
        "clusters_sha256": sha256_file(clusters_path),
        "cluster_summary_sha256": sha256_file(cluster_summary_path),
    }
    write_json(BUTINA / "validation.json", cluster_validation)
    write_json(
        BUTINA / "metadata.json",
        {
            "protocol_name": "butina_achiral_morgan_r2_fp2048_v2",
            "purpose": "Structural clusters for the corrected 12,650-molecule cohort and registered split generation.",
            "input": {
                "df_final": "data/df_final.csv",
                "df_final_sha256": sha256_file(SOURCE),
                "excluded_molecules": "data/excluded_molecules.csv",
                "excluded_molecules_sha256": sha256_file(EXCLUSIONS),
                "excluded_ligand_ids": excluded_ids,
                "active_molecules": int(len(cohort)),
                "smiles_column": SMILES,
                "input_order": "ascending ligand_id from the active table",
            },
            "fingerprint": {
                "type": "Morgan bit vector",
                "radius": RADIUS,
                "fpSize": FP_SIZE,
                "includeChirality": INCLUDE_CHIRALITY,
                "countSimulation": COUNT_SIMULATION,
            },
            "distance": {
                "metric": "1 - TanimotoSimilarity",
                "dtype": "float64",
                "similarity_threshold": TANIMOTO_SIMILARITY_THRESHOLD,
                "distance_cutoff": BUTINA_DISTANCE_CUTOFF,
            },
            "butina": {
                "rdkit_call": "Butina.ClusterData(distances, n, 0.35, isDistData=True, reordering=False)",
                "isDistData": True,
                "reordering": False,
                "cluster_numbering": "RDKit output order, 1-based",
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "rdkit": rdBase.rdkitVersion,
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
                "rdkit_package": rdkit.__version__,
            },
            "outputs": {
                "directory": "data/splits/butina",
                "clusters": "clusters.csv",
                "cluster_summary": "cluster_summary.csv",
                "validation": "validation.json",
            },
            "result": cluster_validation,
        },
    )

    registry, split_summary = build_split_registry(clusters)
    registry_path = SPLITS / "split_indices.csv"
    summary_path = SPLITS / "split_summary.csv"
    registry.to_csv(registry_path, index=False)
    split_summary.to_csv(summary_path, index=False)
    split_validation = {
        "success": True,
        "n_splits": int(len(split_summary)),
        "seeds": list(SEEDS),
        "molecules_per_split": N_MOLECULES,
        "clusters_per_split": int(clusters[CLUSTER].nunique()),
        "all_split_signatures_distinct": bool(split_summary["split_signature_sha256"].is_unique),
        "registry_rows": int(len(registry)),
        "registry_sha256": sha256_file(registry_path),
        "summary_sha256": sha256_file(summary_path),
        "max_shared_clusters_between_sets": int(
            split_summary[
                ["shared_clusters_train_val", "shared_clusters_train_test", "shared_clusters_val_test"]
            ].to_numpy().max()
        ),
    }
    write_json(SPLITS / "split_validation.json", split_validation)
    write_json(
        SPLITS / "split_metadata.json",
        {
            "protocol_name": "butina_cluster_safe_random_splits_v2",
            "purpose": "Thirty reproducible cluster-disjoint train/validation/test assignments for the corrected cohort.",
            "input": {
                "clusters": "data/splits/butina/clusters.csv",
                "clusters_sha256": cluster_validation["clusters_sha256"],
                "input_order": "ascending ligand_id from clusters.csv",
                "n_molecules": N_MOLECULES,
                "n_clusters": int(clusters[CLUSTER].nunique()),
            },
            "algorithm": {
                "class": "sklearn.model_selection.GroupShuffleSplit",
                "first_stage": {
                    "groups": CLUSTER,
                    "test_size": TEST_SIZE,
                    "random_state": "seed for each split",
                    "output": "development and held-out test",
                },
                "second_stage": {
                    "input": "development",
                    "groups": CLUSTER,
                    "test_size": VALIDATION_SIZE_WITHIN_DEVELOPMENT,
                    "random_state": "same seed for each split",
                    "output": "train and validation",
                },
                "set_order": ["train", "val", "test"],
                "split_indices": list(range(len(SEEDS))),
                "seeds": list(SEEDS),
            },
            "environment": {
                "python": sys.version,
                "platform": platform.platform(),
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scikit_learn": sklearn.__version__,
            },
            "outputs": {
                "registry": "data/splits/split_indices.csv",
                "summary": "data/splits/split_summary.csv",
                "validation": "data/splits/split_validation.json",
            },
            "result": split_validation,
        },
    )

    print(f"Created {cluster_validation['n_clusters']:,} Butina clusters and {len(SEEDS)} registered splits.")
    print(f"Cluster checksum: {cluster_validation['clusters_sha256']}")
    print(f"Registry checksum: {split_validation['registry_sha256']}")


if __name__ == "__main__":
    main()
