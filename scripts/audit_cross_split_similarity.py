#!/usr/bin/env python3
"""Audit residual structural similarity across the 30 registered splits."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
SPLITS = DATA / "splits"
TABLES = RESULTS / "tables"

SOURCE = DATA / "df_final.csv"
REGISTRY = SPLITS / "split_indices.csv"
SPLIT_METADATA = SPLITS / "split_metadata.json"
LIGAND_ID = "ligand_id"
SMILES = "Canonical SMILES"
N_MOLECULES = 12_650
COMPARISONS = (("val", "train"), ("test", "train"), ("test", "val"))
THRESHOLDS = (0.70, 0.80, 0.90)
STATISTICS = (
    "median",
    "p90",
    "p95",
    "p99",
    "maximum",
    "fraction_ge_0.70",
    "fraction_ge_0.80",
    "fraction_ge_0.90",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def validate_split_protocol() -> dict:
    metadata = json.loads(SPLIT_METADATA.read_text(encoding="utf-8"))
    fingerprint = metadata["input"] if "fingerprint" not in metadata else metadata["fingerprint"]
    # The fingerprint description is stored in the Butina metadata, while the
    # split metadata identifies the cluster artifact. Read the former explicitly.
    butina_metadata = json.loads((SPLITS / "butina" / "metadata.json").read_text(encoding="utf-8"))
    fingerprint = butina_metadata["fingerprint"]
    expected = {
        "radius": 2,
        "fpSize": 2048,
        "includeChirality": False,
        "countSimulation": False,
    }
    if {key: fingerprint[key] for key in expected} != expected:
        raise AssertionError("Similarity audit must use the registered split fingerprint")
    return {
        "split_metadata_sha256": sha256_file(SPLIT_METADATA),
        "butina_metadata_sha256": sha256_file(SPLITS / "butina" / "metadata.json"),
        "fingerprint": fingerprint,
        "n_splits": metadata["result"]["n_splits"],
    }


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    structures = pd.read_csv(SOURCE, usecols=[LIGAND_ID, SMILES])
    structures[LIGAND_ID] = structures[LIGAND_ID].astype(int)
    if len(structures) != N_MOLECULES or structures[LIGAND_ID].nunique() != N_MOLECULES:
        raise AssertionError("Unexpected active molecular cohort")
    if structures[SMILES].isna().any():
        raise AssertionError("Canonical SMILES contains missing values")

    registry = pd.read_csv(REGISTRY)
    required = {"split_index", "set", LIGAND_ID}
    if missing := required - set(registry.columns):
        raise ValueError(f"Split registry lacks columns: {sorted(missing)}")
    registry[LIGAND_ID] = registry[LIGAND_ID].astype(int)
    if registry["split_index"].nunique() != 30 or len(registry) != N_MOLECULES * 30:
        raise AssertionError("Unexpected split registry size")
    if set(registry["set"].unique()) != {"train", "val", "test"}:
        raise AssertionError("Unexpected split-set labels")
    for split_index, split in registry.groupby("split_index", sort=True):
        if len(split) != N_MOLECULES or split[LIGAND_ID].nunique() != N_MOLECULES:
            raise AssertionError(f"Split {split_index} does not assign each molecule exactly once")
    return structures, registry


def build_fingerprints(structures: pd.DataFrame) -> dict[int, object]:
    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048,
        includeChirality=False,
        countSimulation=False,
    )
    fingerprints = {}
    for row in structures.itertuples(index=False):
        molecule = Chem.MolFromSmiles(getattr(row, "_1"))
        if molecule is None:
            raise ValueError(f"Cannot parse {SMILES!r} for ligand_id={row.ligand_id}")
        fingerprints[int(row.ligand_id)] = generator.GetFingerprint(molecule)
    if len(fingerprints) != N_MOLECULES:
        raise AssertionError("Incomplete fingerprint dictionary")
    return fingerprints


def calculate_nearest_neighbours(registry: pd.DataFrame, fingerprints: dict[int, object]) -> pd.DataFrame:
    rows = []
    for split_index, split in registry.groupby("split_index", sort=True):
        seed = int(split_index) + 42
        ids_by_set = {
            name: group[LIGAND_ID].astype(int).tolist() for name, group in split.groupby("set", sort=False)
        }
        for query_set, reference_set in COMPARISONS:
            reference_ids = ids_by_set[reference_set]
            reference_fps = [fingerprints[ligand_id] for ligand_id in reference_ids]
            for ligand_id in ids_by_set[query_set]:
                similarities = DataStructs.BulkTanimotoSimilarity(fingerprints[ligand_id], reference_fps)
                max_similarity = max(similarities)
                nearest_ligand_id = min(
                    reference_ids[index]
                    for index, value in enumerate(similarities)
                    if value == max_similarity
                )
                rows.append(
                    {
                        "split_index": int(split_index),
                        "seed": seed,
                        "query_set": query_set,
                        "reference_set": reference_set,
                        LIGAND_ID: ligand_id,
                        "nearest_ligand_id": nearest_ligand_id,
                        "max_tanimoto_similarity": max_similarity,
                    }
                )
    nearest = pd.DataFrame(rows)
    expected_rows = sum(
        len(registry.loc[registry["set"].eq(query_set)])
        for query_set, _ in COMPARISONS
    )
    if len(nearest) != expected_rows:
        raise AssertionError("Incomplete nearest-neighbour calculation")
    return nearest


def similarity_summary(values: pd.Series) -> dict[str, float | int]:
    summary: dict[str, float | int] = {
        "n_queries": int(len(values)),
        "median": float(values.median()),
        "p90": float(values.quantile(0.90)),
        "p95": float(values.quantile(0.95)),
        "p99": float(values.quantile(0.99)),
        "maximum": float(values.max()),
    }
    for threshold in THRESHOLDS:
        summary[f"fraction_ge_{threshold:.2f}"] = float((values >= threshold).mean())
    return summary


def build_summaries(nearest: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_rows = []
    group_columns = ["split_index", "seed", "query_set", "reference_set"]
    for keys, group in nearest.groupby(group_columns, sort=True):
        split_index, seed, query_set, reference_set = keys
        summary_rows.append(
            {
                "scope": "per_split",
                "split_index": split_index,
                "seed": seed,
                "query_set": query_set,
                "reference_set": reference_set,
                **similarity_summary(group["max_tanimoto_similarity"]),
            }
        )
    for keys, group in nearest.groupby(["query_set", "reference_set"], sort=True):
        query_set, reference_set = keys
        summary_rows.append(
            {
                "scope": "all_molecules_pooled",
                "split_index": "",
                "seed": "",
                "query_set": query_set,
                "reference_set": reference_set,
                **similarity_summary(group["max_tanimoto_similarity"]),
            }
        )
    summary = pd.DataFrame(summary_rows)

    across_rows = []
    per_split = summary.loc[summary["scope"].eq("per_split")]
    for keys, group in per_split.groupby(["query_set", "reference_set"], sort=True):
        query_set, reference_set = keys
        for statistic in STATISTICS:
            values = group[statistic].astype(float)
            across_rows.append(
                {
                    "query_set": query_set,
                    "reference_set": reference_set,
                    "statistic": statistic,
                    "n_splits": int(len(values)),
                    "mean_across_splits": float(values.mean()),
                    "std_across_splits": float(values.std(ddof=1)),
                    "minimum_across_splits": float(values.min()),
                    "maximum_across_splits": float(values.max()),
                }
            )
    return summary, pd.DataFrame(across_rows)


def pooled_table_value(summary: pd.DataFrame, query_set: str, reference_set: str, statistic: str, percent: bool = False) -> str:
    row = summary.loc[
        (summary["scope"].eq("all_molecules_pooled"))
        & (summary["query_set"].eq(query_set))
        & (summary["reference_set"].eq(reference_set))
    ].iloc[0]
    scale = 100.0 if percent else 1.0
    suffix = r"\%" if percent else ""
    return f"{float(row[statistic]) * scale:.3f}{suffix}"


def main() -> None:
    TABLES.mkdir(parents=True, exist_ok=True)
    protocol = validate_split_protocol()
    structures, registry = load_inputs()
    fingerprints = build_fingerprints(structures)
    nearest = calculate_nearest_neighbours(registry, fingerprints)
    _, across = build_summaries(nearest)
    across_path = TABLES / "nearest_neighbor_summary_across_splits.csv"
    across.to_csv(across_path, index=False)
    print(f"Prepared fingerprints for {len(fingerprints):,} molecules.")
    print(f"Calculated {len(nearest):,} query/reference comparisons in memory.")
    print(f"Wrote: {across_path.relative_to(ROOT)}")
    test_train = across.loc[
        across["query_set"].eq("test") & across["reference_set"].eq("train")
    ].set_index("statistic")
    for statistic in ("median", "p95", "p99", "maximum", "fraction_ge_0.80", "fraction_ge_0.90"):
        row = test_train.loc[statistic]
        print(f"test_to_train {statistic}: {row['mean_across_splits']:.9f} ± {row['std_across_splits']:.9f}")


if __name__ == "__main__":
    main()
