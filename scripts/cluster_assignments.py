"""Load Butina assignments by ligand ID rather than from the raw source table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


LIGAND_ID_COL = "ligand_id"
CLUSTER_COL = "Butina_clusters"


def attach_cluster_assignments(
    df: pd.DataFrame,
    assignments_path: Path,
    *,
    allow_missing: bool = False,
) -> pd.DataFrame:
    """Return a row-order-preserving frame with assignments joined by ligand ID."""
    assignments = pd.read_csv(assignments_path, usecols=[LIGAND_ID_COL, CLUSTER_COL])
    if assignments[LIGAND_ID_COL].duplicated().any():
        raise ValueError(f"Duplicate ligand IDs in {assignments_path}")

    source = df.drop(columns=[CLUSTER_COL], errors="ignore").copy()
    source[LIGAND_ID_COL] = source[LIGAND_ID_COL].astype(int)
    assignments[LIGAND_ID_COL] = assignments[LIGAND_ID_COL].astype(int)
    joined = source.merge(assignments, on=LIGAND_ID_COL, how="left", sort=False, validate="one_to_one")
    if len(joined) != len(source):
        raise AssertionError("Cluster assignment join changed the source row count")
    if not allow_missing and joined[CLUSTER_COL].isna().any():
        missing = joined.loc[joined[CLUSTER_COL].isna(), LIGAND_ID_COL].head().tolist()
        raise ValueError(f"Missing Butina assignments from {assignments_path}: {missing}")
    return joined
