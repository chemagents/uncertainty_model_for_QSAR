#!/usr/bin/env python3
"""Assemble and verify the article release from its two computed sources."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FULL_DOMAIN_SUMMARY = ROOT / "results" / "table1_full_domain" / "tables" / "summary.csv"
REGISTERED_SPLIT_SUMMARY = ROOT / "results" / "tables" / "model_comparison_summary.csv"
ARTICLE_SOURCE = ROOT / "article" / "manuscript" / "article.tex"
ARTICLE_OUTPUT = ROOT / "article" / "manuscript"
VERIFICATION = ROOT / "article" / "verification"
FULL_DOMAIN_SUMMARY_SHA256 = "4c08a0cd63dc6c6c8327f3d84b89f918cc4c30e208ac6d0ec716111d8c451231"
REGISTERED_SPLIT_SUMMARY_SHA256 = "dcbbb321dd38799dcfcddcd94c4499b68861168d84c09e5a2ca70d20148bc518"
METRICS = ("RMSE", "MAE", "MSE", "R2", "Spearman", "CCC", "CI_0.8_size", "CI_0.9_size", "CI_0.95_size")
REGISTERED_SPLIT_METRICS = {
    "RMSE": "RMSE_mean", "MAE": "MAE_mean", "MSE": "MSE_mean", "R2": "R2_mean",
    "Spearman": "Spearman_mean", "CCC": "CCC_mean", "CI_0.8_size": "CI_0.80_size_mean",
    "CI_0.9_size": "CI_0.90_size_mean", "CI_0.95_size": "CI_0.95_size_mean",
}
FIGURE_HASHES = {
    "fig1.png": "12d67821d1349671e976d06e5f44028b55d77ed17fdd06e08d8e5537bb3a5589",
    "fig2.png": "09ad3e83828a2e6331dca3273a042a97106c85360bae7f43ab27dbff1958ea87",
    "fig3.png": "33ebac5f38c1b33a773d97c45a44ea7cd6c5d3ac92984d863ecab08151c8f645",
    "fig4a.png": "fc5fe8072df551322bfbda7d59c63ac0d5b017bfb6f43ea8a4136675c7410cee",
    "fig4b.png": "7c45cccdcaa12ab8e6a961a3be0eb7b9febaef710bc5a4fde7bf1371e46ec147",
    "fig4c.png": "61f2470f64e354cd8f0b100555c56bd8209616802f0902814c95cae54e5de988",
    "fig5.png": "cf5b622411bfbfebeae8c8a529c42becdf5ddb27157cc20feb7bb64ca79def7f",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_hash(path: Path, expected: str) -> None:
    actual = sha256(path)
    if actual != expected:
        raise ValueError(f"{path}: expected SHA-256 {expected}, got {actual}")


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def one_row(rows: list[dict[str, str]], **conditions: str) -> dict[str, str]:
    selected = [row for row in rows if all(row[key] == value for key, value in conditions.items())]
    if len(selected) != 1:
        raise ValueError(f"Expected one source row matching {conditions}, got {len(selected)}")
    return selected[0]


def formatted_row(label: str, row: dict[str, str], fields: dict[str, str]) -> dict[str, str]:
    return {"model": label, **{metric: f"{float(row[fields[metric]]):.3f}" for metric in METRICS}}


def current_table() -> list[dict[str, str]]:
    full_domain_rows = csv_rows(FULL_DOMAIN_SUMMARY)
    registered_split_rows = csv_rows(REGISTERED_SPLIT_SUMMARY)
    full_domain_fields = {metric: metric for metric in METRICS}
    rows = [
        formatted_row(label, one_row(full_domain_rows, model=model), full_domain_fields)
        for label, model in (("Baseline", "Baseline"), ("PCA", "PCA"), ("ADME", "ADME"), ("Plain", "Plain"))
    ]
    rows.extend((
        formatted_row("Plain (full train)", one_row(registered_split_rows, model="Plain", evaluation_domain="bbb_pass_test"), REGISTERED_SPLIT_METRICS),
        formatted_row("BBB pass (BBB train)", one_row(registered_split_rows, model="BBB pass", evaluation_domain="bbb_pass_test"), REGISTERED_SPLIT_METRICS),
    ))
    return rows


def verify_table_in_tex(rows: list[dict[str, str]]) -> None:
    source = ARTICLE_SOURCE.read_text()
    for row in rows:
        values = r"\s*&\s*".join(re.escape(row[metric]) for metric in METRICS)
        pattern = re.escape(row["model"]) + r"\s*&\s*" + values + r"\s*\\\\"
        if not re.search(pattern, source):
            raise ValueError(f"article.tex does not contain the verified Table 1 row for {row['model']}")


def stage_article_figures() -> dict[str, dict[str, object]]:
    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError("Pillow is required to verify PNG dimensions") from error
    figures = {}
    for name, expected_hash in FIGURE_HASHES.items():
        source = ROOT / "manuscript" / name
        destination = ARTICLE_OUTPUT / name
        require_hash(source, expected_hash)
        shutil.copyfile(source, destination)
        require_hash(destination, expected_hash)
        with Image.open(destination) as image:
            figures[name] = {"sha256": expected_hash, "size_px": list(image.size), "mode": image.mode}
    return figures


def main() -> None:
    require_hash(FULL_DOMAIN_SUMMARY, FULL_DOMAIN_SUMMARY_SHA256)
    require_hash(REGISTERED_SPLIT_SUMMARY, REGISTERED_SPLIT_SUMMARY_SHA256)
    rows = current_table()
    verify_table_in_tex(rows)
    figures = stage_article_figures()
    VERIFICATION.mkdir(parents=True, exist_ok=True)
    with (VERIFICATION / "table1_current_article.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("model", *METRICS))
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "status": "current two-component article assembled and verified",
        "reference_protocol_commit": "2c24f1905d861b34bfff3481fc4b10514b0e4371",
        "full_domain_table_rows": "fixed full-domain Table 1 component using fixed Butina assignments",
        "bbb_table_rows": "30 registered-split BBB component",
        "figure_sources": "notebooks 03-06",
        "article_pngs": "fresh h notebook renders copied directly into the article build directory",
        "fresh_figure_outputs": figures,
        "table_rows": rows,
    }
    (VERIFICATION / "provenance_report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(f"Verified current article release in {ARTICLE_OUTPUT}")


if __name__ == "__main__":
    main()
