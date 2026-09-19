#!/usr/bin/env python3
"""Run every computational stage needed for the current article release."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import nbformat
from jupyter_client import AsyncKernelManager
from nbclient import NotebookClient


ROOT = Path(__file__).resolve().parent
H_NOTEBOOKS_BEFORE_COMPARISON = ("01_data_inventory.ipynb", "02_model_training.ipynb")
H_COMPARISON_NOTEBOOK = "03_fig1ab_fig2_table1.ipynb"
H_NOTEBOOKS_AFTER_COMPARISON = (
    "04_shap_fig3_fig4.ipynb",
    "05_fig5_conformal_landscape.ipynb",
    "06_manuscript_tables.ipynb",
)
OUTPUT_DIRECTORIES = (
    ROOT / "results",
    ROOT / "manuscript" / "tables",
    ROOT / "article" / "verification",
)
H_FIGURES = tuple(ROOT / "manuscript" / name for name in ("fig1.png", "fig2.png", "fig3.png", "fig4a.png", "fig4b.png", "fig4c.png", "fig5.png"))
SOURCE_HASHES = {
    ROOT / "data" / "df_final.csv": "296a7b89cefeb2a925aeaaf9e8b6eeaf273199c889252930899f73c85c81dbb4",
    ROOT / "data" / "butina_clusters.csv": "04e585aaec9b66b404b71a34748742fc6e6281d1b2ac692e7e9c4127ff0d3973",
    ROOT / "data" / "excluded_molecules.csv": "637a75427e5496aa12a7eb307e2ce1b4cbf25c05d8be1f2fe1fa9d446ccafa9a",
    ROOT / "data" / "receptor_panel_annotation.csv": "efec450885ece0c271f47555233af1bf44de06ffda2710e43fe2c021d86ca54b",
    ROOT / "data" / "table1_full_domain_splits" / "split_indices.csv": "102be182ec759218534d42b5e9f3c81dce95664bf05d1bff785b6c41e8672dc4",
}


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def validate_source_inputs() -> None:
    for path, expected in SOURCE_HASHES.items():
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Input hash mismatch for {path}: expected {expected}, got {actual}")


def execute_notebook(path: Path) -> dict[str, object]:
    notebook = nbformat.read(path, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
            cell.metadata.pop("execution", None)
    kernel = AsyncKernelManager(kernel_name="python3")
    kernel.kernel_spec.argv = [sys.executable, "-m", "ipykernel_launcher", "-f", "{connection_file}"]
    client = NotebookClient(
        notebook,
        km=kernel,
        timeout=7200,
        allow_errors=False,
        resources={"metadata": {"path": str(path.parent)}},
    )
    started = time.time()
    client.execute(cwd=str(path.parent), cleanup_kc=True)
    nbformat.write(notebook, path)
    return {"notebook": str(path.relative_to(ROOT)), "seconds": round(time.time() - started, 3)}


def execute_script(path: Path, *args: str) -> dict[str, object]:
    started = time.time()
    subprocess.run([sys.executable, str(path), *args], cwd=ROOT, check=True)
    return {"script": str(path.relative_to(ROOT)), "seconds": round(time.time() - started, 3)}


def clear_outputs() -> None:
    for directory in OUTPUT_DIRECTORIES:
        if directory.exists():
            shutil.rmtree(directory)
    for path in H_FIGURES:
        path.unlink(missing_ok=True)
    for suffix in ("aux", "bbl", "blg", "log", "out", "pdf"):
        (ROOT / "article" / "manuscript" / f"article.{suffix}").unlink(missing_ok=True)
    for name in ("fig1.png", "fig2.png", "fig3.png", "fig4a.png", "fig4b.png", "fig4c.png", "fig5.png"):
        (ROOT / "article" / "manuscript" / name).unlink(missing_ok=True)


def assert_clean_or_requested(clean: bool, resume: bool) -> None:
    if clean and resume:
        raise RuntimeError("Use either --clean for a new rebuild or --resume for an interrupted rebuild, not both.")
    existing = [path for path in OUTPUT_DIRECTORIES if path.exists()]
    existing.extend(path for path in H_FIGURES if path.exists())
    if existing and not clean and not resume:
        names = ", ".join(str(path.relative_to(ROOT)) for path in existing)
        raise RuntimeError(f"Generated outputs already exist: {names}. Use --clean for a new rebuild or --resume after interruption.")
    if clean:
        clear_outputs()


def build_article_pdf() -> None:
    manuscript = ROOT / "article" / "manuscript"
    for command in (
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "article.tex"],
        ["bibtex", "article"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "article.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "article.tex"],
    ):
        subprocess.run(command, cwd=manuscript, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clean", action="store_true", help="Remove known generated outputs before rebuilding.")
    parser.add_argument("--resume", action="store_true", help="Continue an interrupted rebuild using saved 30-split models.")
    parser.add_argument("--skip-pdf", action="store_true", help="Do not build article/manuscript/article.pdf.")
    parser.add_argument("--skip-article", action="store_true", help="Do not run article assembly or build the PDF.")
    args = parser.parse_args()

    validate_source_inputs()
    assert_clean_or_requested(args.clean, args.resume)
    execution_path = ROOT / "results" / "run_manifest.json"
    if args.resume and execution_path.is_file():
        execution = json.loads(execution_path.read_text())
    else:
        execution = []
    completed = {record.get("notebook") or record.get("script") for record in execution}

    def run_notebook_once(path: Path) -> None:
        key = str(path.relative_to(ROOT))
        if key in completed:
            return
        execution.append(execute_notebook(path))
        completed.add(key)
        write_json(execution_path, execution)

    def run_script_once(path: Path, *script_args: str) -> None:
        key = str(path.relative_to(ROOT))
        if key in completed:
            return
        execution.append(execute_script(path, *script_args))
        completed.add(key)
        write_json(execution_path, execution)

    if not args.resume:
        for name in H_NOTEBOOKS_BEFORE_COMPARISON:
            run_notebook_once(ROOT / "notebooks" / name)
    elif not (ROOT / "results" / "splits" / "split_indices.csv").is_file():
        raise RuntimeError("--resume requires results/splits/split_indices.csv from notebooks 01-02.")
    run_script_once(ROOT / "scripts" / "train_model_splits.py", "--resume")
    run_notebook_once(ROOT / "notebooks" / H_COMPARISON_NOTEBOOK)

    for name in H_NOTEBOOKS_AFTER_COMPARISON:
        run_notebook_once(ROOT / "notebooks" / name)
        if name == "05_fig5_conformal_landscape.ipynb":
            run_script_once(ROOT / "scripts" / "audit_comment8_reproducibility.py")

    run_script_once(ROOT / "scripts" / "train_table1_full_domain.py")

    if not args.skip_article:
        run_notebook_once(ROOT / "notebooks" / "07_assemble_article.ipynb")
    if not args.skip_pdf and not args.skip_article:
        build_article_pdf()
    print("Current article reproduction completed." if not args.skip_article else "Analysis reproduction completed without article assembly.")


if __name__ == "__main__":
    main()
