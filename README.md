# A biology-informed uncertainty model for toxicological QSAR

This repository reproduces the tables and figures for a biology-informed
acute-toxicity quantitative structure-activity relationship (QSAR) workflow.
It combines molecular fingerprints, physicochemical descriptors, docking scores
for 44 safety-relevant proteins, ADMET descriptors, CatBoost regression, SHAP
interpretation, and local conformal prediction.

The repository is self-contained for the analysis workflow: it does not read
`reporev2g1`, `reporev2h`, or an external article directory. The private LaTeX
article workspace is intentionally excluded from this public repository.

## Project Structure

```text
├── data/
│   ├── df_final.csv                      # Canonical molecular table
│   ├── excluded_molecules.csv            # Exclusion registry
│   ├── receptor_panel_annotation.csv     # Protein-panel annotation
│   ├── butina_clusters.csv               # Fixed full-domain cluster assignment
│   └── table1_full_domain_splits/        # Fixed full-domain split registry
├── manuscript/
│   ├── fig1.png ... fig5.png             # Recomputed scientific figures
│   └── tables/                           # Generated LaTeX table fragments and macros
├── notebooks/                            # Numbered reproducible analysis workflow
├── results/
│   ├── figures/                          # Generated figure data and diagnostics
│   ├── splits/                           # Generated cluster and split assignments
│   ├── table1_full_domain/               # Fixed full-domain Table 1 component
│   ├── tables/                           # Metrics, predictions, and comparison tables
│   └── run_manifest.json                 # Completed-stage timing manifest
├── scripts/                              # Training, audit, table, and assembly scripts
├── reproduce_article.py                  # End-to-end reproduction runner
└── requirements.txt
```

Generated figures, split registries, tables, and verification-friendly result
files are versioned as release artifacts. Reproducible serialized model caches
and the private `article/` workspace are excluded from version control.

## Input Features

- **Molecular structure:** 2,048 stored Morgan fingerprint bits.
- **Physicochemical descriptors:** molecular weight (MW) and lipophilicity
  (logP).
- **Protein interaction descriptors:** docking scores for 44 safety-relevant
  protein targets.
- **ADME descriptors:** a descriptor block used by the ADME model.
- **Biological context coordinates:** docking-burden and predicted toxicity
  phenotype coordinates used for local conformal calibration.

The prediction target is `-lgLD50, mol/kg`: larger values correspond to higher
acute toxicity.

## Models

| Model | Feature space |
| --- | --- |
| Baseline | Molecular structure and physicochemical descriptors plus 44 docking scores |
| PCA | Molecular structure and physicochemical descriptors plus three principal components of the docking panel |
| ADME | Baseline features plus the ADME descriptor block |
| Plain | Molecular structure and physicochemical descriptors only |
| BBB pass | Baseline model evaluated on the BBB-permeant domain |

The point predictor used for the local confidence analysis is deliberately the
structural Plain model. Docking and toxicity-phenotype coordinates are used to
calibrate local conformal prediction intervals rather than being required by the
point predictor.

## Split Registries

The workflow has two explicitly distinct evaluation components.

### Fixed Full-Domain Table 1

The submitted Table 1 rows use the fixed full-domain cluster assignment in
`data/butina_clusters.csv` and the corresponding fixed split registry in
`data/table1_full_domain_splits/split_indices.csv`. These files are retained so
that the submitted values can be reproduced exactly from `data/df_final.csv`.

### 30 Registered Splits

For robustness analyses, notebook 01 regenerates Butina assignments from the
canonical molecular table. The resulting 30 split registry is written under:

```text
results/splits/butina/
```

The 30-split mean and standard deviation describe a different evaluation
component from the fixed full-domain Table 1 rows. In particular, BBB-restricted
rows are evaluated only in the BBB-permeant test domain and are not pooled with
full-domain results.

## Installation

The workflow was verified on Linux with Python 3.14.5.

```bash
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
```

Building the PDF additionally requires a TeX installation that provides
`pdflatex` and `bibtex`.

## Reproduce The Article

Run the public analysis workflow from the repository root:

```bash
.venv/bin/python reproduce_article.py --skip-article
```

The runner validates source-file hashes, executes notebooks and scripts in the
registered order, records completed stages in `results/run_manifest.json`,
and verifies the analysis figures and Table 1 inputs.

The private article workspace enables the final assembly stage and PDF build.
When it is available, run the full workflow without `--skip-article`.

Start a deliberate clean rebuild of generated outputs with:

```bash
.venv/bin/python reproduce_article.py --clean --skip-article
```

`--clean` removes only runtime output paths; it does not remove source data,
notebooks, or scripts. Because release artifacts under `results/` and
`manuscript/` are versioned, use it only when deliberately rebuilding them.

Resume an interrupted long run without repeating notebooks 01 and 02:

```bash
.venv/bin/python reproduce_article.py --resume --skip-article
```

To verify all computational stages without building the PDF, add `--skip-pdf`.

## Workflow

The end-to-end runner executes these stages in order.

1. **Data inventory**: `notebooks/01_data_inventory.ipynb` validates
   `data/df_final.csv`, records feature groups, and creates the registered
   Butina split assignments.
2. **Model preparation**: `notebooks/02_model_training.ipynb` prepares the
   five model feature spaces and split inputs.
3. **30-split model training**: `scripts/train_model_splits.py` trains and
   evaluates the registered-split CatBoost models.
4. **Performance figures and Table 1 source**:
   `notebooks/03_fig1ab_fig2_table1.ipynb` creates the primary performance
   figures and comparison table source.
5. **SHAP interpretation**: `notebooks/04_shap_fig3_fig4.ipynb` computes
   feature-attribution analyses and figures.
6. **Local confidence landscape**:
   `notebooks/05_fig5_conformal_landscape.ipynb` fits the local conformal
   uncertainty analysis and creates the confidence figure.
7. **Reproducibility audit**:
   `scripts/audit_comment8_reproducibility.py` checks the registered-split
   artifacts and robustness summary.
8. **Manuscript tables**: `notebooks/06_manuscript_tables.ipynb` writes LaTeX
   table fragments and numerical macros to `manuscript/tables/`.
9. **Fixed full-domain Table 1**:
   `scripts/train_table1_full_domain.py` reproduces the fixed reference
   component against the canonical molecular table.
10. **Article assembly**: the private `notebooks/07_assemble_article.ipynb`
    verifies Table 1 and figure hashes, stages figures for LaTeX, and writes
    verification records to `article/verification/`. Public reproductions use
    `--skip-article`.

## Results

### Fixed Full-Domain Table 1

The following values are from the fixed full-domain component used for the
article Table 1. Prediction intervals (PI) are 90% conformal interval widths.

| Model | RMSE | MAE | R2 | Spearman | CCC | 90% PI |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 0.520 | 0.359 | 0.486 | 0.697 | 0.659 | 1.625 |
| PCA | 0.517 | 0.360 | 0.492 | 0.700 | 0.661 | 1.651 |
| ADME | 0.509 | 0.349 | 0.507 | 0.714 | 0.662 | 1.573 |
| Plain | 0.507 | 0.353 | 0.510 | 0.714 | 0.670 | 1.618 |
| BBB pass | 0.479 | 0.328 | 0.433 | 0.668 | 0.607 | 1.465 |

`BBB pass` is evaluated on the filtered BBB-permeant domain. It is therefore
not a universal replacement for the full-domain models.

### Registered-Split Robustness

Across 30 registered group-holdout splits, the full-domain Plain model has mean
RMSE `0.514 +/- 0.017`, mean Spearman correlation `0.701 +/- 0.014`, and mean
90% interval width `1.563 +/- 0.060`. The full result table, including all
models, interval coverage, split sizes, and standard deviations, is written to:

```text
results/tables/model_comparison_summary.csv
```

### Local Confidence Model

The local conformal analysis operates on the structural Plain predictor and
uses biological-context coordinates to vary interval width between local regions
of ligand space. It distinguishes regions where prediction errors are expected
to be smaller from regions where they are expected to be larger, instead of
reporting one identical uncertainty interval for every molecule.

## Key Findings

- Molecular fingerprints with MW and logP provide a strong global acute-toxicity
  prediction baseline.
- Adding docking scores or ADME descriptors does not consistently improve global
  point-prediction accuracy over the structural model.
- Docking-derived interaction descriptors and toxicity-phenotype coordinates
  remain useful as a biological context space for local uncertainty estimation.
- The BBB-restricted evaluation is a domain-specific analysis and must be
  interpreted separately from full-domain evaluation.

## Verification Outputs

After a public analysis run, inspect the versioned result tables and figures
under `results/` and `manuscript/`. With the private article workspace present,
the final assembly additionally writes:

```text
article/verification/table1_current_article.csv
article/verification/provenance_report.json
article/manuscript/fig1.png ... fig5.png
article/manuscript/article.pdf
```

The assembly stage verifies source tables and generated figures with SHA-256
hashes before copying the figures to the article build directory. The original
rasterization command is unavailable, so newly rendered PNG files preserve the
computed data and labels but are not claimed to be pixel-identical to an earlier
article export.

## Configuration

The analysis is a sequential notebook workflow. Feature definitions, model
parameters, split settings, and plotting choices are defined in the relevant
notebooks and scripts. All runtime paths are relative to the repository root.

## Citation

Citation details will be added with the final article release.

## Contact

For questions or issues, contact `safronov@phystech.edu`.
