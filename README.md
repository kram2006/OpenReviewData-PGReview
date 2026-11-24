# Peer Review GNN Artifact

Supplementary artifact for a VLDB 2027 submission on peer-review outcome modeling with heterogeneous graph neural networks.

- Paper: "Reliability of graph based consolidation of reviews"
- Venue: VLDB 2027 / PVLDB Volume 20
- Repository: https://github.com/kram2006/OpenReviewData-PGReview

VLDB 2027 uses single-blind review, so this repository is intentionally public and does not need anonymization.

## Artifact Scope

This repository supports the experiments that build a heterogeneous graph from review text and reviewer/paper interactions, embed review summaries with pretrained language models, train a two-layer heterogeneous GNN, and predict mean paper ratings.

The current entry point runs all three included ICLR datasets with three language models:

- `allenai/scibert_scivocab_uncased`
- `KISTI-AI/Scideberta-full`
- `KISTI-AI/scideberta-cs`

All three CSV files are used by the default run. The dataset list is the `csv` list near the top of `rk_gnn.py`.

This repository contains the source code and open datasets only. The problem statement, experimental results, tables, and figures are reported in the main paper; generated outputs are not stored in Git.

## Repository Layout

| Path | Purpose |
|---|---|
| `rk_gnn.py` | Main end-to-end preprocessing, embedding, GNN training, evaluation, and output script |
| `requirements.txt` | Fully pinned Python environment, including CUDA 13 PyTorch wheel |
| `myrun.sh` | Fresh-environment setup and run script for Linux/macOS |
| `run.ps1` | Fresh-environment setup and run script for Windows PowerShell |
| `iclr_reviews_24.csv` | ICLR 2024 review data |
| `iclr_reviews_25.csv` | ICLR 2025 review data |
| `iclr_reviews_26.csv` | ICLR 2026 review data used by the default experiment |
| `REPRODUCIBILITY.md` | Exact environment, commands, generated outputs, and paper-result mapping |
| `SUPPLEMENTARY.md` | Artifact inventory and supplementary-material index |
| `DATA_SOURCES.md` | Dataset provenance, schemas, checksums, and licensing notes |
| `CITATION.cff` | Citation metadata for GitHub and archival services |
| `.gitignore` | Excludes virtual environments, caches, embeddings, and generated model outputs |

## Software Requirements

- Python 3.12.3
- PyTorch 2.13.0 with CUDA 13.0 wheel on NVIDIA systems
- `torch-geometric==2.8.0.post1`
- `transformers==5.16.1`
- `pandas==3.0.5`, `numpy==2.5.2`, `scikit-learn==1.9.0`
- NVIDIA driver compatible with CUDA 13
- Internet access on first run to download Hugging Face checkpoints

## Hardware Requirements

| Tier | GPU | System RAM | Storage | Notes |
|---|---|---:|---:|---|
| Minimum | NVIDIA GPU with at least 24 GB VRAM | 32 GB | 30 GB free SSD | Needed for the full ICLR 2026 run |
| Recommended | NVIDIA GPU with 48 GB or more VRAM | 64 GB or more | 50 GB or more SSD | Comfortable full three-dataset run |
| CPU only | Not recommended for full runs | 128 GB or more | Same as above | Use only for smoke tests; full runtime is impractical |

The current implementation creates a reviewer-node identity feature matrix. Its raw size is about 2.92 GiB for ICLR 2024, 8.54 GiB for ICLR 2025, and 21.29 GiB for ICLR 2026 before activations, gradients, and optimizer states.

The experiments reported in the paper were run on a cloud GPU with 48 GB VRAM; the exact GPU model was not recorded in the artifact.

## Quickstart: Windows PowerShell

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python rk_gnn.py
```

Alternatively:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

## Quickstart: Linux/macOS

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python rk_gnn.py
```

Alternatively:

```bash
bash myrun.sh
```

## Method Summary

1. Load review records from the CSV file.
2. Keep the required columns and drop rows with missing or nonnumeric IDs and ratings.
3. Assign integer indices to papers, reviewers, and reviews.
4. Embed each review summary using mean pooling over pretrained language-model hidden states.
5. Build a heterogeneous graph with `paper` and `reviewer` nodes and review edges carrying text embeddings plus the rating feature.
6. Train a heterogeneous GNN to predict the mean rating of each paper.
7. Evaluate on a deterministic 80/20 paper-level split and print/write generated outputs on the reproduction machine.

## Main Configuration

| Setting | Value |
|---|---:|
| `MAX_LENGTH` | 256 |
| `BATCH_SIZE` | 8 |
| `EPOCHS` | 100 |
| `LEARNING_RATE` | 0.001 |
| `HIDDEN_DIM` | 128 |
| `DROPOUT` | 0.3 |
| `RANDOM_SEED` | 42 |
| Train/test split | 80/20 by paper |
| Target | Mean review rating per paper |
| Loss | MSE |
| Metrics | MSE, MAE, R², RMSE |

## Generated Outputs

| File | Meaning | Git status |
|---|---|---|
| `iclr_reviews_<year>_<model>_embeddings.dat` | Raw review-summary embeddings | Ignored; regenerate with `rk_gnn.py` |
| `iclr_reviews_<year>_<model>_gnn_paper_embeddings.npy` | Learned paper embeddings | Ignored; regenerate with `rk_gnn.py` |
| `iclr_reviews_<year>_<model>__gnn_predictions.csv` | Actual and predicted ratings per paper | Ignored; regenerate with `rk_gnn.py` |

`<year>` is `24`, `25`, or `26`. All generated outputs are excluded from Git. The reported results are in the paper; the generated files are only for reproduction checks. The 2026 SciBERT embedding file is about 233 MB and exceeds GitHub's recommended file-size limit.

## Reproduction

See `REPRODUCIBILITY.md` for the fresh-machine recipe, exact data checksums, generated-output mapping, and known limitations.

## Data and Licensing

See `DATA_SOURCES.md` for dataset provenance, schemas, counts, checksums, and licensing. No code license file is included in this release.

## Citation

See `CITATION.cff` for citation metadata.

## Limitations

- A full run processes three datasets and three language models, so runtime scales with dataset size.
- There is no validation set or early stopping.
- CUDA kernels may produce small numerical differences across hardware.
- Hugging Face model availability is required unless checkpoints are pre-cached.
- The current code does not deduplicate by `review_id`; see `DATA_SOURCES.md` for the audited duplicate counts.
