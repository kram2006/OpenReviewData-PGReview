# Reproducibility

Paper: "Reliability of graph based consolidation of reviews", VLDB 2027 / PVLDB Volume 20.

The main paper is the authoritative source for the problem statement, results, tables, figures, and analysis. This document only explains how to execute the released code and open datasets.

## Environment

| Component | Version |
|---|---|
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| Paper hardware | Cloud GPU with 48 GB VRAM; exact model not recorded in the artifact |
| torch-geometric | 2.8.0.post1 |
| transformers | 5.16.1 |
| tokenizers | 0.23.1 |
| huggingface-hub | 1.28.0 |
| numpy | 2.5.2 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |

The complete pinned dependency set is in `requirements.txt`. It includes `--extra-index-url https://download.pytorch.org/whl/cu130` so that `torch==2.13.0+cu130` resolves correctly.

## Hardware Requirements

| Tier | GPU | System RAM | Storage | Notes |
|---|---|---:|---:|---|
| Minimum | NVIDIA GPU with at least 24 GB VRAM | 32 GB | 30 GB free SSD | Needed for the full ICLR 2026 run |
| Recommended | NVIDIA GPU with 48 GB or more VRAM | 64 GB or more | 50 GB or more SSD | Comfortable full three-dataset run |
| CPU only | Not recommended for full runs | 128 GB or more | Same as above | Use only for smoke tests |

Raw reviewer identity feature memory by dataset:

| Dataset | Reviewers | Approximate identity-matrix size |
|---|---:|---:|
| `iclr_reviews_24.csv` | 27,992 | 2.92 GiB |
| `iclr_reviews_25.csv` | 47,867 | 8.54 GiB |
| `iclr_reviews_26.csv` | 75,604 | 21.29 GiB |

Activations, gradients, optimizer states, embeddings, and model parameters add more memory. The reported experiments used a cloud GPU with 48 GB VRAM; the exact GPU model was not recorded in the artifact.

## Fresh-Machine Setup

Windows PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Linux/macOS:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

Expected: `2.13.0+cu130 True` on a compatible NVIDIA GPU. `False` indicates CPU fallback.

## Data Files

| File | Size | Rows | Papers | Reviewers | Reviews | SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| `iclr_reviews_24.csv` | 20,634,412 bytes | 28,067 | 7,262 | 27,992 | 28,028 | `90CE3278D822041188A549FECDF2A923E0BDC29AB28DF2092B33E06180FA9BDE` |
| `iclr_reviews_25.csv` | 35,671,760 bytes | 47,984 | 11,837 | 47,867 | 47,984 | `CAB6A4049D2159A5F461BD881C2F67305527BEB3FF7D1A02CC533668FC81F31E` |
| `iclr_reviews_26.csv` | 61,575,115 bytes | 78,951 | 19,475 | 75,604 | 75,860 | `2704D14598F09D5D107DD61B98E047C1E483C67E74A9776E2F00808FB74E284A` |

The default experiment uses all three included CSV files.

Required columns:

```text
submission_id, title, reviewer_id, review_id, summary,
soundness, presentation, contribution, rating, confidence
```

Cleaning rules:

- Keep only the required columns.
- Drop rows missing `submission_id`, `reviewer_id`, `review_id`, `soundness`, `presentation`, `contribution`, or `confidence`.
- Convert `rating` to numeric and drop rows with missing rating; this also removes the literal-column-name rows in `iclr_reviews_26.csv`.
- Do not deduplicate by `review_id`; see `DATA_SOURCES.md` for audited duplicate counts.
- Reset indices.

## Exact Experiment Configuration

| Setting | Value |
|---|---:|
| Datasets | `iclr_reviews_24.csv`, `iclr_reviews_25.csv`, `iclr_reviews_26.csv` |
| Language models | SciBERT, Scideberta-full, Scideberta-cs |
| `MAX_LENGTH` | 256 |
| `BATCH_SIZE` | 8 |
| `EPOCHS` | 100 |
| `LEARNING_RATE` | 0.001 |
| `HIDDEN_DIM` | 128 |
| `DROPOUT` | 0.3 |
| `RANDOM_SEED` | 42 |
| GNN layers | 2 heterogeneous SAGEConv layers with mean aggregation |
| Optimizer | Adam |
| Loss | MSE |
| Split | 80/20 paper-level random permutation |
| Target | Mean rating per paper |

Random seeds are set for Python, NumPy, CPU PyTorch, and CUDA PyTorch. The script does not enable fully deterministic CUDA kernels.

## Run

```powershell
python rk_gnn.py
```

or:

```bash
python rk_gnn.py
```

The script first loads or generates one embedding file per year and language model:

```text
iclr_reviews_<year>_<model>_embeddings.dat
```

`<year>` is `24`, `25`, or `26`, and `<model>` is `scibert`, `scidebertafull`, or `scidebertacs`. Each file is a `float32` memory-mapped array with shape `(number_of_clean_reviews, 768)`.

It then writes:

```text
iclr_reviews_<year>_<model>_gnn_paper_embeddings.npy
iclr_reviews_<year>_<model>__gnn_predictions.csv
```

and prints MSE, MAE, R², and RMSE for the test split.

## Claim-to-Command Map

| Paper item | Command | Generated output | Comparison |
|---|---|---|---|
| Main result table, ICLR 2024 | `python rk_gnn.py` | Console metrics and `iclr_reviews_24_<model>__gnn_predictions.csv` | Values reported in the paper |
| Main result table, ICLR 2025 | `python rk_gnn.py` | Console metrics and `iclr_reviews_25_<model>__gnn_predictions.csv` | Values reported in the paper |
| Main result table, ICLR 2026 | `python rk_gnn.py` | Console metrics and `iclr_reviews_26_<model>__gnn_predictions.csv` | Values reported in the paper |
| Paper embedding analysis | `python rk_gnn.py` | Generated `*_gnn_paper_embeddings.npy` | Analysis reported in the paper |

## Verifying a Fresh Clone

1. Confirm Python 3.12.3.
2. Install from pinned `requirements.txt`.
3. Confirm `torch.cuda.is_available()` is `True` if GPU reproduction is intended.
4. Run the exact command above.
5. Compare the printed metrics with the values reported in the main paper.
6. Treat small deviations as hardware-dependent.

## Generated Files Not Committed

The following are intentionally excluded by `.gitignore`:

- virtual environments and all Python caches
- `*_embeddings.dat` and other embedding arrays
- `*_gnn_paper_embeddings.npy` and other generated tensors
- `*__gnn_predictions.csv` and other generated prediction files
- model checkpoints and logs

The main paper contains the reported results; generated outputs are not committed.

## Known Limitations

- The full run is long because it processes three datasets and three language models.
- No separate validation split or early stopping.
- The train/test split is generated at runtime from the seed.
- The script evaluates mean paper rating regression, not individual review rating classification.
- Hugging Face checkpoints are downloaded at runtime.
- GPU numerical differences can affect late decimal places.
- Dataset terms and redistribution notes are documented in `DATA_SOURCES.md`.
- The current code does not deduplicate by `review_id`; `iclr_reviews_24.csv` contains 39 duplicate review rows, and `iclr_reviews_26.csv` contains 3,092 literal-column-name rows that are removed by numeric rating conversion.
