# Peer Review GNN Artifact

Supplementary artifact for "Reliability of graph based consolidation of reviews", an Experiment, Analysis & Benchmark (EA&B) submission to VLDB 2027 / PVLDB Volume 20.

- Repository: https://github.com/kram2006/OpenReviewData-PGReview
- Citation metadata: `CITATION.cff`

The artifact contains source code and open review datasets. The paper contains the problem statement, results, tables, figures, and analysis. Baseline and figure-generation scripts are not included.

## Contents

| Path | Purpose |
|---|---|
| `rk_gnn.py` | End-to-end preprocessing, embedding, GNN training, evaluation, and output generation |
| `run.sh` | Fresh-environment setup and run script for Linux |
| `run.ps1` | Fresh-environment setup and run script for Windows (NVIDIA GPU) PowerShell |
| `requirements.txt` | Direct Python dependencies used by the setup scripts |
| `requirements-lock.txt` | Full frozen dependency set for the tested environment |
| `iclr_reviews_24.csv` | ICLR 2024 review records |
| `iclr_reviews_25.csv` | ICLR 2025 review records |
| `iclr_reviews_26.csv` | ICLR 2026 review records |
| `DATA.md` | Dataset provenance, schema, counts, checksums, and terms |
| `CITATION.cff` | Citation metadata |
| `LICENSE` | MIT License for the code |
| `.gitignore` | Excludes environments, caches, embeddings, and generated outputs |

## Run

Linux:

```bash
bash run.sh
```

Windows (NVIDIA GPU) PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\run.ps1
```

Both scripts create a virtual environment, install `requirements.txt`, and run `python rk_gnn.py`. The default run loops over three datasets and three language models:

- `allenai/scibert_scivocab_uncased`
- `KISTI-AI/Scideberta-full`
- `KISTI-AI/scideberta-cs`

## Tested Environment

The code requires an NVIDIA GPU with a driver that supports CUDA 13.0.

The following versions installed and imported successfully in a fresh virtual environment on Windows 11 with a GPU present:

| Component | Version |
|---|---:|
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| torch-geometric | 2.8.0.post1 |
| transformers | 5.16.1 |
| tokenizers | 0.23.1 |
| huggingface-hub | 1.28.0 |
| numpy | 2.5.2 |
| pandas | 3.0.5 |
| scikit-learn | 1.9.0 |
| tqdm | 4.70.0 |
| sentencepiece | 0.2.3 |

First runs download Hugging Face checkpoints unless they are pre-cached.

The script creates a reviewer identity matrix of about 2.92 GiB for 2024, 8.54 GiB for 2025, and 21.29 GiB for 2026, before activations, gradients, optimizer states, and embeddings.

## Method as Implemented

1. Load one review CSV.
2. Keep the required columns and clean missing IDs and nonnumeric ratings.
3. Assign integer indices to papers, reviewers, and reviews.
4. Embed each review summary with mean pooling over pretrained language-model hidden states.
5. Build a heterogeneous graph with `paper` and `reviewer` nodes. Generated review embeddings and ratings are stored as edge attributes; `forward()` passes only node features and edge indices to the convolutions.
6. Train a two-layer heterogeneous GNN to predict mean paper rating.
7. Print MSE, MAE, R², and RMSE and write generated embedding and prediction files.

## Configuration

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
| Train/test split | 80/20 paper-level permutation |
| Target | Mean review rating per paper |
| Metrics | MSE, MAE, R², RMSE |

## Generated Outputs

| File | Meaning | Git status |
|---|---|---|
| `iclr_reviews_<year>_<model>_embeddings.dat` | Review-summary embeddings | Ignored |
| `iclr_reviews_<year>_<model>_gnn_paper_embeddings.npy` | Learned paper embeddings | Ignored |
| `iclr_reviews_<year>_<model>__gnn_predictions.csv` | Actual and predicted ratings per paper | Ignored |

`<year>` is `24`, `25`, or `26`; `<model>` is `scibert`, `scidebertafull`, or `scidebertacs`. Metrics are printed to the console.

## Data

See `DATA.md` for dataset counts, checksums, schema, cleaning rules, provenance, and terms.

## Implementation Notes

- Generated review embeddings and ratings are stored as edge attributes, but `SAGEConv` receives edge indices only; `ReviewGNN.edge_encoder` is defined and not called.
- Each edge feature includes a review rating, while the paper target is the mean review rating. Adding edge features to the convolution would require handling label leakage.
- Clean `reviewer_id` values match `Reviewer_XXXX`. Repeated IDs across papers are consistent with chance collisions of per-submission anonymous IDs.
- Reviewer node features are identity rows. Reviewers seen only in the test split receive no training gradient.
- No baseline comparing the model with a training-mean predictor is included.
- Training and test losses are printed once after training, not every 10 epochs. The printed time measures only the timer setup.
- The train/test split uses `torch.randperm(..., device=device)`; it is identical on the same device type only.
- Embedding failures assign zero vectors for the affected batch.

## Limitations

- The full run covers 3 datasets and 3 language models.
- CUDA kernels can produce small numerical differences across hardware.
- No short run path is included.
