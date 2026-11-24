# Supplementary Material Index

Paper: "Reliability of graph based consolidation of reviews"

Venue: VLDB 2027 / PVLDB Volume 20

Artifact URL: https://github.com/kram2006/OpenReviewData-PGReview

## Scope

The main paper contains the problem statement, method description, experimental setup, results, tables, figures, and analysis. This file only indexes the released source code and open datasets.

No supplementary result tables or figures are stored in this repository. Generated outputs are intentionally excluded and can be reproduced on a separate machine with the released code.

## Artifact Inventory

| File or directory | Type | Purpose |
|---|---|---|
| `rk_gnn.py` | Source code | End-to-end data loading, embedding, heterogeneous graph construction, GNN training, evaluation, and generated-output creation |
| `requirements.txt` | Environment | Fully pinned Python dependency set, including CUDA 13 PyTorch index |
| `myrun.sh` | Script | Linux/macOS fresh-environment setup and run |
| `run.ps1` | Script | Windows PowerShell fresh-environment setup and run |
| `iclr_reviews_24.csv` | Open dataset | ICLR 2024 review records |
| `iclr_reviews_25.csv` | Open dataset | ICLR 2025 review records |
| `iclr_reviews_26.csv` | Open dataset | ICLR 2026 review records |
| `README.md` | Documentation | Project overview, quickstart, and repository map |
| `REPRODUCIBILITY.md` | Documentation | Fresh-machine recipe and mapping from paper items to commands |
| `DATA_SOURCES.md` | Documentation | Dataset provenance, schemas, checksums, and licensing notes |
| `CITATION.cff` | Metadata | Citation information for GitHub, Zenodo, and readers |

## Main Paper to Artifact Mapping

| Main paper item | Released artifact | Command or file |
|---|---|---|
| Problem statement | Not duplicated here | Main paper |
| Method / algorithm | Source code | `rk_gnn.py` |
| Experimental setup | Environment and reproduction notes | `requirements.txt`, `REPRODUCIBILITY.md` |
| Data description | Open datasets and data notes | `iclr_reviews_*.csv`, `DATA_SOURCES.md` |
| Reported results | Not duplicated here | Main paper; regenerate on a separate machine with `python rk_gnn.py` |
| Generated outputs | Not committed | Regenerate with `python rk_gnn.py` |

## Checksums

Dataset checksums are recorded in `DATA_SOURCES.md`.

## Notes for Reviewers

- VLDB 2027 requires supplemental material; public GitHub is permitted: https://www.vldb.org/2027/submission-guidelines.html
- PVLDB reproducibility asks for code, data, experiment scripts, hardware notes, and restricted-resource access details: https://vldb.org/pvldb/reproducibility
- Use a tagged release or archival DOI for permanence.
- ICLR is the source venue of the datasets; VLDB 2027 is the submission venue.
- Dataset license and redistribution terms are stated in `DATA_SOURCES.md`.
- Generated embeddings, learned representations, and prediction files are excluded from Git.
