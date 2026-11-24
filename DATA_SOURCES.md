# Data Sources

## Current Status

This artifact is prepared for a VLDB 2027 / PVLDB Volume 20 submission. ICLR is the source venue of the included review datasets, not the submission venue.

The CSV files are included as extracted ICLR official-review records from OpenReview. Official reviews became publicly visible on the release dates below. OpenReview Terms classify tasked reviews as Comments and release Comments under CC BY 4.0; associated metadata is dedicated under CC0 1.0.

## Dataset Files

| File | Size | Rows | Papers | Reviewers | Reviews | SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| `iclr_reviews_24.csv` | 20,634,412 bytes | 28,067 | 7,262 | 27,992 | 28,028 | `90CE3278D822041188A549FECDF2A923E0BDC29AB28DF2092B33E06180FA9BDE` |
| `iclr_reviews_25.csv` | 35,671,760 bytes | 47,984 | 11,837 | 47,867 | 47,984 | `CAB6A4049D2159A5F461BD881C2F67305527BEB3FF7D1A02CC533668FC81F31E` |
| `iclr_reviews_26.csv` | 61,575,115 bytes | 78,951 | 19,475 | 75,604 | 75,860 | `2704D14598F09D5D107DD61B98E047C1E483C67E74A9776E2F00808FB74E284A` |

## Dataset Audit

Checked locally on 2026-11-24 with the included CSV files:

| File | Rows | Unique papers | Unique reviews | Unique reviewers | Duplicate `review_id` rows | Rows with missing required fields |
|---|---:|---:|---:|---:|---:|---:|
| `iclr_reviews_24.csv` | 28,067 | 7,262 | 28,028 | 27,992 | 39 | 1 |
| `iclr_reviews_25.csv` | 47,984 | 11,837 | 47,984 | 47,867 | 0 | 1 |
| `iclr_reviews_26.csv` | 78,951 | 19,475 | 75,860 | 75,604 | 3,091 | 4 |

`iclr_reviews_26.csv` contains 3,092 rows whose fields are literal column names (`submission_id`, `title`, `reviewer_id`, `review_id`, `invitation`, `summary`, `soundness`, `presentation`, `contribution`, `rating`, `confidence`). Numeric rating conversion removes those rows during preprocessing. The 39 duplicate `review_id` rows in `iclr_reviews_24.csv` are exact duplicates and are not removed by the current code.

## Provenance

Official venue pages checked on 2026-11-24:

- ICLR 2024: https://openreview.net/group?id=ICLR.cc/2024/Conference ; reviews released 2023-11-10 according to https://iclr.cc/Conferences/2024/AuthorGuide
- ICLR 2025: https://openreview.net/group?id=ICLR.cc/2025/Conference ; reviews released 2024-11-13 according to https://iclr.cc/Conferences/2025/AuthorGuide
- ICLR 2026: https://openreview.net/group?id=ICLR.cc/2026/Conference ; reviews released 2025-11-11 according to https://iclr.cc/Conferences/2026/AuthorGuide

OpenReview retrieval and export references:

- API v2 notes endpoint: https://api2.openreview.net/notes ; OpenAPI definition: https://api2.openreview.net/docs/api.yml
- Retrieval guide: https://docs.openreview.net/how-to-guides/data-retrieval-and-modification/how-to-get-all-notes-for-submissions-reviews-rebuttals-etc
- CSV export guide: https://docs.openreview.net/how-to-guides/data-retrieval-and-modification/how-to-export-all-reviews-into-a-csv
- Terms: https://openreview.net/legal/terms

Live OpenReview group configuration checked on 2026-11-24 for `ICLR.cc/2024/Conference`, `ICLR.cc/2025/Conference`, and `ICLR.cc/2026/Conference`:

- `submission_name`: `Submission`
- `review_name`: `Official_Review`
- `review_rating`: `rating`
- `review_confidence`: `confidence`
- `public_submissions`: `true`

| Field | Value |
|---|---|
| Source name | ICLR 2024, ICLR 2025, and ICLR 2026 Conference official reviews (`Official_Review` notes) |
| Source URL or API endpoint | OpenReview venue pages above; API base `https://api2.openreview.net`; notes endpoint `https://api2.openreview.net/notes` |
| Access date | Not recorded in the original artifact; the exact snapshots are identified by the checksums below |
| Export version or snapshot ID | Not recorded in the original artifact; the exact snapshots are identified by the checksums below |
| Download/export command | Not recorded verbatim; the official reproducible method is API v2 `get_all_notes(..., details='replies')`, filter replies ending in `Official_Review`, then export selected note content to CSV as described in the OpenReview CSV export guide |
| Data license or terms | Official reviews are Comments under OpenReview Terms and are released under CC BY 4.0; metadata is CC0 1.0; article licenses are separate and do not apply to these review records |
| Redistribution permitted | Yes for public official-review records under CC BY 4.0, with attribution |
| Contact or maintainer | ICLR/OpenReview venue pages listed above; OpenReview policy contact: policy@openreview.net |

## Schema

Required columns used by `rk_gnn.py`:

| Column | Meaning |
|---|---|
| `submission_id` | Paper identifier |
| `title` | Paper title |
| `reviewer_id` | Reviewer identifier |
| `review_id` | Review identifier |
| `summary` | Review text embedded by the language model |
| `soundness` | Review score component |
| `presentation` | Review score component |
| `contribution` | Review score component |
| `rating` | Overall rating used as the regression signal |
| `confidence` | Reviewer confidence |

Additional columns may exist in the CSV files but are ignored by the current experiment.

## Preprocessing

Applied in `rk_gnn.py`:

1. Select the required columns.
2. Drop rows missing any required identifier or score component.
3. Convert `rating` to numeric, which also removes the literal-column-name rows in `iclr_reviews_26.csv`.
4. Drop rows with missing rating.
5. Create contiguous integer indices for papers, reviewers, and reviews.
6. Use mean rating per paper as the supervised target.
7. Do not deduplicate by `review_id`; the 39 exact duplicate review rows in `iclr_reviews_24.csv` remain in the current experiment.

## Privacy and Ethics

Review text and reviewer identifiers may be sensitive even when public. This release follows the source terms by retaining only experiment fields used by `rk_gnn.py`, documenting the CC BY 4.0 source terms, and excluding generated embeddings and predictions from Git.

## Versioning

The SHA-256 values above identify the exact included dataset files. If any CSV changes, update this table and create a new archived artifact version.
