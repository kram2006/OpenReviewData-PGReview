# Data

The experiments use three included CSV files of ICLR official reviews retrieved from OpenReview. ICLR is the source venue of the data; this repository is the VLDB 2027 artifact.

## Files

| File | Raw rows | Clean rows | Papers | Reviewers | Reviews | Size | SHA-256 |
|---|---:|---:|---:|---:|---:|---:|---|
| `iclr_reviews_24.csv` | 28,067 | 28,067 | 7,262 | 27,992 | 28,028 | 20,634,412 bytes | `90CE3278D822041188A549FECDF2A923E0BDC29AB28DF2092B33E06180FA9BDE` |
| `iclr_reviews_25.csv` | 47,984 | 47,984 | 11,837 | 47,867 | 47,984 | 35,671,760 bytes | `CAB6A4049D2159A5F461BD881C2F67305527BEB3FF7D1A02CC533668FC81F31E` |
| `iclr_reviews_26.csv` | 78,951 | 75,859 | 19,474 | 75,603 | 75,859 | 61,575,115 bytes | `2704D14598F09D5D107DD61B98E047C1E483C67E74A9776E2F00808FB74E284A` |

Clean counts are the values after the cleaning steps in `rk_gnn.py`. “Reviews” is the number of unique clean `review_id` values.

## Audit Details

| File | Empty `summary` rows | Repeated header rows | Duplicate `review_id` rows retained |
|---|---:|---:|---:|
| `iclr_reviews_24.csv` | 1 | 0 | 39 |
| `iclr_reviews_25.csv` | 1 | 0 | 0 |
| `iclr_reviews_26.csv` | 4 | 3,092 | 0 after header removal |

All clean `reviewer_id` values match `Reviewer_XXXX`. The repeated header rows in `iclr_reviews_26.csv` contain literal column names; numeric rating conversion removes them. The 39 duplicate rows in `iclr_reviews_24.csv` are exact duplicates and are retained by the current code.

## Schema

`rk_gnn.py` uses these columns:

| Column | Meaning |
|---|---|
| `submission_id` | Paper identifier |
| `title` | Paper title |
| `reviewer_id` | Anonymous reviewer identifier |
| `review_id` | Review identifier |
| `summary` | Review text embedded by the language model |
| `soundness` | Review score component |
| `presentation` | Review score component |
| `contribution` | Review score component |
| `rating` | Overall rating used as the regression target |
| `confidence` | Reviewer confidence |

Additional columns are ignored.

## Cleaning in `rk_gnn.py`

1. Select the required columns.
2. Drop rows missing `submission_id`, `reviewer_id`, `review_id`, `soundness`, `presentation`, `contribution`, or `confidence`.
3. Replace missing `summary` values with empty strings.
4. Convert `rating` with `pd.to_numeric(..., errors="coerce")`.
5. Drop rows with missing rating.
6. Reset the row index.
7. Do not deduplicate by `review_id`.

## Provenance

Venue sources:

- ICLR 2024: https://openreview.net/group?id=ICLR.cc/2024/Conference ; review release date from https://iclr.cc/Conferences/2024/AuthorGuide
- ICLR 2025: https://openreview.net/group?id=ICLR.cc/2025/Conference ; review release date from https://iclr.cc/Conferences/2025/AuthorGuide
- ICLR 2026: https://openreview.net/group?id=ICLR.cc/2026/Conference ; review release date from https://iclr.cc/Conferences/2026/AuthorGuide

Review release dates stated by the author guides: 2023-11-10, 2024-11-13, and 2025-11-11, respectively.

OpenReview references:

- API: https://api2.openreview.net
- Retrieval guide: https://docs.openreview.net/how-to-guides/data-retrieval-and-modification/how-to-get-all-notes-for-submissions-reviews-rebuttals-etc
- CSV export guide: https://docs.openreview.net/how-to-guides/data-retrieval-and-modification/how-to-export-all-reviews-into-a-csv

A data collection script is not included in this repository.

## Licence

OpenReview Terms: https://openreview.net/legal/terms

"By submitting a Comment or Configuration Record (and their accompanying Edits) to OpenReview the Submitter agrees that it shall be released to the public under the Creative Commons Attribution 4.0 International (CC BY 4.0) license."
