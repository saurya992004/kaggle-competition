# Data Audit

Status: completed locally on 2026-10-16. The source data was not changed.

## Dataset reconciliation

| Item | Verified result |
|---|---:|
| Train rows | 742 |
| Test rows | 503 |
| Sample-submission rows | 503 |
| Referenced images | 1,245 |
| Files in `clips/` | 1,245 |
| Files missing from `clips/` | 0 |
| Unreferenced files in `clips/` | 0 |
| Decoding failures | 0 |
| Train/test ID overlap | 0 |
| Train/test path overlap | 0 |
| Exact byte-identical duplicate groups | 0 |

Every CSV row resolves to exactly one file below `clips/`. Both `id` columns are unique, all metadata cells are non-null, and the sample-submission IDs equal the test IDs as a set. The row order is preserved by `MetadataImageDataset`.

## Metadata contract observed on disk

`train.csv` has `id`, `path`, `category`, and `label`; `test.csv` has the same columns except `label`; and `sample_submission.csv` has `id,label`.

| Category | Train | Test |
|---|---:|---:|
| bottle | 135 | 83 |
| cable | 145 | 150 |
| hazelnut | 254 | 110 |
| screw | 208 | 160 |

All 742 training labels are numeric `0`. The sample submission's 503 placeholder labels are also `0`. The Kaggle Dataset Description supplied by the user says these represent `good`, and that `defective` is the hidden positive class; the numeric encoding still requires confirmation from the Evaluation tab before a submission is produced. There are no malformed category prefixes or duplicate IDs.

## Image integrity and variation

All images decoded successfully using Pillow. The raw files comprise 1,047 PNGs and 198 JPEGs (15.9% JPEG), all stored as three-channel RGB. There are no alpha-channel source images. Although every file is RGB, 420 have exactly identical R/G/B pixel values and are therefore visually grayscale.

| Decoded resolution | Images |
|---|---:|
| 192x192 | 300 |
| 224x224 | 287 |
| 256x256 | 342 |
| 288x288 | 316 |

The checked range is therefore 192–288 pixels in both dimensions. Representative train and test montages were manually inspected in `artifacts/audit/plots/`. Training images are visually category-aligned; their framing and lighting still vary. Test examples visibly include marked local appearance changes in several categories, but those observations were not converted into labels or used to recover hidden test truth. Histogram and intensity plots are descriptive only: they do not prove sensor noise, compression artefacts, or defects.

## Duplicate and leakage checks

SHA-256 found no exact duplicate group. An exhaustive within-category 64-bit difference-hash scan found 21,760 candidate pairs at Hamming distance <=4, including 10,029 train/test candidates. These are **not confirmed duplicates**: dHash is particularly collision-prone for aligned products on simple backgrounds. The candidate table is a review queue, not a leakage finding. Do not randomly split across such candidates without visual/embedding review in Phase 2.

## Preprocessing decision

The Phase 1 baseline in `config/preprocessing.yaml` applies EXIF orientation, converts any supported source to RGB, then letterboxes to 256x256 with bilinear interpolation and ImageNet normalization. A real source image produced a float32 CHW tensor of shape `(3, 256, 256)`. Letterboxing is the default because it retains the entire object; `center_crop` and `stretch` are implemented as explicit alternatives to test later. No file is silently skipped and no random preprocessing is used.

## Audit artefacts

- `artifacts/audit/audit_summary.json` — machine-readable verified totals.
- `artifacts/audit/image_manifest.csv` — image-level format, dimensions, hash, intensity, and status.
- `artifacts/audit/rejected_or_unreadable_files.csv` — empty, as expected.
- `artifacts/audit/exact_duplicate_groups.csv` and `near_duplicate_pairs_dhash_le_4.csv` — duplication evidence.
- `artifacts/audit/plots/` — montage and distribution diagnostics.

## Limitations and Phase 2 implication

There are no labeled defective training examples. A normal-only holdout can measure reconstruction/score stability or normal false-positive behaviour, but it cannot estimate defect F1. Phase 2 must report any synthetic-defect or held-out-normal result as a proxy, not hidden-test F1. Begin with category-aware normality models and deferred threshold calibration; do not use a single supervised binary classifier trained only on label `0`.
