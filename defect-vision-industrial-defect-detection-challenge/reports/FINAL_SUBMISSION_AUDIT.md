# Final Submission Audit

## Result

`submission.csv` was generated from the saved Phase 3 selection using all permitted normal training images and independently validated with `src.submission_validator`.

| Check | Result |
|---|---|
| Exact sample columns and order: `id,label` | Pass |
| Test rows / submission rows | 503 / 503 |
| IDs unique, complete, and in test CSV order | Pass |
| Missing values / non-finite scores | None / none |
| Score file traceability and label agreement | Pass |
| Labels emitted | numeric `0` and `1` only |
| Final submission SHA-256 | `a42650b2b089dc209846f74519d85af845c0dac283e01ea05cb5546a9be4d6a3` |
| Immutable backup | `outputs/immutable_submissions/submission_a42650b2b089dc209846f74519d85af845c0dac283e01ea05cb5546a9be4d6a3.csv` |

## Prediction distribution

| Category | Test rows | Label `1` score-based candidates |
|---|---:|---:|
| bottle | 83 | 53 |
| cable | 150 | 34 |
| hazelnut | 110 | 31 |
| screw | 160 | 15 |
| Total | 503 | 133 |

These are score-based anomaly candidates, not confirmed defects. Variation across categories follows separately calibrated models and thresholds; it was not adjusted toward a preferred class balance.

## Traceability

`outputs/final_prediction_scores.csv` contains every ID, path, category, selected method, continuous score, LOO-calibrated threshold, and output label. `outputs/final_calibration.csv` records the threshold source. `outputs/final_audit_plots/` contains score distributions plus highest-scoring and borderline image grids.

## Remaining contract risk

The downloaded sample submission uses numeric labels and the normal-only training label is `0`; `1` is the documented inferred defective encoding. Direct live Kaggle Evaluation/Rules access was unavailable in this environment. Do not upload before confirming the permitted label vocabulary and the `0=good`, `1=defective` mapping in Kaggle's Evaluation tab.
