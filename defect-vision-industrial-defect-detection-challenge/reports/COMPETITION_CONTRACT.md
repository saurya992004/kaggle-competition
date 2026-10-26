# Competition Contract

## Evidence level

The local CSVs/images and sample submission were directly inspected. The Dataset Description below was supplied by the enrolled user as copied Kaggle content. Direct live inspection was attempted at the supplied competition URL on 2026-10-16; unauthenticated HTTP access returned 404, and no signed-in browser or Kaggle credential is available to this environment. Claims marked **unresolved** are not assumptions.

## Confirmed from local files and user-provided Kaggle Dataset Description

| Contract item | Verified status |
|---|---|
| Task | Image-level binary prediction for each test ID |
| Object categories | `bottle`, `cable`, `screw`, `hazelnut` |
| Train images | 742 images; a single numeric label (`0`) only |
| Semantics stated in Description | Training images are good/normal; test is a hidden mix of normal and defective images |
| Test images | 503 rows; no `label` column |
| Image folder | `clips/`, paths from CSV metadata |
| Submission columns | Exactly `id,label` |
| Submission row identity | Sample IDs match test IDs exactly |
| Submission placeholder | Numeric `0` in every sample row |
| Formats and sizing | PNG/JPEG; 192–288 pixels; 420 pixel-monochrome RGB images |
| Dataset license | CC BY-NC-ND 4.0, as stated in supplied Kaggle content |

## Metric and label encoding

The initial brief says official metric is F1 with `defective` as positive. The supplied Dataset Description instead expresses labels as `good`/`defective`, while the actual downloaded files use numeric `0`. The data strongly suggests `0 = good` (train is normal-only and sample is all zero), but the mapping and exact F1 implementation have not been verified from the Evaluation page. No submission should be emitted until the following are confirmed:

1. allowed prediction vocabulary (`0/1` versus strings);
2. which class is positive and F1 averaging/threshold details;
3. required order (the implementation will preserve test CSV order regardless);
4. whether score accepts labels only or scores/probabilities.

## Rules and operational constraints

The following requirements are **unresolved** because the Rules and Evaluation tabs could not be reached live:

- allowance for external data and pretrained weights;
- team size/membership requirements;
- submission quota, deadline, and private/public leaderboard policy;
- code/data sharing constraints beyond the displayed dataset license.

Until confirmed, Phase 2 must use only the downloaded competition data and already-installed local tooling. It may document potential pretrained approaches but must not assume they are allowed.

## Modelling consequence

This is a one-class anomaly-detection problem, not a conventional labeled binary-classification problem. Report normal-only evaluation and synthetic anomaly experiments separately from true defect F1. Category-specific modelling and thresholds should be treated as likely necessary because object geometry differs markedly across the four categories.
