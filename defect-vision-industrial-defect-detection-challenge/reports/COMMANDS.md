# Executed Commands and Verification

All commands below were executed in `C:\Users\saury\Desktop\kaggle-competition\defect-vision-industrial-defect-detection-challenge` unless stated otherwise.

| Command / action | Result |
|---|---|
| Workspace and CSV preview inspection | Passed; located the three CSVs and `clips/` images. |
| Direct HTTP requests to the supplied Kaggle overview/API endpoints | Not accessible: public URLs returned HTTP 404; no live rules were inferred. |
| Hardware, disk, framework, and credential-presence inspection | Passed; reports RTX 3050 hardware but CPU-only Torch; no Kaggle credential file/CLI. No secret contents were printed. |
| `python -m src.data_audit --data-root . --output-dir artifacts/audit` | Initial run exposed a pandas `mode` attribute collision during report construction; fixed before acceptance. A later full plotting run exceeded the 30-second terminal-response window after regenerating diagnostics, so diagnostics from the successful first run were retained. |
| `python -u -m src.data_audit --data-root . --output-dir artifacts/audit --skip-plots` | Passed; regenerated all tabular audit evidence and summary against all 1,245 images. |
| `python -m unittest discover -s tests -v` | Passed: 4 tests, 0 failures. |
| Visual inspection of `artifacts/audit/plots/train_montage.png` and `test_montage.png` | Completed. No hidden labels were assigned from this inspection. |

Generated outputs were inspected after creation: the summary reconciles 742 + 503 metadata rows with 1,245 successfully decoded clips; the rejected-file CSV is empty; the two montages and distribution plots exist and render.

## Phase 2 additions (2026-10-19)

| Command / action | Result |
|---|---|
| `python -m src.phase2_baseline --data-root . --output-dir artifacts/phase2` | Passed. Fit normal-only category baselines, ran 3-fold synthetic-proxy evaluation, scored all 503 test rows, and wrote diagnostics. |
| `python -m unittest discover -s tests -v` | Passed: 8 tests, 0 failures. |
| Post-run output validation | Passed: test score and candidate files each contain 503 rows; candidate IDs exactly match test order; selected-method columns are present. |
| Visual inspection of highest-score bottle and hazelnut montages | Completed as a diagnostic only; no labels or predictions were manually edited. |

## Phase 3 additions (2026-10-20)

| Command / action | Result |
|---|---|
| `python -m src.phase3_benchmark --data-root . --output-dir artifacts/phase3` | Passed after one documented runtime correction: reduced the deterministic patch coreset from 2,048 to 512 entries. Ran all four methods in leakage-safe 3-fold category CV and saved models, scores, plots, and ledger. |
| `python -m src.phase3_benchmark --data-root . --output-dir artifacts/phase3 --refine-selection` | Passed. Screened all completed methods on fixed held-out normals under brightness/noise shifts, then refit the robustness-screened candidate. |
| `python -m unittest discover -s tests -v` | Passed: 11 tests, 0 failures after Phase 3. |
| Saved-model reload verification | Passed: max absolute score error was 0 for all four selected category models. |

No test labels were accessed, no model was trained on test images, and no competition submission was generated.
