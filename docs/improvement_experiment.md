# Modelling revision: experiment plan

The configuration was fixed before running the revised experiment. The reference is the completed validation-calibrated run `20260928T092901Z_calibrated`.

Two changes are evaluated together:

1. Google receives two additional predictors: average CPU usage divided by requested CPU, and average memory usage divided by requested memory. Non-positive requests produce missing ratios, filled with training-fitted medians. These features use only the same record's existing resource fields, with no identifiers, outcomes or future observations.
2. Every candidate is calibrated on labelled validation data before candidate selection by validation F1. The former workflow selected candidates using a normal-percentile threshold and recalibrated only the winner. The revised workflow aligns selection with the final decision rule. Where validation lacks a class, the normal-percentile fallback remains; without positives, the first predefined candidate is used.

The source records, chronological split, purged tasks, endpoint alignment, training sample cap, random seeds, candidate grid and SMD features remain unchanged. All 45 dataset/model/seed evaluations are fitted afresh. The LSTM remains inapplicable to Google. No automatic second round of tuning will follow inspection of test results.

The comparison will include F1, precision, recall, false-positive rate, ROC-AUC, average precision and newly measured resource costs. All gains and regressions will be retained. Google combines both changes, so its differences cannot be attributed to ratios alone without an additional ablation. SMD isolates the changed selection procedure, subject to numerical variation.

The test period was examined in earlier work; this is an exploratory comparison, not an untouched confirmation set. The existing baseline and its source hashes will not be overwritten. A separate run and updated document will report the measured findings.

Reproduce with:

```bat
.venv\Scripts\python.exe -m cloud_anomaly.cli run --config configs\improved.json
```
