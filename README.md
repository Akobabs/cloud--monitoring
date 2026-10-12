# Comparative cloud anomaly detection

An offline research implementation of Isolation Forest, One-Class SVM, Random Forest and an LSTM Autoencoder. Includes a local Streamlit results interface, reproducible configurations, data provenance and report figures. **No unit tests are included**, as requested. Runtime data validation and end-to-end verification remain part of the research workflow.

## Retrained experiment — 30 September 2026

The new run `20260930T155937Z_research_improved` refits all 45 model/seed evaluations. It adds CPU and memory usage-to-request ratios to Google and calibrates every candidate before selecting by validation F1. The previous workflow recalibrated only the candidate selected at a fixed percentile threshold. Splits, seeds, original features, sampled fitting endpoints and candidate grids remain unchanged. The baseline is `20260928T092901Z_calibrated`.

| Dataset | Model | Previous F1 | Retrained F1 |
| --- | --- | ---: | ---: |
| Google | Isolation Forest | 0.4141 | 0.4188 |
| Google | One-Class SVM | 0.4473 | 0.4306 |
| Google | Random Forest | 0.4339 | 0.4438 |
| SMD | Isolation Forest | 0.2596 | 0.2643 |
| SMD | LSTM Autoencoder | 0.1942 | 0.1954 |
| SMD | One-Class SVM | 0.1566 | 0.1557 |
| SMD | Random Forest | 0.1043 | 0.1043 |

Google Random Forest's false-positive rate fell from 0.7612 to 0.5928; SMD Isolation Forest's fell from 0.1210 to 0.1120. Gains are modest and not universal: the strongest earlier Google F1 remains higher than the strongest revised Google F1. Both evidence audits passed, including identical evaluation records and training endpoints. This is exploratory reuse of a previously reviewed test period, not independent confirmation or deployment readiness.

```bat
run_project.bat improved
```

This selects `configs/improved.json` and generates the comparison with the retained local baseline. It does not apply a second calibration pass. The current dashboard displays the before-and-after table; complete metrics, ranking scores, resource costs and regressions are retained in `artifacts/runs/20260930T155937Z_research_improved/improvement_comparison.csv`. See `docs/improvement_experiment.md` for the fixed experiment plan.

## Online dashboard

See [hosting instructions](docs/hosting.md) for deployment. Use `deploy/app.py`
on Streamlit Community Cloud with Python 3.12. The repository includes a
dashboard-only results snapshot, so no training or dataset download is required
on the hosting server. For other hosts, install `requirements-dashboard.txt`.

## Start on Windows

Install **64-bit Python 3.12** (with the Python launcher) and ensure Internet access for the first run. Keep at least 5 GB of free disk space. CPU execution is configured for the research comparison; GPU is not required.

```bat
run_project.bat
```

This creates `.venv`, installs dependencies and the project, downloads the datasets, prepares the splits, trains the models, and exports baseline results. It then produces a separate validation-calibrated revision using the frozen models. Both runs are retained. Downloads are cached. The research profile evaluates three seeds and three predefined SMD machines. It may take substantial time on a laptop. For a smaller end-to-end demonstration:

```bat
run_project.bat quick
```

Then start the interface:

```bat
run_frontend.bat
```

Open http://localhost:8501 if your browser does not open automatically. The frontend reads saved results; it does not retrain or contact cloud services. Stop it with Ctrl+C. Both launchers support project paths containing spaces and stop on errors.

## Dataset interpretation

**Google 2019 Cluster sample**, Kaggle version 1: https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample

The sample contains a `failed` flag equal to `event == FAIL`. It is a failure proxy, not a curated generic anomaly label. Outcome fields, IDs and event timestamps are excluded from features. Repeated telemetry keys are removed, and tasks spanning chronological partition boundaries are purged. After deduplication, each task has at most four observations: a 30-step LSTM is therefore **not applicable** to this sample. The three tabular models are compared on it. Google results are not claims of prospective failure prediction because upstream join/label timing has not been independently established. Kaggle lists CC BY-NC-SA 4.0; the original Google trace is CC BY 4.0. Honour the applicable source terms and cite both.

**Server Machine Dataset**, from OmniAnomaly: https://github.com/NetManAIOps/OmniAnomaly

All four models are evaluated on `machine-1-1`, `machine-2-1`, and `machine-3-1` separately. The source is pinned to commit `7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4`. The published training portion is not anomaly-labelled. To support supervised Random Forest, this project explicitly uses a **60/20/20 chronological re-partition of the published labelled test sequence**. The original training files are downloaded and hashed for provenance but are not used to fit models. These are adapted benchmark results and must not be compared directly with scores using the official SMD train/test protocol. The selected machines are a subset of the 28-machine benchmark. The repository has an MIT licence; a separate dataset-specific licence was not identified.

## Baseline experimental protocol

- The split precedes fitting imputation and scaling. Median imputation and standard scaling are fitted only on training data.
- SMD features are 38 unnamed metric channels, their causal 5-observation means, and first differences (114 features). No unsupported CPU/memory names or sample intervals are assigned to anonymised channels.
- All models evaluate the same SMD endpoints, discarding the first 29 records of each partition. LSTM inputs are 30-step windows; its point score is the mean squared reconstruction error at the final step.
- A seeded sample of at most 6,000 common training endpoints bounds computation. Random Forest uses both labels. The other methods use the subset whose full input window is normal. Normal-only eligibility is shared by the three methods.
- All model thresholds are the 95th percentile of scores on entirely normal validation windows. Predictions are positive strictly above the threshold. Hyperparameters are selected by validation F1. If validation has no positives, the first predefined candidate is used without consulting test labels.
- LSTM has two encoder and two decoder LSTM layers, 8 or 16 latent units, Adam, MSE loss, up to 15 epochs, batch size 64 and patience 3. Early stopping uses up to 2,000 normal validation windows.
- Models run sequentially in separate processes. Seeds are 42, 43 and 44. The LSTM runs on CPU, with two compute threads and deterministic TensorFlow operations requested.
- F1 is primary; precision, recall, ROC-AUC, average precision and false-positive rate are also reported. Metrics are raw point-level scores without point adjustment. Test labels never select thresholds or hyperparameters.
- Training time is the selected candidate's fit time. Tuning time is recorded separately. Warm batch inference is scored three times and reported as median milliseconds per observation; it is not single-request or end-to-end serving latency. Process RSS is sampled every 10 ms during fitting, including native allocations and library overhead; instantaneous peaks may be missed.
- SMD family summaries macro-average machines within a seed, then report mean and sample standard deviation across seeds. They are not confidence intervals, significance tests or proof of generalisation.

## Validation calibration follow-up

The first experiment showed strong SMD score ranking but weak F1 at the fixed percentile threshold. The revised analysis isolates this decision-rule effect: trained weights, feature preparation, source records and continuous test scores are unchanged. For each model, `scripts/recalibrate_results.py` chooses the threshold with maximum **labelled-validation F1**, resolving exact ties with the higher threshold. If validation lacks either class, it retains the original threshold. All threshold decisions are saved before revised test predictions are generated.

The baseline is never overwritten. The revised run records its parent run, calibration script hash, validation scores, threshold decisions and a before/after table. Resource measurements are reused from the frozen baseline, not presented as newly measured training times. A threshold change cannot improve ROC-AUC or average precision because it leaves score ranking unchanged. It can increase F1 while also increasing false alarms; the dashboard and document report both.

The revision was motivated by an already reviewed test period. It is therefore an **exploratory follow-up**, not independent confirmation, even though its threshold function never uses test labels. All gains and regressions are retained. A new evaluation period is needed before claiming generalisation. The normal-only detectors use labelled supervision at calibration time.

To reproduce the follow-up from an existing original run:

```bat
.venv\Scripts\python.exe scripts\recalibrate_results.py --source artifacts\runs\YOUR_BASELINE_RUN
```

## Outputs and reproducibility

Input validation rejects missing Google telemetry keys, invalid usage times and any disagreement between `failed` and `event == FAIL`. SMD labels are checked before integer conversion so fractional or out-of-range labels cannot be silently changed. Evaluation and calibration require non-empty, aligned, one-dimensional binary labels and finite scores and thresholds; single-class validation retains the baseline threshold. A test period without positives reports undefined F1 as not applicable without failing the worker.

These maintenance safeguards do not retrain the saved experiments or establish an accuracy improvement. Historical run hashes continue to identify the original implementation; resuming those runs with changed source is intentionally rejected. Start a new run to evaluate the current code.

Each execution writes `artifacts/runs/<UTC timestamp>_<profile>/`:

```text
run.json, config.json, environment.txt, data_manifest.json
datasets/<dataset>/dataset.json, prepared.npz, preprocessor.joblib
models/<dataset>/<algorithm>/<seed>/
    model.joblib or model.keras
    result.json, predictions.csv, tuning.json, fit_indices.npz
    execution.log, training_history.json (LSTM only)
metrics.csv, comparison.csv, summary.csv, figures/
```

Source and configuration hashes, raw-file SHA-256 checksums, dependency versions and hardware are saved. Resume an interrupted run only with its original source/configuration:

```bat
.venv\Scripts\python.exe -m cloud_anomaly.cli run --config configs\research.json --resume artifacts\runs\YOUR_RUN
```

To regenerate tables and figures from an existing run:

```bat
.venv\Scripts\python.exe -m cloud_anomaly.cli report artifacts\runs\YOUR_RUN
```

If Kaggle download access changes, place the original `borg_traces_data.csv` in `data/raw/google/` and rerun. Do not place API tokens in source code. The `.gitignore` excludes Python caches, virtual environments, credentials, raw data, generated model artifacts, and all `.docx` files. Downloaded data and Word documents remain on disk but are not committed by Git.

## Research document

The current edition is `docs/Research_Implementation_and_Results_Retrained.docx`; its Markdown companion is `docs/research_manuscript.md`. It reports fresh results, gains and regressions, newly measured resource costs and current dashboard screenshots. Earlier Word editions are preserved, and the prior manuscript is retained as `docs/research_manuscript_before_retraining.md`.

Regenerate it after the experiment, audits and dashboard capture using the managed document runtime with `scripts/build_improved_document.py --run artifacts/runs/20260930T155937Z_research_improved`.

The delivered Word document contains the implementation plan, Chapters Four and Five, and Appendices A–C. Chapter Five contains exactly:

1. 5.1 Summary of Study and Findings
2. 5.2 Conclusion
3. 5.3 Recommendation for Future Research

Appendix A includes the codebase map and source listings; the runnable source files are the authoritative implementation. Chapter Four and Appendix B contain actual screenshots of the interface. Appendix C records dataset provenance, fields, partitions and caveats. Results are generated from completed runs, never invented. The supplied Chapters One–Three remain unchanged; the continuation explains the methodology amendments required by the datasets.

