# Research implementation plan

The implementation compares four established detectors on labelled server telemetry and uses the requested Google Cluster 2019 sample as a secondary tabular failure study. All conclusions must be backed by saved experimental outputs. No unit-test suite will be created.

## Scope and amendments to Chapters One to Three

The primary four-model experiment uses three predefined machines from the Server Machine Dataset (SMD): machine-1-1, machine-2-1 and machine-3-1. Each machine is fitted separately. Because the published SMD training portion does not supply anomaly labels for supervised learning, its labelled test sequence is re-partitioned chronologically into 60% training, 20% validation and 20% final evaluation. This is explicitly an adapted supervised comparison, not reproduction of the original benchmark protocol.

The Google Kaggle sample contains 405,894 rows and 34 columns. Repeated telemetry keys account for 152,300 rows. After deduplication, the maximum task history is four observations. It cannot support the proposed 30-step LSTM without constructing artificial histories. The three tabular algorithms will therefore be evaluated on Google, with LSTM marked not applicable. The `failed` label agrees with `event == FAIL`; neither that event field nor related outcome fields may be supplied as features. A nonfailure label is not a guarantee of healthy operation.

Other necessary amendments are to split before fitting preprocessing, keep model selection out of the test partition, replace the claim of identical input matrices with shared base features and aligned endpoints, and measure process memory rather than Python allocations alone. Offline batch results do not establish deployment readiness, real-time latency or root-cause diagnosis. The datasets do not support the original claim of combined CPU, memory, disk and connection-level telemetry; the implemented feature coverage must be stated accurately.

## Work packages and acceptance evidence

| Work package | Concrete output | Completion evidence |
|---|---|---|
| Dataset acquisition | Versioned downloads and manifest | Source URLs, SHA-256 hashes, file sizes and licence notes |
| Data audit | Dataset metadata and partition counts | Label validation, missing-value report, duplicate and temporal continuity checks |
| Common preparation | Saved imputer, scaler and feature definitions | Training-only fitting, disjoint partitions and common held-out row identifiers |
| Model implementation | Three classical detectors and LSTM autoencoder | Saved fitted models, selected configuration and training histories |
| Experiment execution | Three-seed research run | Completed run metadata and complete per-model predictions |
| Comparative analysis | Metric tables and publication figures | Recalculation from stored predictions agrees with saved results |
| Frontend | Local results explorer | Manual navigation, export checks and actual screenshots |
| Documentation | Chapters Four and Five and Appendices A to C | Every reported value traces to run outputs; Word pages visually reviewed |

## Experimental controls

The split fractions and SMD machines are fixed before fitting. Google is partitioned by telemetry start time, placing equal timestamps together; tasks that cross partition boundaries are excluded. Median imputation and standardisation learn only from training rows. For SMD, rolling means and first differences are computed causally and restart at each partition boundary. Each SMD model is evaluated at the endpoint of the same 30-observation window, with the first 29 records excluded from every partition.

Each seed selects at most 6,000 common training endpoints. Random Forest sees both labels; normal-only methods use the subset for which all observations in the input window are normal. The normal-only methods consequently have less labelled supervision and may fit fewer samples. This distinction must appear in the results chapter rather than being hidden under a fairness claim.

Thresholds are calibrated at the 95th percentile of scores on entirely normal validation windows. The model predicts positive only when its score exceeds the threshold. Hyperparameters are selected by validation F1 within the declared candidate grid. If a validation partition lacks positives, use the first predefined candidate and disclose the absence of label-based tuning. The final test partition never chooses a threshold, feature, model or architecture.

CPU execution, process isolation, thread limits, source/configuration hashes and dependency snapshots make the computational comparison reviewable. Report training fit time, total search time, warm batch inference time and sampled process RSS with their precise measurement boundaries. Do not describe batch-amortised latency as an online service response time.

## Verification without unit tests

Verification consists of syntax compilation, dependency consistency, a small end-to-end run, the configured research experiments, direct checks on saved prediction alignment and confusion counts, frontend navigation and document render review. These checks protect the integrity of the research without introducing a unit-test framework or test directory.

## Chapter and appendix structure

Chapter Four covers the implementation environment, dataset audit, preprocessing, model implementation, experiment protocol, measured results, computational cost, interface, discussion and limitations. Results tables distinguish Google failure labels from SMD anomaly labels and show the Google LSTM entry as not applicable.

Chapter Five has exactly three sections: **5.1 Summary of Study and Findings**, **5.2 Conclusion**, and **5.3 Recommendation for Future Research**. Limitations belong in Chapter Four and inform the future research discussion; no additional Chapter Five section is introduced.

Appendix A documents the runnable codebase, launch commands, configurations and source listings. Appendix B contains genuine screenshots captured from the implemented frontend. Appendix C records source/version/licence details, feature definitions, labels, missingness, exclusions, partition boundaries and raw-file checksums.

## Delivery and maintenance

The subsequent modelling revision is recorded in `docs/improvement_experiment.md` and `configs/improved.json`. It adds two Google resource-utilisation predictors and selects candidates after validation calibration. Run `20260930T155937Z_research_improved` completed 45 fresh model/seed evaluations; four family/model combinations gained F1, two regressed and one was unchanged. The current report is `Research_Implementation_and_Results_Retrained.docx`. Earlier documents and baseline artifacts remain available.

The maintained implementation enforces Google label/event agreement, complete telemetry keys and valid usage times, and validates SMD labels before integer conversion. Calibration and evaluation share strict input checks. A no-positive test period reports undefined F1 without crashing the worker. Verification reproduced metrics and thresholds for all 45 saved evaluations, checked 200 tied-score calibration cases against exhaustive search, confirmed identical prepared arrays for all four datasets, and completed a Random Forest worker run with a no-positive test period. These are maintenance checks, not a new accuracy experiment; historical results and hashes remain unchanged.

`run_project.bat` creates or reuses the Python environment, installs dependencies, acquires data and runs experiments. Its optional `quick` argument selects a small verification profile. `run_frontend.bat` opens the existing results interface without retraining. A failure must return a nonzero exit code and retain diagnostic output.

The `.gitignore` excludes virtual environments, Python caches, secrets, datasets, fitted model artifacts and Word files. The final Word document remains available locally. Markdown planning material and source code remain versionable. The original supplied research document is retained unchanged.

## Calibration follow-up requested after the initial results

The revised analysis preserves the original experiment, keeps its selected model weights fixed, and optimises decision thresholds using labelled validation F1. The baseline percentile threshold is retained when validation contains no positive examples. Every threshold is persisted before revised test predictions are produced; all outcomes are retained, including any regressions or increases in false alarms. Training time and resource measurements are inherited from the original frozen models and explicitly labelled as such.

Because the original evaluation period had already been reviewed, this follow-up is exploratory and requires independent confirmation on new data. Its purpose is to distinguish model discrimination from the choice of alert threshold, not to manufacture favourable scores. Chapter Four will show the baseline-versus-revised comparison, the false-alarm trade-off, a trivial all-positive reference for Google, and actual frontend screenshots within the chapter itself.

