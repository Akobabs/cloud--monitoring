# Comparative Analysis of Anomaly Detection for Cloud Infrastructure

Implementation plan and research continuation

Chapter Four, Chapter Five and Appendices A to C

September 2026

This document records an implemented offline experiment and its measured results. Evidence comes from run 20260928T092901Z_calibrated, completed on 2026-09-28. The original Chapters One to Three are retained separately and unchanged. The methodological amendments below must accompany any later consolidation of the thesis.

The primary comparison covers all four algorithms on three selected SMD machines. The Google Cluster 2019 Kaggle sample supports a secondary comparison of the three tabular models. Its task histories are too short for the specified LSTM; that model is reported as not applicable on Google.

# Implementation Plan

The work was organised around data acquisition, controlled model implementation, experiment execution, evidence capture and research writing. No unit tests were included. Verification used input checks, dependency checks, end-to-end execution, review of saved predictions and manual inspection of the interface and document.

Plan 1 Work packages and deliverables

| Work package | Deliverable | Evidence |
| --- | --- | --- |
| Acquire and audit data | Google sample and selected SMD files | Version and SHA-256 manifest |
| Prepare common inputs | Training-only transformations and split records | Saved preprocessing and dataset metadata |
| Implement detectors | Four configurable algorithms | Model files and source code |
| Run comparison | Fixed validation and held-out evaluation | 45 model and seed evaluations |
| Present evidence | Frontend and comparison figures | Actual screenshots and exported metrics |
| Write research | Chapters Four and Five and appendices | Results linked to saved run artifacts |

The first launcher, run_project.bat, creates or reuses the Python environment, installs pinned dependencies, downloads data and executes the configured experiment. The second, run_frontend.bat, opens the local results interface without training models. The optional quick profile checks the complete workflow on a smaller fitting budget before the research profile is used.

Version control includes the source, configuration, planning material and dependency files. The .gitignore excludes Word files, virtual environments, Python caches, credentials, downloaded datasets and generated model artifacts. Exclusion from Git does not remove the delivered document or results from the local project directory.

# Methodology Decisions

The Google sample is a sparse set of failure-labelled usage records. Duplicate removal and task continuity checks precede model selection. The sample is not converted into artificial temporal sequences. SMD supplies real ordered, labelled observations, but its official normal-only training data cannot directly train a two-class Random Forest. A declared chronological re-partition of the labelled sequence enables the intended supervised comparison. Consequently, the reported SMD results do not reproduce the original benchmark protocol.

Dataset selection and chronological fractions were fixed before original model fitting. The original run used a normal-validation percentile threshold. The revised analysis retains those fitted models and selects decision thresholds from labelled validation data. Because the evaluation period had already been reviewed when this follow-up was proposed, the revised results are exploratory rather than an independent confirmation. No test labels are supplied to the threshold-selection routine. Chapter Five contains only the three requested sections.

# CHAPTER FOUR

# IMPLEMENTATION AND RESULTS

## 4.1 Introduction

This chapter describes the implemented comparative framework and reports the results of its execution. The framework evaluates Isolation Forest, One-Class Support Vector Machine, Random Forest and an LSTM Autoencoder using common base features, aligned evaluation records and a fixed validation procedure. It addresses both detection effectiveness and computational cost. The evidence is limited to offline data and does not establish the performance of a deployed streaming service.

## 4.2 Implementation Environment

The implementation used Python modules for acquisition, preprocessing, detector training, evaluation and reporting. Streamlit provided a local, read-only frontend for reviewing completed experiments. Models were executed sequentially in separate processes, avoiding contention among simultaneous model fits. TensorFlow was explicitly restricted to CPU execution.

Table 4.1 Recorded implementation environment

| Component | Recorded value |
| --- | --- |
| Operating system | Windows-11-10.0.26200-SP0 |
| Processor | Intel Core i5-8250U at nominal 1.60 GHz |
| Physical and logical cores | 4 physical and 8 logical |
| Installed memory | 7.85 GiB |
| Python | 3.12.14 |
| NumPy and Pandas | 2.2.6 and 2.3.3 |
| Scikit-learn | 1.7.2 |
| TensorFlow and Keras | 2.20.0 and 3.15.1 |
| Streamlit | 1.50.0 |
| Compute threads | 2 |
| Random seeds | 42, 43, 44 |

The exact installed environment is saved with the run. Random seeds and deterministic TensorFlow operations were set, but these controls do not imply bitwise equality across every hardware and library combination. Timings reflect this machine and its operating conditions, not a universal speed ranking.

## 4.3 Amendments to the Proposed Methodology

Implementation exposed several differences between the proposed design and the available evidence. First, fitted preprocessing must follow the split; fitting it before splitting would allow held-out information into training. Second, the LSTM requires windows while the other methods accept rows, so the comparison uses shared base features and common endpoint labels rather than literally identical input tensors. Third, Random Forest receives anomalous training labels, whereas the other detectors learn from normal windows. Their results therefore compare learning regimes as well as model families.

The original threshold rule used the 95th percentile of normal validation scores. Its weak F1 despite strong SMD ranking metrics motivated a controlled calibration follow-up. The revised rule maximises F1 on labelled validation data when both classes are present; otherwise the original threshold remains unchanged. Model weights, source records and continuous test scores are held fixed so the effect of the decision threshold can be measured separately. No unit-test suite was created.

## 4.4 Dataset Acquisition and Audit

Two data sources were used. The first was version 1 of Derrick Mwiti's Google 2019 Cluster sample on Kaggle, derived from Google's cluster traces. The second was the Server Machine Dataset distributed with OmniAnomaly at the pinned source revision recorded in Appendix C. The SMD experiment selected machine-1-1, machine-2-1 and machine-3-1 and treated them as separate series. This is a deliberately bounded subset, not a full evaluation of all SMD machines.

The Google file contained 405,894 rows and 34 columns. Deduplication on cluster, collection, instance, start time and end time removed 152,300 repeated records, leaving 253,594 distinct usage records. No conflicting failure label was found within a repeated key. A further 11,000 rows were removed because their task appeared across chronological partition boundaries.

The Google failed flag matched the FAIL event indicator for every source record. This is a failure proxy; observations labelled zero also include other event types and are not independently verified normal operation. Event fields and their numeric equivalents were excluded from predictors. The relationship between the sampled usage intervals and the event labels could not be reconstructed from documented join code, so these results describe retrospective discrimination and not prospective failure prediction.

After deduplication, the longest Google task history contained four observations. The configured LSTM requires 30. Concatenating different tasks would create an artificial history, so the Google LSTM entry is not applicable. By contrast, SMD provides ordered multivariate records and point anomaly labels. Its metric columns are anonymised; assigning unsupported CPU, disk or network names to them would be misleading.

Table 4.2 Eligible records after partitioning and sequence warm-up

| Dataset | Partition | Records | Positives |
| --- | --- | --- | --- |
| Google sample | train | 146,406 | 38,809 |
| Google sample | validation | 48,807 | 15,038 |
| Google sample | test | 47,381 | 12,683 |
| SMD 1-1 | train | 17,058 | 670 |
| SMD 1-1 | validation | 5,667 | 1,988 |
| SMD 1-1 | test | 5,667 | 7 |
| SMD 2-1 | train | 14,187 | 156 |
| SMD 2-1 | validation | 4,710 | 577 |
| SMD 2-1 | test | 4,710 | 437 |
| SMD 3-1 | train | 17,191 | 183 |
| SMD 3-1 | validation | 5,711 | 0 |
| SMD 3-1 | test | 5,711 | 125 |

SMD's published labelled test sequence was split 60/20/20 in time order. Its official training files were downloaded for provenance but were not used for fitting. The first 29 observations of each adapted partition were excluded from evaluation for every model. Table 4.2 records eligible endpoints before the training sample cap. The positive class means an anomaly in SMD and a failure-labelled record in Google.

## 4.5 Preprocessing and Feature Engineering

The loader validated the required schema and binary labels. Infinite values were treated as missing. Features with no observed training value were excluded, median values were fitted on the training partition, and a StandardScaler was fitted on the resulting training features. Validation and test data used these fitted objects without refitting. The pipeline saved both the imputer and scaler.

Google used 11 numeric resource features: average CPU and memory usage, maximum CPU and memory usage, sampled CPU usage, requested CPU and memory, assigned memory, page-cache memory, cycles per instruction and memory accesses per instruction. Identifiers, event outcomes and absolute timestamps were not predictors. No rolling statistics were created across unrelated Google records.

SMD used its 38 supplied metrics, a causal five-observation mean for each metric, and the first difference of each metric, producing 114 features. Feature history restarted at each partition boundary. A sequence contained 30 consecutive observations from one machine and one partition. Its label was the final observation's label, and the LSTM score was mean squared reconstruction error at that final observation. A training sequence qualified as normal only if all 30 labels were zero.

## 4.6 Implementation of the Algorithms

Each detector exposed fit, score, predict and save operations. Scores were oriented so that a larger value always indicated greater abnormality. For Isolation Forest and One-Class SVM, the library's normality score was negated. Random Forest used the predicted probability of the positive class. The LSTM produced reconstruction error. A shared threshold rule converted scores into predictions.

Table 4.3 Declared model search space

| Model | Candidate settings |
| --- | --- |
| Isolation Forest | 100 or 200 trees; maximum 256 samples per tree; normal-only fitting |
| One-Class SVM | RBF kernel; nu 0.01 or 0.05; gamma scale; 256 MB kernel cache |
| Random Forest | 100 trees and depth 10, or 200 trees and depth 20; balanced class weights; minimum leaf size 2 |
| LSTM Autoencoder | Two encoder and two decoder LSTM layers; latent width 8 or 16; Adam at 0.001; MSE; batch 64; maximum 15 epochs; patience 3 |

For each seed, up to 6,000 common training endpoints were sampled without replacement. Random Forest fitted both classes at these endpoints. The normal-only methods fitted the shared subset with entirely normal windows; consequently their fitted sample counts can be smaller. This cap controls laptop resource use and is part of the reported experimental design.

The LSTM used an encoder with twice the latent width followed by the latent layer, a repeat-vector operation, two decoder layers and a time-distributed linear output. Training reconstructed the full window. Early stopping monitored reconstruction loss on up to 2,000 normal validation windows, restoring the best observed weights. Source listings appear in Appendix A.

## 4.7 Experiment and Evaluation Procedure

The baseline fitted each candidate on training data and calibrated the 95th-percentile threshold on entirely normal validation windows. It retained the candidate with the highest validation F1, or the first predefined candidate when validation contained no positives. The revised analysis did not refit these models or search their hyperparameters again. It scored the same validation records using the frozen selected model, evaluated every distinct score boundary, and chose the boundary yielding maximum validation F1. Exact ties were resolved in favour of the higher threshold.

For machine-3-1, validation contained no positive examples, so the revised procedure retained the original threshold. It did not infer an optimum from final test labels. All 45 threshold decisions were saved before revised test predictions were produced. Strict greater-than comparisons were handled consistently at tied scores. The labelled calibration step supplies supervision to the decision rule even for the normal-only detectors and must not be described as a wholly unsupervised experiment.

The baseline completed 45 model and seed evaluations. The follow-up reevaluated these frozen models without additional fits: three tabular Google models and four models for each of three SMD machines, repeated over seeds 42, 43 and 44. Every improved and worsened result is retained. The evaluation period had already informed the decision to investigate calibration; independent confirmation on new records is still required. Point-level metrics are reported without expanding detections to whole anomaly intervals.

Precision, recall, F1, ROC-AUC, average precision and false-positive rate were computed. F1 remained the primary ranking criterion. Each SMD machine was first evaluated separately. For the SMD summary, machine scores were macro-averaged within a seed, then the mean and sample standard deviation across the three seeds were reported. Google scores were summarised across seeds directly. Seed standard deviation describes these repeated fits, not uncertainty across new cloud environments.

## 4.8 Detection Results

Table 4.4 compares the original percentile rule with the validation-selected threshold. The underlying anomaly scores are identical, so ROC-AUC and average precision do not change. Any difference in F1, precision, recall or false-positive rate is caused by the operating threshold. A larger F1 can come at the cost of more alerts; both outcomes are reported.

Table 4.4 Effect of validation threshold calibration

| Dataset | Algorithm | Baseline F1 | Revised F1 | Change |
| --- | --- | --- | --- | --- |
| Google | Isolation Forest | 0.0000 | 0.4141 | +0.4141 |
| Google | One-Class SVM | 0.0007 | 0.4473 | +0.4466 |
| Google | Random Forest | 0.0698 | 0.4339 | +0.3640 |
| SMD | Isolation Forest | 0.2170 | 0.2596 | +0.0426 |
| SMD | LSTM AE | 0.2314 | 0.1942 | -0.0372 |
| SMD | One-Class SVM | 0.1759 | 0.1566 | -0.0193 |
| SMD | Random Forest | 0.0416 | 0.1043 | +0.0626 |

Calibration did not improve every model. F1 decreased for LSTM Autoencoder on SMD (0.2314 to 0.1942); One-Class SVM on SMD (0.1759 to 0.1566). These regressions are retained rather than selecting whichever threshold happened to perform best on the test period. The result shows that a validation optimum can fail to transfer to later observations.

Table 4.5 SMD macro-average detection performance

| Algorithm | Precision | Recall | F1 | ROC-AUC | FPR |
| --- | --- | --- | --- | --- | --- |
| Isolation Forest | 0.2104 | 0.7883 | 0.2596 | 0.9099 | 0.1210 |
| LSTM AE | 0.3899 | 0.7005 | 0.1942 | 0.9839 | 0.0838 |
| One-Class SVM | 0.3711 | 0.6967 | 0.1566 | 0.9873 | 0.0779 |
| Random Forest | 0.0769 | 0.6171 | 0.1043 | 0.8271 | 0.2607 |

Table 4.6 Google failure-proxy detection performance

| Algorithm | Precision | Recall | F1 | ROC-AUC | FPR |
| --- | --- | --- | --- | --- | --- |
| One-Class SVM | 0.2935 | 0.9451 | 0.4473 | 0.5357 | 0.8386 |
| Random Forest | 0.2908 | 0.8538 | 0.4339 | 0.6596 | 0.7612 |
| Isolation Forest | 0.2735 | 0.8788 | 0.4141 | 0.5323 | 0.8587 |
| LSTM AE | N/A | N/A | N/A | N/A | N/A |

On the adapted SMD comparison, Isolation Forest had the highest mean F1 of 0.2596, with mean recall 0.7883 and false-positive rate 0.1210. On the Google failure-proxy task, One-Class SVM had the highest mean F1 of 0.4473. These rankings are specific to the selected data, supervision and threshold policy. They do not establish one algorithm as best for every cloud workload.

A trivial Google rule that labels every record positive would obtain F1 0.4223, recall 1.0000 and false-positive rate 1.0000 on this final segment. It is included as an interpretation baseline, not a deployable detector. Revised Google results must be judged against this baseline and their false-alarm rate, not simply against the much lower original F1. The positive prevalence is 0.2677.

Table 4.7 F1 variability and average precision

| Dataset | Algorithm | Mean F1 | F1 SD | Average precision |
| --- | --- | --- | --- | --- |
| Google | Isolation Forest | 0.4141 | 0.0143 | 0.2721 |
| Google | One-Class SVM | 0.4473 | 0.0251 | 0.2660 |
| Google | Random Forest | 0.4339 | 0.0119 | 0.4029 |
| SMD | Isolation Forest | 0.2596 | 0.0180 | 0.2961 |
| SMD | LSTM AE | 0.1942 | 0.0087 | 0.8615 |
| SMD | One-Class SVM | 0.1566 | 0.0017 | 0.8267 |
| SMD | Random Forest | 0.1043 | 0.0247 | 0.0927 |

![Figure 4.1 SMD macro-average F1 and false-positive rate with seed standard deviation](../artifacts/runs/20260928T092901Z_calibrated/figures/smd_comparison.png)

![Figure 4.2 Google failure-proxy F1 and false-positive rate with seed standard deviation](../artifacts/runs/20260928T092901Z_calibrated/figures/google_comparison.png)

Table 4.8 Mean F1 by SMD machine

| Machine | Isolation Forest | One-Class SVM | Random Forest | LSTM AE |
| --- | --- | --- | --- | --- |
| 1-1 | 0.0134 | 0.0200 | 0.0150 | 0.0137 |
| 2-1 | 0.4348 | 0.1642 | 0.2140 | 0.1826 |
| 3-1 | 0.3306 | 0.2855 | 0.0838 | 0.3862 |

The machine-level table is essential because an aggregate can conceal very different operating conditions. The final machine-1-1 segment contains only seven positive points before warm-up exclusion, so a small change in detections can substantially alter its recall. Machine-3-1 has no positive validation points, limiting label-based model selection. Neither class imbalance nor these temporal shifts is corrected by averaging seeds.

![Figure 4.3 Confusion matrices for SMD machine-2-1 at seed 42](../artifacts/runs/20260928T092901Z_calibrated/figures/smd_machine-2-1_confusion.png)

## 4.9 Computational Results

Table 4.9 Mean computational measurements

| Dataset | Algorithm | Fit time seconds | Latency ms per point | Peak RSS MB |
| --- | --- | --- | --- | --- |
| Google | Isolation Forest | 0.44 | 0.0055 | 171.8 |
| Google | One-Class SVM | 0.04 | 0.0099 | 172.7 |
| Google | Random Forest | 1.83 | 0.0108 | 182.9 |
| SMD | Isolation Forest | 0.37 | 0.0106 | 175.5 |
| SMD | LSTM AE | 55.25 | 1.9284 | 889.7 |
| SMD | One-Class SVM | 0.24 | 0.0407 | 186.9 |
| SMD | Random Forest | 2.40 | 0.0164 | 176.4 |

The lowest mean SMD fitting time was recorded by One-Class SVM at 0.24 seconds. Fitting time measures the selected candidate's fit call, excluding preprocessing, model construction and validation scoring. Total tuning time is preserved separately in the run files. These distinctions matter particularly for the neural model, whose framework initialisation and architecture construction are not included in fit time.

Inference was warmed up and repeated three times over the held-out array during the original run; median time was divided by evaluated observations. Table 4.9 reuses those measurements because model weights and scoring computations did not change. The calibration follow-up did not retrain the models or remeasure their training cost. Its additional loading and calibration times are saved separately. The latency is batch-amortised scoring time, not single-request delay. Memory is sampled process RSS, including libraries and data, and can miss short allocation spikes.

## 4.10 Frontend and Operational Verification

The interface presents an overview, dataset-specific model comparisons, a prediction explorer, dataset details and reproducibility metadata. Users can inspect confusion matrices and score curves, identify false positives and missed anomalies, and download results or predictions. It reads saved artifacts and neither retrains models automatically nor sends production alerts. Appendix B contains screenshots captured from this running application.

![Figure 4.4 Frontend overview of the completed comparative experiment](../artifacts/screenshots/01_overview.png)

Figure 4.4 shows the implemented dashboard with the completed evaluation count and separate SMD and Google comparisons. The two sources remain separate because their labels represent different targets. The run selector also preserves access to the baseline experiment.

![Figure 4.5 Frontend comparison of all four algorithms on SMD machine-2-1](../artifacts/screenshots/06_smd_comparison.png)

Figure 4.5 shows the comparison screen used to inspect algorithm metrics and export the underlying records. The interface presents the revised decision metrics alongside the recorded resource measurements, allowing the reader to assess detection quality and computation together.

Verification included Python syntax compilation, dependency consistency, a quick end-to-end execution and the research run. Saved prediction identifiers and labels were checked for agreement across models, and confusion counts and F1 values were recalculated from the prediction files. The frontend was opened in a browser, its five views were visited, and a CSV download was exercised. These checks were performed without adding unit tests to the codebase.

## 4.11 Discussion and Limitations

The LSTM's mean F1 was 0.1942, compared with 0.2596 for Isolation Forest. Its mean fitting time was approximately 151 times greater and its scoring time per point approximately 182 times greater. One-Class SVM's mean ROC-AUC was 0.9873, compared with 0.9839 for the LSTM. These score-ranking values did not change after calibration, demonstrating that an improved operating point is distinct from improved learned discrimination.

The highest-F1 Google model had mean recall 0.9451, F1 0.4473 and false-positive rate 0.8386. These measures must be considered together with the all-positive baseline. A threshold that increases recall by raising many additional alerts is not automatically an operational improvement. Unverified label timing, the sampled population and temporal distribution differences still limit deployment claims.

The revised Google false-positive rates are very high, so its larger F1 values do not support practical deployment. On SMD, the higher-F1 Isolation Forest operating point also increases false alarms relative to its baseline, while the LSTM and One-Class SVM lose F1 after calibration. The improvement in this chapter is a more complete, auditable comparison and a measured calibration experiment; it is not evidence that all detector weaknesses have been solved.

The experiment meets the implementation objective by providing a common orchestration and evaluation pipeline for four model families. It also demonstrates that fair comparison requires more than supplying the same table to every algorithm. The LSTM uses temporal context, Random Forest uses labelled anomalies, and the normal-only detectors learn a different training distribution. These differences are relevant to deployment choice and prevent attributing every performance difference solely to model architecture.

The calibration follow-up addresses the mismatch between score ranking and binary decisions by selecting thresholds on validation labels. It does not guarantee a false-positive budget, and its validation optimum may transfer poorly when anomaly prevalence or score distributions change. Machine-3-1 remains on the original rule because labelled calibration is unsupported there. The follow-up was motivated by previously observed test performance; although test labels do not enter the calibration function, the revised findings should be treated as exploratory until confirmed on a new evaluation period.

The Google sample narrows the study to a failure-proxy task and lacks sufficient within-task history for the temporal model. Its positive rate is not representative of the original production trace by default. The SMD adaptation uses only three machines, changes the original benchmark split and relies on supplied labels. Selected fitting caps, a small candidate grid and three seeds limit the range of conclusions. The experiment does not evaluate streaming drift, incident response, attack novelty, explanation quality, model serving or cost in a production cloud.

Comparison with prior literature is therefore qualitative. The results can illustrate the trade-off between supervised discrimination, normal-pattern detection and reconstruction cost, but numerical claims of improvement over published Google or SMD studies would require matching their labels, splits, preprocessing, thresholding and evaluation definitions. This study makes no such superiority claim.

## 4.12 Chapter Summary

The implemented framework produced repeatable experiment artifacts, measured detection and resource metrics, and a usable local interface. Dataset auditing required a documented SMD split adaptation and the exclusion of LSTM from the sparse Google sample. The recorded evidence supports conditional conclusions about the selected workloads and provides the basis for Chapter Five.

# CHAPTER FIVE

# SUMMARY CONCLUSION AND RECOMMENDATION

## 5.1 Summary of Study and Findings

The study investigated the relative effectiveness and computational requirements of Isolation Forest, One-Class SVM, Random Forest and an LSTM Autoencoder for offline anomaly detection in cloud-related telemetry. A modular Python implementation provided data acquisition, training-only preprocessing, controlled model fitting, common metric computation and a local reporting frontend. Three seeds were used to expose variation among repeated fits.

The primary four-model comparison used three SMD machines under an explicitly adapted chronological protocol that provided labelled training examples for Random Forest. A secondary experiment used the requested Google Cluster 2019 sample. Auditing that sample identified extensive repeated telemetry and task histories of at most four observations, so only the three tabular algorithms could be meaningfully evaluated there. Its failed flag represented a failure proxy rather than a comprehensive cloud anomaly label.

The highest SMD macro-average F1 was 0.2596, achieved by Isolation Forest. The highest Google F1 was 0.4473, achieved by One-Class SVM. The findings also showed that results differed across SMD machines and that metric rankings must be considered alongside threshold behaviour and computational measurements. One-Class SVM recorded the lowest mean SMD fitting time under the specified measurement boundary.

A controlled follow-up retained the fitted models and revised decision thresholds using labelled validation data. The baseline was preserved and reported alongside the revision. This isolated the effect of calibration from learning and showed why a high score-ranking metric can coexist with weak F1 at a particular operating point. The revised evaluation reused a reviewed test period and therefore remains exploratory.

The objectives were addressed by implementing the framework and algorithms, evaluating the stated detection and resource metrics, and identifying selection criteria grounded in the observed conditions. The four-model objective was fulfilled on SMD; the Google experiment remained a three-model supplementary comparison. The final source code, launchers, saved results and dataset documentation provide an auditable record of the implementation.

## 5.2 Conclusion

Within this exploratory adapted SMD evaluation, Isolation Forest ranked highest by mean F1 under the revised decision rule. This conclusion is limited to the selected machines, supervision, features, sample budget and CPU environment. It does not establish universal superiority or justify deployment. A new evaluation period is required to confirm that the apparent calibration gains generalise.

The study concludes that dataset suitability and evaluation design are central to meaningful anomaly-detector comparison. Temporal models require genuine histories, supervised methods require representative anomaly labels, and validation choices influence the operational false-alarm trade-off. Resource measurements add necessary context, but offline batch results alone do not establish real-time monitoring capability. Preserving these qualifications makes the implementation a defensible basis for further investigation.

## 5.3 Recommendation for Future Research

Future research should first broaden the temporal evidence. A verified contiguous extraction from the original Google trace should link usage windows to event outcomes with explicit prediction horizons and exclude post-outcome information. Such an extraction would permit a meaningful LSTM comparison and a clearer distinction between anomaly detection and failure prediction. Extending the SMD evaluation to all machines and adding independent infrastructure benchmarks would improve the evidence for generalisation.

Further work should compare the present supervised adaptation with a separate normal-only benchmark protocol. Where anomaly labels are unavailable for fitting, Random Forest should either be omitted or trained under a clearly different, independently evaluated weak-supervision design. Sensitivity analyses should vary temporal window length, training budget, anomaly prevalence and threshold calibration while preserving a final untouched evaluation period.

A subsequent deployment study should measure concept drift, event-level detection delay, false alerts per unit time and end-to-end serving latency. It should assess retraining frequency, model loading and preprocessing cost, memory under sustained operation and the effect of hardware acceleration. Repeated temporal holdouts, uncertainty estimates that account for serial dependence and inspection of failure cases would support stronger conclusions than seed repetition alone.

# APPENDIX A

# CODEBASE

The complete runnable codebase is supplied in the project directory. This appendix maps every main component and reproduces the detector and evaluation modules. The acquisition, orchestration and frontend files remain available as source alongside these listings; the printed appendix does not replace the runnable files. Line wrapping below is for presentation only.

Table A.1 Codebase organisation

| Path | Responsibility |
| --- | --- |
| src/cloud_anomaly/data.py | Downloads, provenance, schema checks, splitting, imputation and feature preparation |
| src/cloud_anomaly/models.py | Classical detectors and LSTM Autoencoder |
| src/cloud_anomaly/worker.py | Candidate fitting, threshold calibration, selection, prediction and persistence |
| src/cloud_anomaly/evaluation.py | Detection metrics, RSS sampling and inference measurements |
| src/cloud_anomaly/cli.py | Sequential isolated execution and resume checks |
| src/cloud_anomaly/reporting.py | Aggregated tables and figures |
| app.py | Local Streamlit results interface |
| configs/research.json and quick.json | Recorded research and verification settings |
| requirements.txt and requirements-lock.txt | Direct and exact installed dependency versions |
| run_project.bat | Environment setup, data acquisition and full execution |
| run_frontend.bat | Frontend launcher without retraining |
| scripts/recalibrate_results.py | Validation-only threshold revision with frozen models and preserved baseline |
| .gitignore | Python, credentials, datasets, artifacts and Word exclusions |

## A.1 Execution Instructions

Install 64-bit Python 3.12 with the Python launcher, then run run_project.bat from the project directory. The optional command run_project.bat quick uses the smaller verification configuration. After completion, run run_frontend.bat and open http://localhost:8501. The launchers quote paths and return nonzero on failure. Internet access is required for uncached dependencies and datasets.

Each run directory preserves its configuration, raw-file manifest, dependency snapshot, dataset preparation metadata, fitted models, per-record predictions and figures. A run can be resumed only when its source and configuration hashes match. The .gitignore deliberately excludes all .docx files, including this research document; retain or share the document separately from Git.

## A.2 Detector Implementations

Subsequent implementation maintenance strengthens input validation without changing the reported experimental results. Google records must contain telemetry keys and finite, ordered usage times, and the failed flag must agree with event=FAIL. SMD labels are validated before integer conversion. Calibration and metric calculation require non-empty, aligned one-dimensional labels and finite scores and thresholds. Single-class validation retains the baseline threshold, and an evaluation period with no positives reports F1 as not applicable without terminating the worker. These safeguards do not constitute a new training experiment or evidence of improved detection accuracy. Historical source hashes remain attached to the original runs; the listings below reflect the maintained implementation.

```python
# src/cloud_anomaly/models.py
"""Common scoring convention: larger scores always mean more anomalous."""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.svm import OneClassSVM
import joblib

NAMES = {
    "isolation_forest": "Isolation Forest", "one_class_svm": "One-Class SVM",
    "random_forest": "Random Forest", "lstm_autoencoder": "LSTM Autoencoder",
}


class TabularDetector:
    def __init__(self, kind: str, parameters: dict, seed: int, threads: int):
        self.kind = kind
        self.threshold = None
        if kind == "isolation_forest":
            self.model = IsolationForest(**parameters, contamination="auto", random_state=seed, n_jobs=threads)
        elif kind == "one_class_svm":
            self.model = OneClassSVM(**parameters, kernel="rbf", cache_size=256)
        elif kind == "random_forest":
            self.model = RandomForestClassifier(**parameters, class_weight="balanced",
                min_samples_leaf=2, random_state=seed, n_jobs=threads)
        else:
            raise ValueError(kind)

    def fit(self, x, y=None, validation=None):
        self.model.fit(x, y)
        return self

    def score(self, x):
        if self.kind == "random_forest":
            index = list(self.model.classes_).index(1)
            return self.model.predict_proba(x)[:, index]
        return -self.model.score_samples(x)

    def predict(self, x):
        if self.threshold is None:
            raise ValueError("Threshold has not been calibrated")
        return (self.score(x) > self.threshold).astype(np.int8)

    def save(self, path):
        joblib.dump(self, path.with_suffix(".joblib"))


class LSTMAutoencoder:
    def __init__(self, parameters, seed, threads, shape):
        import tensorflow as tf
        self.tf = tf
        tf.keras.backend.clear_session()
        tf.keras.utils.set_random_seed(seed)
        tf.config.experimental.enable_op_determinism()
        try:
            tf.config.threading.set_intra_op_parallelism_threads(threads)
            tf.config.threading.set_inter_op_parallelism_threads(1)
        except RuntimeError:
            pass  # Thread settings are process-wide and already set for subsequent candidates.
        self.parameters = parameters
        self.threshold = None
        width = parameters["latent_dim"]
        inputs = tf.keras.Input(shape=shape)
        x = tf.keras.layers.LSTM(width * 2, return_sequences=True)(inputs)
        x = tf.keras.layers.LSTM(width)(x)
        x = tf.keras.layers.RepeatVector(shape[0])(x)
        x = tf.keras.layers.LSTM(width, return_sequences=True)(x)
        x = tf.keras.layers.LSTM(width * 2, return_sequences=True)(x)
        output = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(shape[1]))(x)
        self.model = tf.keras.Model(inputs, output)
        self.model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), loss="mse")
        self.history = {}

    def fit(self, x, y=None, validation=None):
        tf = self.tf
        options = tf.data.Options()
        options.threading.private_threadpool_size = 1
        options.threading.max_intra_op_parallelism = 1
        def dataset(values):
            return tf.data.Dataset.from_tensor_slices((values, values)).batch(
                self.parameters["batch_size"]).with_options(options)
        callbacks = [tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=3,
                                                     restore_best_weights=True)]
        history = self.model.fit(dataset(x), validation_data=dataset(validation),
            epochs=self.parameters["epochs"], callbacks=callbacks, verbose=0, shuffle=False)
        self.history = {key: [float(v) for v in values] for key, values in history.history.items()}
        return self

    def score(self, x):
        # Last-step error produces a point score aligned with each window endpoint.
        scores = []
        for start in range(0, len(x), 256):
            batch = x[start:start + 256]
            reconstruction = self.model(batch, training=False).numpy()
            scores.append(np.mean(np.square(batch[:, -1, :] - reconstruction[:, -1, :]), axis=1))
        return np.concatenate(scores)

    def predict(self, x):
        if self.threshold is None:
            raise ValueError("Threshold has not been calibrated")
        return (self.score(x) > self.threshold).astype(np.int8)

    def save(self, path):
        self.model.save(path.with_suffix(".keras"))

```

## A.3 Metric and Resource Measurement Implementations

```python
# src/cloud_anomaly/evaluation.py
"""Metrics, measured process memory, and publication figures."""
from __future__ import annotations

import threading
import time

import numpy as np
import psutil
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                            precision_score, recall_score, roc_auc_score)


def validate_scores(y, scores, threshold):
    """Validate aligned binary labels and scores before any decisions are made."""
    y = np.asarray(y)
    scores = np.asarray(scores, dtype=np.float64)
    if y.ndim != 1 or scores.ndim != 1 or not len(y) or y.shape != scores.shape:
        raise ValueError("Labels and scores must be non-empty aligned one-dimensional arrays")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(scores).all():
        raise ValueError("Expected binary labels and finite scores")
    if np.ndim(threshold) != 0 or not np.isfinite(threshold):
        raise ValueError("Threshold must be a finite scalar")
    return y.astype(np.int8), scores


def metrics(y, scores, threshold):
    y, scores = validate_scores(y, scores, threshold)
    prediction = (scores > threshold).astype(np.int8)
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    both_classes = len(np.unique(y)) == 2
    return {
        "precision": float(precision_score(y, prediction, zero_division=0)),
        "recall": float(recall_score(y, prediction, zero_division=0)) if (tp + fn) else None,
        "f1": float(f1_score(y, prediction, zero_division=0)) if (tp + fn) else None,
        "roc_auc": float(roc_auc_score(y, scores)) if both_classes else None,
        "average_precision": float(average_precision_score(y, scores)) if both_classes else None,
        "false_positive_rate": float(fp / (fp + tn)) if (fp + tn) else None,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
    }


class MemoryMonitor:
    """Sample process RSS, including native allocations, every 10 ms.

    This is a sampled maximum and can miss short-lived allocation spikes.
    Windows lifetime peak working-set is also reported separately.
    """
    def __enter__(self):
        self.process = psutil.Process()
        self.baseline = self.process.memory_info().rss
        self.peak = self.baseline
        self.stop = threading.Event()
        def sample():
            while not self.stop.wait(0.01):
                self.peak = max(self.peak, self.process.memory_info().rss)
        self.thread = threading.Thread(target=sample, daemon=True)
        self.thread.start()
        return self

    def __exit__(self, *args):
        self.peak = max(self.peak, self.process.memory_info().rss)
        self.stop.set()
        self.thread.join()
        memory = self.process.memory_info()
        self.lifetime_peak = getattr(memory, "peak_wset", self.peak)


def benchmark_score(model, x, repetitions=3):
    model.score(x[:min(256, len(x))])  # untimed warm-up
    timings = []
    for _ in range(repetitions):
        start = time.perf_counter()
        scores = model.score(x)
        timings.append(time.perf_counter() - start)
    return scores, float(np.median(timings) * 1000 / len(x)), timings


```

## A.4 Experiment Worker

```python
# src/cloud_anomaly/worker.py
"""Isolated model process; called sequentially by the experiment runner."""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .data import write_json
from .evaluation import MemoryMonitor, benchmark_score, metrics
from .models import LSTMAutoencoder, NAMES, TabularDetector


def windows(x, length):
    return np.lib.stride_tricks.sliding_window_view(x, length, axis=0).transpose(0, 2, 1)


def run(dataset_dir: Path, output: Path, config: dict, kind: str, seed: int):
    output.mkdir(parents=True, exist_ok=True)
    prepared = np.load(dataset_dir / "prepared.npz")
    metadata = json.loads((dataset_dir / "dataset.json").read_text(encoding="utf-8"))
    length = metadata["window"]
    if kind == "lstm_autoencoder" and not metadata["temporal"]:
        raise ValueError("LSTM cannot train on unrelated Google tasks")
    raw_x = {part: prepared[f"x_{part}"] for part in ("train", "validation", "test")}
    raw_y = {part: prepared[f"y_{part}"] for part in raw_x}
    aligned_y = {part: y[length - 1:] for part, y in raw_y.items()}
    normal_windows = {part: np.lib.stride_tricks.sliding_window_view(y, length).sum(axis=1) == 0
                      for part, y in raw_y.items()}
    rng = np.random.default_rng(seed)
    eligible = np.arange(len(aligned_y["train"]))
    chosen = np.sort(rng.choice(eligible, min(len(eligible), config["max_fit_samples"]), replace=False))
    supervised = kind == "random_forest"
    fit_indices = chosen if supervised else chosen[normal_windows["train"][chosen]]
    if len(fit_indices) < 32:
        raise ValueError("Insufficient normal training windows")
    if supervised and len(np.unique(aligned_y["train"][fit_indices])) < 2:
        raise ValueError("The common training sample lacks a positive class")
    # Models share identical candidate endpoints and held-out evaluation endpoints.
    x = ({part: windows(values, length) for part, values in raw_x.items()}
         if kind == "lstm_autoencoder" else {part: values[length - 1:] for part, values in raw_x.items()})
    x_fit = np.ascontiguousarray(x["train"][fit_indices])
    y_fit = aligned_y["train"][fit_indices] if supervised else None
    valid_normal_mask = normal_windows["validation"]
    if valid_normal_mask.sum() < 20:
        raise ValueError("Too few normal validation windows to calibrate a threshold")
    x_validation_normal = np.ascontiguousarray(x["validation"][valid_normal_mask])
    # Bound only the reconstruction-loss validation set, not threshold calibration or metric evaluation.
    x_early_stop = x_validation_normal[:min(2000, len(x_validation_normal))]
    np.savez_compressed(output / "fit_indices.npz", common_endpoints=chosen + length - 1,
                        fitted_endpoints=fit_indices + length - 1)
    candidates = config["models"][kind]
    has_validation_positives = bool(aligned_y["validation"].sum())
    if not has_validation_positives:
        candidates = candidates[:1]  # Predeclared first candidate, not selected using the held-out test.
    best_quality = -np.inf
    selected = None
    tuning_records = []
    tuning_start = time.perf_counter()
    for candidate in candidates:
        print(f"{metadata['dataset']} seed={seed} {kind}: {candidate}", flush=True)
        model = (LSTMAutoencoder(candidate, seed, config["threads"], x_fit.shape[1:])
                 if kind == "lstm_autoencoder" else TabularDetector(kind, candidate, seed, config["threads"]))
        with MemoryMonitor() as memory:
            start = time.perf_counter()
            model.fit(x_fit, y_fit, validation=x_early_stop)
            elapsed = time.perf_counter() - start
        scores = model.score(x["validation"])
        if not np.isfinite(scores).all():
            raise ValueError("Non-finite validation scores")
        threshold = float(np.quantile(scores[valid_normal_mask], 1 - config["normal_fpr_target"]))
        validation_metrics = metrics(aligned_y["validation"], scores, threshold)
        quality = validation_metrics["f1"] if has_validation_positives else 0.0
        record = {"parameters": candidate, "threshold": threshold,
                  "validation": validation_metrics, "fit_seconds": elapsed,
                  "peak_rss_mb": memory.peak / 1024**2,
                  "incremental_peak_rss_mb": (memory.peak - memory.baseline) / 1024**2,
                  "lifetime_peak_rss_mb": memory.lifetime_peak / 1024**2}
        tuning_records.append(record)
        if quality > best_quality:
            best_quality = quality
            model.threshold = threshold
            model.save(output / "model")
            selected = record.copy()
            if kind == "lstm_autoencoder":
                write_json(output / "training_history.json", model.history)
        del model
        gc.collect()
    tuning_seconds = time.perf_counter() - tuning_start
    # Reload the selected frozen model; no test-based model or threshold choice.
    if kind == "lstm_autoencoder":
        import tensorflow as tf
        model = object.__new__(LSTMAutoencoder)
        model.model = tf.keras.models.load_model(output / "model.keras", compile=False)
        model.threshold = selected["threshold"]
    else:
        import joblib
        model = joblib.load(output / "model.joblib")
    test_scores, latency, timings = benchmark_score(model, x["test"])
    if not np.isfinite(test_scores).all():
        raise ValueError("Non-finite test scores")
    test_y = aligned_y["test"]
    threshold = selected["threshold"]
    result = {
        "dataset": metadata["dataset"], "model": kind, "model_name": NAMES[kind],
        "seed": seed, "status": "complete", "parameters": selected["parameters"],
        "threshold": threshold, "threshold_policy": "95th percentile of scores on entirely normal validation windows; predict positive only above threshold",
        "selection_policy": "Maximum validation F1" if has_validation_positives else "First predefined candidate: validation contains no positives",
        "fit_rows": len(fit_indices), "fit_normal": int(normal_windows["train"][fit_indices].sum()),
        "test_rows": len(test_y), "feature_count": metadata["feature_count"],
        "train_seconds": selected["fit_seconds"], "tuning_seconds": tuning_seconds,
        "latency_ms_per_observation": latency, "inference_repetition_seconds": timings,
        "peak_rss_mb": selected["peak_rss_mb"], "incremental_peak_rss_mb": selected["incremental_peak_rss_mb"],
        "lifetime_peak_rss_mb": selected["lifetime_peak_rss_mb"],
        "validation_f1": selected["validation"]["f1"], **metrics(test_y, test_scores, threshold),
    }
    prediction = (test_scores > threshold).astype(np.int8)
    pd.DataFrame({"row_id": prepared["id_test"][length - 1:], "label": test_y,
                  "score": test_scores, "prediction": prediction}).to_csv(output / "predictions.csv", index=False)
    write_json(output / "tuning.json", tuning_records)
    write_json(output / "result.json", result)
    f1_text = "not applicable" if result["f1"] is None else f"{result['f1']:.4f}"
    print(f"Completed: F1={f1_text}, fit={result['train_seconds']:.2f}s", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    run(args.dataset, args.output, config, args.model, args.seed)


if __name__ == "__main__":
    main()


```

## A.5 Threshold Calibration

The original worker listing above produces the preserved percentile-threshold baseline. The full follow-up implementation is supplied in scripts/recalibrate_results.py. It first saves validation-selected thresholds for every model, then applies those frozen decisions to unchanged test scores. The normal-only fallback, tie handling, score persistence and baseline link are recorded in calibration.json and run.json. The full project launcher executes both stages.

# APPENDIX B

# SAMPLE SCREENSHOTS

These screenshots were captured from the running local Streamlit application using saved research outputs. They are interface evidence, not design mock-ups. The overview distinguishes the two label semantics, and the model comparison displays detection and resource measurements. The interface supports CSV exports for further inspection.

![Figure B.1 Research overview with completed evaluations and separate dataset comparisons](../artifacts/screenshots/01_overview.png)

![Figure B.2 Dataset-specific detection and computational results](../artifacts/screenshots/02_model_comparison.png)

![Figure B.3 Prediction explorer showing scores and misclassified observations](../artifacts/screenshots/03_prediction_explorer.png)

![Figure B.4 Dataset partition counts and interpretation notes](../artifacts/screenshots/04_dataset_details.png)

# APPENDIX C

# DATASET DETAILS

## C.1 Provenance and Licence

Google sample: Derrick Mwiti, Google 2019 Cluster sample, Kaggle version 1. Source: https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample. The Kaggle listing identifies CC BY-NC-SA 4.0. The upstream Google trace documentation identifies CC BY 4.0. Both provenance layers are recorded; the derivative sample should not be assumed to have identical licensing or preparation guarantees to the upstream trace.

SMD source: NetManAIOps OmniAnomaly repository, revision 7fb0e0acf89ea49908896bcc9f9e80fcfff6baf4. Source: https://github.com/NetManAIOps/OmniAnomaly. The repository includes an MIT licence. No separate dataset-specific licence was identified in the downloaded source documentation. The experiment downloads train, test and test_label files for each selected machine, but fits only the declared re-partitioned labelled sequence.

## C.2 Feature Dictionary

Table C.1 Google feature definitions

| Feature | Meaning and unit |
| --- | --- |
| average_cpu and maximum_cpu | Mean and maximum CPU usage in the usage window; normalised trace CPU units |
| average_memory and maximum_memory | Mean and maximum memory usage; normalised trace memory units |
| sample_cpu | CPU usage from the supplied random sample; normalised units |
| requested_cpu and requested_memory | Requested CPU and memory parsed from resource_request; normalised units |
| assigned_memory | Assigned memory limit in normalised trace units |
| page_cache_memory | Memory used for page cache in normalised trace units |
| cycles_per_instruction | Supplied cycles-per-instruction ratio |
| memory_accesses_per_instruction | Supplied memory-accesses-per-instruction ratio |

SMD metric_01 through metric_38 preserve the 38 source columns without assigning invented semantic names. Their rolling_mean features are trailing five-observation means and their delta features are first differences. Absolute timestamps, physical channel units and sampling interval are not encoded in these downloaded files. Observation indices therefore identify ordering rather than elapsed wall-clock time.

## C.3 Labels and Exclusions

SMD label 1 denotes a published point anomaly and label 0 denotes a normal-labelled point. Google label 1 is the supplied failed flag; its exact agreement with event=FAIL was checked. Google outcome fields, identifiers and timestamps are excluded from the 11-feature predictor set. The source sample does not supply verified disk or network throughput channels. Synthetic anomalies were not substituted for genuine labels.

Google chronological boundaries in source microseconds are 1504505000000 and 2124000000000. Rows with a timestamp equal to a boundary enter the later partition. Entire tasks appearing on both sides of a boundary are purged. The final sample contains 242,594 records after deduplication and purging.

Table C.2 SMD chronological boundaries in zero-based observation positions

| Machine | Training | Validation | Final test |
| --- | --- | --- | --- |
| 1-1 | [0, 17087) | [17087, 22783) | [22783, 28479) |
| 2-1 | [0, 14216) | [14216, 18955) | [18955, 23694) |
| 3-1 | [0, 17220) | [17220, 22960) | [22960, 28700) |

Intervals are left-inclusive and right-exclusive. Within each SMD partition, the first eligible endpoint is its 30th observation. Overlapping windows never cross a partition or machine boundary. Sampled training endpoint indices are retained for every fitted model and seed.

## C.4 Missing Values and Preparation Records

Table C.3 Missing values in retained raw predictors before imputation

| Dataset | Partition | Feature | Missing |
| --- | --- | --- | --- |
| Google | train | maximum_memory | 2,995 |
| Google | train | requested_cpu | 383 |
| Google | train | requested_memory | 383 |
| Google | train | cycles_per_instruction | 40,366 |
| Google | train | memory_accesses_per_instruction | 40,366 |
| Google | validation | maximum_memory | 800 |
| Google | validation | requested_cpu | 280 |
| Google | validation | requested_memory | 280 |
| Google | validation | cycles_per_instruction | 11,246 |
| Google | validation | memory_accesses_per_instruction | 11,246 |
| Google | test | maximum_memory | 692 |
| Google | test | requested_cpu | 103 |
| Google | test | requested_memory | 103 |
| Google | test | cycles_per_instruction | 15,575 |
| Google | test | memory_accesses_per_instruction | 15,575 |

The table counts missing values after Google deduplication and task purging, before the training-fitted median is applied. Complete preparation details are stored in datasets/<dataset>/dataset.json, and the fitted transformation is stored in preprocessor.joblib. No transformation is fitted to the final evaluation set.

## C.5 File Integrity

The following SHA-256 values identify the raw inputs used for this run. The complete machine-readable manifest also records source URLs and byte sizes. An exact match verifies file identity, not correctness of the original labels.

data\raw\google\borg_traces_data.csv (328,310,714 bytes)

c3362a78555b7bd10d27a693da4e4c7230ac323359a5bfe0e75340c27b416b0d

data\raw\smd\train\machine-1-1.txt (9,739,818 bytes)

ecebd2933ccdc1126e2abbfb538762dbde16410cebc77257a7b96e1e1fbcf753

data\raw\smd\test\machine-1-1.txt (9,739,818 bytes)

bdb801d8d52ce6b0c4a20c310edf31b5cee9bc82eb84fb38e7d4a19309b16ddb

data\raw\smd\test_label\machine-1-1.txt (56,958 bytes)

cddf33441ae480c6074e173c897d4715eb893f24a7078d3d3e1b79b200c7cd19

data\raw\smd\train\machine-2-1.txt (8,103,006 bytes)

d6eb13ff74a537cf33686319fffdd4cdcb394862570a3acd519b4c37af97cd2e

data\raw\smd\test\machine-2-1.txt (8,103,348 bytes)

2d103661799271be958227907950a1c76803356442d60a4070671cc0abbefb20

data\raw\smd\test_label\machine-2-1.txt (47,388 bytes)

416dd20a447f38b8d9394382293ed44f509c1c3f99c4aa2fbff179498626dbdc

data\raw\smd\train\machine-3-1.txt (9,815,400 bytes)

c1186ac085809c0590467dd42f4f12180012ebd994c1e615250dbed2a3588111

data\raw\smd\test\machine-3-1.txt (9,815,400 bytes)

b110b204ba4919250351eba7aa253d73ab54af0cf18e272b9b284de5fa7df88d

data\raw\smd\test_label\machine-3-1.txt (57,400 bytes)

549348a9c63ed922e85157885bcb21289a8c9f5af94b50147a21cb7b928220ec

# References for the Implementation

Google. (n.d.). ClusterData 2019 traces and trace format v3. https://github.com/google/cluster-data/blob/master/ClusterData2019.md

Google. Cluster trace format definitions. https://github.com/google/cluster-data/blob/master/clusterdata_trace_format_v3.proto

Mwiti, D. (2022). Google 2019 Cluster sample, version 1 [Data set]. Kaggle. https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample

NetManAIOps. OmniAnomaly and Server Machine Dataset [Repository and data]. https://github.com/NetManAIOps/OmniAnomaly

Scikit-learn developers. Common pitfalls and recommended practices. https://scikit-learn.org/stable/common_pitfalls.html

Scikit-learn developers. Tuning the decision threshold for class prediction. https://scikit-learn.org/stable/modules/classification_threshold.html

Scikit-learn developers. OneClassSVM API reference. https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html

TensorFlow developers. Enable operation determinism. https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism

psutil developers. Process memory information. https://psutil.readthedocs.io/
