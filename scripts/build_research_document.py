"""Build the research continuation from completed experimental evidence.

Run with the document artifact Python runtime (python-docx and pandas).
The source Chapters One to Three are preserved. No metrics are hardcoded.
"""
from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import textwrap

import pandas as pd
from docx import Document
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor

ROOT = Path(__file__).resolve().parents[1]
SHORT = {"Isolation Forest": "Isolation Forest", "One-Class SVM": "One-Class SVM",
         "Random Forest": "Random Forest", "LSTM Autoencoder": "LSTM AE"}


class Manuscript:
    def __init__(self):
        self.doc = Document(ROOT / "Comparative_Analysis_Anomaly_Detection_Cloud_ML.docx")
        for element in list(self.doc.element.body):
            if element.tag != qn("w:sectPr"):
                self.doc.element.body.remove(element)
        self.markdown = []
        for name in ["Normal", "Title", "Subtitle", "Heading 1", "Heading 2", "Heading 3", "Caption"]:
            if name not in self.doc.styles:
                self.doc.styles.add_style(name, WD_STYLE_TYPE.PARAGRAPH)
            style = self.doc.styles[name]
            style.font.name = "Times New Roman"
            style.font.color.rgb = RGBColor(0, 0, 0)
            style.font.size = Pt(12)
            style.paragraph_format.space_after = Pt(6)
        normal = self.doc.styles["Normal"]
        normal.paragraph_format.line_spacing = 1.5
        normal.paragraph_format.widow_control = True
        for name in ["Heading 1", "Heading 2", "Heading 3"]:
            self.doc.styles[name].font.bold = True
            self.doc.styles[name].paragraph_format.keep_with_next = True
            self.doc.styles[name].paragraph_format.space_before = Pt(12)
        self.doc.styles["Title"].font.size = Pt(17)
        self.doc.styles["Title"].font.bold = True
        if "Research Code" not in self.doc.styles:
            self.doc.styles.add_style("Research Code", WD_STYLE_TYPE.PARAGRAPH)
        style = self.doc.styles["Research Code"]
        style.font.name = "Consolas"
        style.font.size = Pt(10)
        style.paragraph_format.line_spacing = 1.0
        style.paragraph_format.space_after = Pt(0)
        style.paragraph_format.widow_control = False
        style.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.LEFT
        for section in self.doc.sections:
            section.page_width, section.page_height = Inches(8.268), Inches(11.693)
            section.top_margin = section.bottom_margin = section.right_margin = Inches(1)
            section.left_margin = Inches(1.5)
            for paragraph in section.footer.paragraphs:
                paragraph.clear()
            footer = section.footer.paragraphs[0]
            footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
            field = OxmlElement("w:fldSimple")
            field.set(qn("w:instr"), "PAGE")
            footer._p.append(field)

    def heading(self, text, level=1, page=False):
        if page:
            self.doc.add_page_break()
        paragraph = self.doc.add_paragraph(text, f"Heading {level}")
        self.markdown.append("#" * level + " " + text + "\n")
        return paragraph

    def paragraph(self, text):
        paragraph = self.doc.add_paragraph(text)
        paragraph.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        self.markdown.append(text + "\n")
        return paragraph

    def table(self, caption, headers, rows, widths=None):
        self.doc.add_paragraph(caption, "Caption")
        self.markdown.append(caption + "\n")
        table = self.doc.add_table(rows=1, cols=len(headers))
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        table.autofit = False
        widths = widths or [5.768 / len(headers)] * len(headers)
        for column, width in zip(table.columns, widths):
            column.width = Inches(width)
        for cell, header in zip(table.rows[0].cells, headers):
            cell.text = str(header)
        for values in rows:
            for cell, value in zip(table.add_row().cells, values):
                cell.text = str(value)
        properties = table._tbl.tblPr
        borders = OxmlElement("w:tblBorders")
        for edge in ["top", "left", "bottom", "right", "insideH", "insideV"]:
            border = OxmlElement("w:" + edge)
            border.set(qn("w:val"), "single")
            border.set(qn("w:sz"), "4")
            border.set(qn("w:color"), "D9D9D9")
            borders.append(border)
        properties.append(borders)
        for row_index, row in enumerate(table.rows):
            if row_index == 0:
                repeat = OxmlElement("w:tblHeader")
                row._tr.get_or_add_trPr().append(repeat)
            for column_index, cell in enumerate(row.cells):
                cell.width = Inches(widths[column_index])
                cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
                props = cell._tc.get_or_add_tcPr()
                margin = OxmlElement("w:tcMar")
                for side in ["top", "bottom", "left", "right"]:
                    element = OxmlElement("w:" + side)
                    element.set(qn("w:w"), "85")
                    element.set(qn("w:type"), "dxa")
                    margin.append(element)
                props.append(margin)
                if row_index == 0:
                    shade = OxmlElement("w:shd")
                    shade.set(qn("w:fill"), "E7EDF5")
                    props.append(shade)
                for paragraph in cell.paragraphs:
                    paragraph.paragraph_format.line_spacing = 1.0
                    paragraph.paragraph_format.space_after = Pt(2)
                    paragraph.paragraph_format.space_before = Pt(2)
                    paragraph.alignment = WD_ALIGN_PARAGRAPH.LEFT if column_index == 0 else WD_ALIGN_PARAGRAPH.CENTER
                    for run in paragraph.runs:
                        run.font.name = "Times New Roman"
                        run.font.size = Pt(10)
                        run.font.bold = row_index == 0
        self.markdown.append("| " + " | ".join(headers) + " |\n| " + " | ".join(["---"] * len(headers)) + " |")
        self.markdown.extend("| " + " | ".join(map(str, row)) + " |" for row in rows)
        self.markdown.append("")
        self.doc.add_paragraph().paragraph_format.space_after = Pt(0)

    def figure(self, path, caption, width=5.65):
        paragraph = self.doc.add_paragraph()
        paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
        paragraph.paragraph_format.keep_with_next = True
        picture = paragraph.add_run().add_picture(str(path), width=Inches(width))
        picture._inline.docPr.set("descr", caption)
        self.doc.add_paragraph(caption, "Caption")
        self.markdown.append(f"![{caption}](../{path.relative_to(ROOT).as_posix()})\n")

    def code(self, relative):
        code = (ROOT / relative).read_text(encoding="utf-8")
        self.markdown.append(f"```python\n# {relative}\n{code}\n```\n")
        for number, line in enumerate(code.splitlines(), 1):
            chunks = textwrap.wrap(line, width=64, replace_whitespace=False,
                drop_whitespace=False, break_long_words=True, break_on_hyphens=False) or [""]
            for index, chunk in enumerate(chunks):
                self.doc.add_paragraph((f"{number:03d} " if index == 0 else "    ") + chunk, "Research Code")


def fmt(value, digits=4):
    return "N/A" if pd.isna(value) else f"{value:.{digits}f}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "docs/Research_Implementation_and_Results.docx")
    args = parser.parse_args()
    run_dir = args.run.resolve()
    run = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
    if run["status"] != "complete":
        raise ValueError("A completed research run is required before writing findings")
    config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    if config["name"] != "research":
        raise ValueError("The report requires the research profile, not the quick demonstration")
    metrics = pd.read_csv(run_dir / "metrics.csv")
    summary = pd.read_csv(run_dir / "summary.csv")
    revised = bool(run.get("calibration_revision"))
    comparison = pd.read_csv(run_dir / "calibration_comparison.csv") if revised else None
    datasets = {p.parent.name: json.loads(p.read_text(encoding="utf-8")) for p in sorted(run_dir.glob("datasets/*/dataset.json"))}
    manifest = json.loads((run_dir / "data_manifest.json").read_text(encoding="utf-8"))
    smd = summary[summary.family == "SMD adapted benchmark"].sort_values("f1", ascending=False)
    google = summary[summary.family == "Google failure proxy"].sort_values("f1", ascending=False)
    best_smd, best_google = smd.iloc[0], google.iloc[0]
    fastest_smd = smd.sort_values("train_seconds").iloc[0]
    m = Manuscript()
    m.markdown.append("# Comparative Analysis of Anomaly Detection for Cloud Infrastructure\n")
    title = m.doc.add_paragraph("Comparative Analysis of Anomaly Detection for Cloud Infrastructure Using Selected Machine Learning Algorithms", "Title")
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    m.paragraph("Implementation plan and research continuation")
    m.paragraph("Chapter Four, Chapter Five and Appendices A to C")
    m.paragraph("September 2026")
    m.paragraph(f"This document records an implemented offline experiment and its measured results. Evidence comes from run {run['run_id']}, completed on {run['completed_utc'][:10]}. The original Chapters One to Three are retained separately and unchanged. The methodological amendments below must accompany any later consolidation of the thesis.")
    m.paragraph("The primary comparison covers all four algorithms on three selected SMD machines. The Google Cluster 2019 Kaggle sample supports a secondary comparison of the three tabular models. Its task histories are too short for the specified LSTM; that model is reported as not applicable on Google.")
    m.heading("Implementation Plan", page=True)
    m.paragraph("The work was organised around data acquisition, controlled model implementation, experiment execution, evidence capture and research writing. No unit tests were included. Verification used input checks, dependency checks, end-to-end execution, review of saved predictions and manual inspection of the interface and document.")
    m.table("Plan 1 Work packages and deliverables", ["Work package", "Deliverable", "Evidence"], [
        ["Acquire and audit data", "Google sample and selected SMD files", "Version and SHA-256 manifest"],
        ["Prepare common inputs", "Training-only transformations and split records", "Saved preprocessing and dataset metadata"],
        ["Implement detectors", "Four configurable algorithms", "Model files and source code"],
        ["Run comparison", "Fixed validation and held-out evaluation", f"{len(metrics)} model and seed evaluations"],
        ["Present evidence", "Frontend and comparison figures", "Actual screenshots and exported metrics"],
        ["Write research", "Chapters Four and Five and appendices", "Results linked to saved run artifacts"],
    ], [1.3, 2.15, 2.318])
    m.paragraph("The first launcher, run_project.bat, creates or reuses the Python environment, installs pinned dependencies, downloads data and executes the configured experiment. The second, run_frontend.bat, opens the local results interface without training models. The optional quick profile checks the complete workflow on a smaller fitting budget before the research profile is used.")
    m.paragraph("Version control includes the source, configuration, planning material and dependency files. The .gitignore excludes Word files, virtual environments, Python caches, credentials, downloaded datasets and generated model artifacts. Exclusion from Git does not remove the delivered document or results from the local project directory.")
    m.heading("Methodology Decisions")
    m.paragraph("The Google sample is a sparse set of failure-labelled usage records. Duplicate removal and task continuity checks precede model selection. The sample is not converted into artificial temporal sequences. SMD supplies real ordered, labelled observations, but its official normal-only training data cannot directly train a two-class Random Forest. A declared chronological re-partition of the labelled sequence enables the intended supervised comparison. Consequently, the reported SMD results do not reproduce the original benchmark protocol.")
    m.paragraph("Dataset selection and chronological fractions were fixed before original model fitting. The original run used a normal-validation percentile threshold. The revised analysis retains those fitted models and selects decision thresholds from labelled validation data. Because the evaluation period had already been reviewed when this follow-up was proposed, the revised results are exploratory rather than an independent confirmation. No test labels are supplied to the threshold-selection routine. Chapter Five contains only the three requested sections.")

    m.heading("CHAPTER FOUR", page=True).alignment = WD_ALIGN_PARAGRAPH.CENTER
    m.heading("IMPLEMENTATION AND RESULTS").alignment = WD_ALIGN_PARAGRAPH.CENTER
    m.heading("4.1 Introduction", 2)
    m.paragraph("This chapter describes the implemented comparative framework and reports the results of its execution. The framework evaluates Isolation Forest, One-Class Support Vector Machine, Random Forest and an LSTM Autoencoder using common base features, aligned evaluation records and a fixed validation procedure. It addresses both detection effectiveness and computational cost. The evidence is limited to offline data and does not establish the performance of a deployed streaming service.")
    m.heading("4.2 Implementation Environment", 2)
    m.paragraph("The implementation used Python modules for acquisition, preprocessing, detector training, evaluation and reporting. Streamlit provided a local, read-only frontend for reviewing completed experiments. Models were executed sequentially in separate processes, avoiding contention among simultaneous model fits. TensorFlow was explicitly restricted to CPU execution.")
    m.table("Table 4.1 Recorded implementation environment", ["Component", "Recorded value"], [
        ["Operating system", run["hardware"]["platform"]],
        ["Processor", "Intel Core i5-8250U at nominal 1.60 GHz"],
        ["Physical and logical cores", f"{run['hardware']['physical_cpus']} physical and {run['hardware']['logical_cpus']} logical"],
        ["Installed memory", f"{run['hardware']['ram_gib']:.2f} GiB"],
        ["Python", run["hardware"]["python"].split()[0]],
        ["NumPy and Pandas", run["versions"]["numpy"] + " and " + run["versions"]["pandas"]],
        ["Scikit-learn", run["versions"]["scikit-learn"]],
        ["TensorFlow and Keras", run["versions"]["tensorflow"] + " and " + run["versions"]["keras"]],
        ["Streamlit", run["versions"]["streamlit"]],
        ["Compute threads", str(config["threads"])],
        ["Random seeds", ", ".join(map(str, config["seeds"]))],
    ], [1.8, 3.968])
    m.paragraph("The exact installed environment is saved with the run. Random seeds and deterministic TensorFlow operations were set, but these controls do not imply bitwise equality across every hardware and library combination. Timings reflect this machine and its operating conditions, not a universal speed ranking.")
    m.heading("4.3 Amendments to the Proposed Methodology", 2)
    m.paragraph("Implementation exposed several differences between the proposed design and the available evidence. First, fitted preprocessing must follow the split; fitting it before splitting would allow held-out information into training. Second, the LSTM requires windows while the other methods accept rows, so the comparison uses shared base features and common endpoint labels rather than literally identical input tensors. Third, Random Forest receives anomalous training labels, whereas the other detectors learn from normal windows. Their results therefore compare learning regimes as well as model families.")
    m.paragraph("The original threshold rule used the 95th percentile of normal validation scores. Its weak F1 despite strong SMD ranking metrics motivated a controlled calibration follow-up. The revised rule maximises F1 on labelled validation data when both classes are present; otherwise the original threshold remains unchanged. Model weights, source records and continuous test scores are held fixed so the effect of the decision threshold can be measured separately. No unit-test suite was created.")
    m.heading("4.4 Dataset Acquisition and Audit", 2)
    m.paragraph("Two data sources were used. The first was version 1 of Derrick Mwiti's Google 2019 Cluster sample on Kaggle, derived from Google's cluster traces. The second was the Server Machine Dataset distributed with OmniAnomaly at the pinned source revision recorded in Appendix C. The SMD experiment selected machine-1-1, machine-2-1 and machine-3-1 and treated them as separate series. This is a deliberately bounded subset, not a full evaluation of all SMD machines.")
    g = datasets["google_cluster"]
    m.paragraph(f"The Google file contained {g['source_rows']:,} rows and {g['source_features']} columns. Deduplication on cluster, collection, instance, start time and end time removed {g['duplicate_telemetry_rows_removed']:,} repeated records, leaving {g['unique_telemetry_rows']:,} distinct usage records. No conflicting failure label was found within a repeated key. A further {g['cross_partition_task_rows_purged']:,} rows were removed because their task appeared across chronological partition boundaries.")
    m.paragraph("The Google failed flag matched the FAIL event indicator for every source record. This is a failure proxy; observations labelled zero also include other event types and are not independently verified normal operation. Event fields and their numeric equivalents were excluded from predictors. The relationship between the sampled usage intervals and the event labels could not be reconstructed from documented join code, so these results describe retrospective discrimination and not prospective failure prediction.")
    m.paragraph("After deduplication, the longest Google task history contained four observations. The configured LSTM requires 30. Concatenating different tasks would create an artificial history, so the Google LSTM entry is not applicable. By contrast, SMD provides ordered multivariate records and point anomaly labels. Its metric columns are anonymised; assigning unsupported CPU, disk or network names to them would be misleading.")
    split_rows = []
    for key, info in datasets.items():
        for partition, counts in info["eligible_partitions"].items():
            split_rows.append([info["display_name"].replace("Google Cluster 2019 sample", "Google sample").replace("SMD machine-", "SMD "), partition, f"{counts['rows']:,}", f"{counts['positive']:,}"])
    m.table("Table 4.2 Eligible records after partitioning and sequence warm-up", ["Dataset", "Partition", "Records", "Positives"], split_rows, [1.65, 1.3, 1.4, 1.418])
    m.paragraph("SMD's published labelled test sequence was split 60/20/20 in time order. Its official training files were downloaded for provenance but were not used for fitting. The first 29 observations of each adapted partition were excluded from evaluation for every model. Table 4.2 records eligible endpoints before the training sample cap. The positive class means an anomaly in SMD and a failure-labelled record in Google.")
    m.heading("4.5 Preprocessing and Feature Engineering", 2)
    m.paragraph("The loader validated the required schema and binary labels. Infinite values were treated as missing. Features with no observed training value were excluded, median values were fitted on the training partition, and a StandardScaler was fitted on the resulting training features. Validation and test data used these fitted objects without refitting. The pipeline saved both the imputer and scaler.")
    m.paragraph("Google used 11 numeric resource features: average CPU and memory usage, maximum CPU and memory usage, sampled CPU usage, requested CPU and memory, assigned memory, page-cache memory, cycles per instruction and memory accesses per instruction. Identifiers, event outcomes and absolute timestamps were not predictors. No rolling statistics were created across unrelated Google records.")
    m.paragraph("SMD used its 38 supplied metrics, a causal five-observation mean for each metric, and the first difference of each metric, producing 114 features. Feature history restarted at each partition boundary. A sequence contained 30 consecutive observations from one machine and one partition. Its label was the final observation's label, and the LSTM score was mean squared reconstruction error at that final observation. A training sequence qualified as normal only if all 30 labels were zero.")
    m.heading("4.6 Implementation of the Algorithms", 2)
    m.paragraph("Each detector exposed fit, score, predict and save operations. Scores were oriented so that a larger value always indicated greater abnormality. For Isolation Forest and One-Class SVM, the library's normality score was negated. Random Forest used the predicted probability of the positive class. The LSTM produced reconstruction error. A shared threshold rule converted scores into predictions.")
    m.table("Table 4.3 Declared model search space", ["Model", "Candidate settings"], [
        ["Isolation Forest", "100 or 200 trees; maximum 256 samples per tree; normal-only fitting"],
        ["One-Class SVM", "RBF kernel; nu 0.01 or 0.05; gamma scale; 256 MB kernel cache"],
        ["Random Forest", "100 trees and depth 10, or 200 trees and depth 20; balanced class weights; minimum leaf size 2"],
        ["LSTM Autoencoder", "Two encoder and two decoder LSTM layers; latent width 8 or 16; Adam at 0.001; MSE; batch 64; maximum 15 epochs; patience 3"],
    ], [1.5, 4.268])
    m.paragraph("For each seed, up to 6,000 common training endpoints were sampled without replacement. Random Forest fitted both classes at these endpoints. The normal-only methods fitted the shared subset with entirely normal windows; consequently their fitted sample counts can be smaller. This cap controls laptop resource use and is part of the reported experimental design.")
    m.paragraph("The LSTM used an encoder with twice the latent width followed by the latent layer, a repeat-vector operation, two decoder layers and a time-distributed linear output. Training reconstructed the full window. Early stopping monitored reconstruction loss on up to 2,000 normal validation windows, restoring the best observed weights. Source listings appear in Appendix A.")
    m.heading("4.7 Experiment and Evaluation Procedure", 2)
    m.paragraph("The baseline fitted each candidate on training data and calibrated the 95th-percentile threshold on entirely normal validation windows. It retained the candidate with the highest validation F1, or the first predefined candidate when validation contained no positives. The revised analysis did not refit these models or search their hyperparameters again. It scored the same validation records using the frozen selected model, evaluated every distinct score boundary, and chose the boundary yielding maximum validation F1. Exact ties were resolved in favour of the higher threshold.")
    m.paragraph("For machine-3-1, validation contained no positive examples, so the revised procedure retained the original threshold. It did not infer an optimum from final test labels. All 45 threshold decisions were saved before revised test predictions were produced. Strict greater-than comparisons were handled consistently at tied scores. The labelled calibration step supplies supervision to the decision rule even for the normal-only detectors and must not be described as a wholly unsupervised experiment.")
    m.paragraph(f"The baseline completed {len(metrics)} model and seed evaluations. The follow-up reevaluated these frozen models without additional fits: three tabular Google models and four models for each of three SMD machines, repeated over seeds 42, 43 and 44. Every improved and worsened result is retained. The evaluation period had already informed the decision to investigate calibration; independent confirmation on new records is still required. Point-level metrics are reported without expanding detections to whole anomaly intervals.")
    m.paragraph("Precision, recall, F1, ROC-AUC, average precision and false-positive rate were computed. F1 remained the primary ranking criterion. Each SMD machine was first evaluated separately. For the SMD summary, machine scores were macro-averaged within a seed, then the mean and sample standard deviation across the three seeds were reported. Google scores were summarised across seeds directly. Seed standard deviation describes these repeated fits, not uncertainty across new cloud environments.")
    m.heading("4.8 Detection Results", 2)
    if revised:
        m.paragraph("Table 4.4 compares the original percentile rule with the validation-selected threshold. The underlying anomaly scores are identical, so ROC-AUC and average precision do not change. Any difference in F1, precision, recall or false-positive rate is caused by the operating threshold. A larger F1 can come at the cost of more alerts; both outcomes are reported.")
        m.table("Table 4.4 Effect of validation threshold calibration", ["Dataset", "Algorithm", "Baseline F1", "Revised F1", "Change"], [
            ["SMD" if row.family.startswith("SMD") else "Google", SHORT[row.model_name], fmt(row.f1_baseline), fmt(row.f1_revised), f"{row.f1_revised - row.f1_baseline:+.4f}"] for row in comparison.itertuples()
        ], [.65, 1.55, 1.2, 1.2, 1.168])
        regressions = comparison[comparison.f1_revised < comparison.f1_baseline]
        if not regressions.empty:
            descriptions = [f"{row.model_name} on {'SMD' if row.family.startswith('SMD') else 'Google'} ({row.f1_baseline:.4f} to {row.f1_revised:.4f})" for row in regressions.itertuples()]
            m.paragraph("Calibration did not improve every model. F1 decreased for " + "; ".join(descriptions) + ". These regressions are retained rather than selecting whichever threshold happened to perform best on the test period. The result shows that a validation optimum can fail to transfer to later observations.")
    for number, title, frame in [(5, "SMD macro-average detection performance", smd), (6, "Google failure-proxy detection performance", google)]:
        rows = [[SHORT[row.model_name], fmt(row.precision), fmt(row.recall), fmt(row.f1), fmt(row.roc_auc), fmt(row.false_positive_rate)] for row in frame.itertuples()]
        if title.startswith("Google"):
            rows.append(["LSTM AE", "N/A", "N/A", "N/A", "N/A", "N/A"])
        m.table(f"Table 4.{number} {title}", ["Algorithm", "Precision", "Recall", "F1", "ROC-AUC", "FPR"], rows, [1.52, .85, .8, .8, .95, .848])
    m.paragraph(f"On the adapted SMD comparison, {best_smd.model_name} had the highest mean F1 of {best_smd.f1:.4f}, with mean recall {best_smd.recall:.4f} and false-positive rate {best_smd.false_positive_rate:.4f}. On the Google failure-proxy task, {best_google.model_name} had the highest mean F1 of {best_google.f1:.4f}. These rankings are specific to the selected data, supervision and threshold policy. They do not establish one algorithm as best for every cloud workload.")
    google_test = g["eligible_partitions"]["test"]
    prevalence = google_test["positive"] / google_test["rows"]
    m.paragraph(f"A trivial Google rule that labels every record positive would obtain F1 {2 * prevalence / (1 + prevalence):.4f}, recall 1.0000 and false-positive rate 1.0000 on this final segment. It is included as an interpretation baseline, not a deployable detector. Revised Google results must be judged against this baseline and their false-alarm rate, not simply against the much lower original F1. The positive prevalence is {prevalence:.4f}.")
    m.table("Table 4.7 F1 variability and average precision", ["Dataset", "Algorithm", "Mean F1", "F1 SD", "Average precision"], [
        ["SMD" if row.family.startswith("SMD") else "Google", SHORT[row.model_name], fmt(row.f1), fmt(row.f1_sd), fmt(row.average_precision)] for row in summary.itertuples()
    ], [0.7, 1.5, 1.0, 1.0, 1.568])
    m.figure(run_dir / "figures/smd_comparison.png", "Figure 4.1 SMD macro-average F1 and false-positive rate with seed standard deviation")
    m.figure(run_dir / "figures/google_comparison.png", "Figure 4.2 Google failure-proxy F1 and false-positive rate with seed standard deviation")
    per_machine = metrics[metrics.dataset.str.startswith("smd_")].groupby(["dataset", "model_name"]).f1.mean().unstack()
    m.table("Table 4.8 Mean F1 by SMD machine", ["Machine", "Isolation Forest", "One-Class SVM", "Random Forest", "LSTM AE"], [
        [index.removeprefix("smd_machine-"), *[fmt(row[name]) for name in ["Isolation Forest", "One-Class SVM", "Random Forest", "LSTM Autoencoder"]]] for index, row in per_machine.iterrows()
    ], [0.7, 1.35, 1.3, 1.3, 1.118])
    m.paragraph("The machine-level table is essential because an aggregate can conceal very different operating conditions. The final machine-1-1 segment contains only seven positive points before warm-up exclusion, so a small change in detections can substantially alter its recall. Machine-3-1 has no positive validation points, limiting label-based model selection. Neither class imbalance nor these temporal shifts is corrected by averaging seeds.")
    m.figure(run_dir / "figures/smd_machine-2-1_confusion.png", "Figure 4.3 Confusion matrices for SMD machine-2-1 at seed 42")
    m.heading("4.9 Computational Results", 2)
    m.table("Table 4.9 Mean computational measurements", ["Dataset", "Algorithm", "Fit time seconds", "Latency ms per point", "Peak RSS MB"], [
        ["SMD" if row.family.startswith("SMD") else "Google", SHORT[row.model_name], fmt(row.train_seconds, 2), fmt(row.latency_ms_per_observation, 4), fmt(row.peak_rss_mb, 1)] for row in summary.itertuples()
    ], [0.65, 1.55, 1.12, 1.23, 1.218])
    m.paragraph(f"The lowest mean SMD fitting time was recorded by {fastest_smd.model_name} at {fastest_smd.train_seconds:.2f} seconds. Fitting time measures the selected candidate's fit call, excluding preprocessing, model construction and validation scoring. Total tuning time is preserved separately in the run files. These distinctions matter particularly for the neural model, whose framework initialisation and architecture construction are not included in fit time.")
    m.paragraph("Inference was warmed up and repeated three times over the held-out array during the original run; median time was divided by evaluated observations. Table 4.9 reuses those measurements because model weights and scoring computations did not change. The calibration follow-up did not retrain the models or remeasure their training cost. Its additional loading and calibration times are saved separately. The latency is batch-amortised scoring time, not single-request delay. Memory is sampled process RSS, including libraries and data, and can miss short allocation spikes.")
    m.heading("4.10 Frontend and Operational Verification", 2)
    m.paragraph("The interface presents an overview, dataset-specific model comparisons, a prediction explorer, dataset details and reproducibility metadata. Users can inspect confusion matrices and score curves, identify false positives and missed anomalies, and download results or predictions. It reads saved artifacts and neither retrains models automatically nor sends production alerts. Appendix B contains screenshots captured from this running application.")
    m.figure(ROOT / "artifacts/screenshots/01_overview.png", "Figure 4.4 Frontend overview of the completed comparative experiment")
    m.paragraph("Figure 4.4 shows the implemented dashboard with the completed evaluation count and separate SMD and Google comparisons. The two sources remain separate because their labels represent different targets. The run selector also preserves access to the baseline experiment.")
    m.figure(ROOT / "artifacts/screenshots/06_smd_comparison.png", "Figure 4.5 Frontend comparison of all four algorithms on SMD machine-2-1")
    m.paragraph("Figure 4.5 shows the comparison screen used to inspect algorithm metrics and export the underlying records. The interface presents the revised decision metrics alongside the recorded resource measurements, allowing the reader to assess detection quality and computation together.")
    m.paragraph("Verification included Python syntax compilation, dependency consistency, a quick end-to-end execution and the research run. Saved prediction identifiers and labels were checked for agreement across models, and confusion counts and F1 values were recalculated from the prediction files. The frontend was opened in a browser, its five views were visited, and a CSV download was exercised. These checks were performed without adding unit tests to the codebase.")
    m.heading("4.11 Discussion and Limitations", 2)
    isolation = smd[smd.model_name == "Isolation Forest"].iloc[0]
    lstm = smd[smd.model_name == "LSTM Autoencoder"].iloc[0]
    ocsvm = smd[smd.model_name == "One-Class SVM"].iloc[0]
    m.paragraph(f"The LSTM's mean F1 was {lstm.f1:.4f}, compared with {isolation.f1:.4f} for Isolation Forest. Its mean fitting time was approximately {lstm.train_seconds / isolation.train_seconds:.0f} times greater and its scoring time per point approximately {lstm.latency_ms_per_observation / isolation.latency_ms_per_observation:.0f} times greater. One-Class SVM's mean ROC-AUC was {ocsvm.roc_auc:.4f}, compared with {lstm.roc_auc:.4f} for the LSTM. These score-ranking values did not change after calibration, demonstrating that an improved operating point is distinct from improved learned discrimination.")
    m.paragraph(f"The highest-F1 Google model had mean recall {best_google.recall:.4f}, F1 {best_google.f1:.4f} and false-positive rate {best_google.false_positive_rate:.4f}. These measures must be considered together with the all-positive baseline. A threshold that increases recall by raising many additional alerts is not automatically an operational improvement. Unverified label timing, the sampled population and temporal distribution differences still limit deployment claims.")
    m.paragraph("The revised Google false-positive rates are very high, so its larger F1 values do not support practical deployment. On SMD, the higher-F1 Isolation Forest operating point also increases false alarms relative to its baseline, while the LSTM and One-Class SVM lose F1 after calibration. The improvement in this chapter is a more complete, auditable comparison and a measured calibration experiment; it is not evidence that all detector weaknesses have been solved.")
    m.paragraph("The experiment meets the implementation objective by providing a common orchestration and evaluation pipeline for four model families. It also demonstrates that fair comparison requires more than supplying the same table to every algorithm. The LSTM uses temporal context, Random Forest uses labelled anomalies, and the normal-only detectors learn a different training distribution. These differences are relevant to deployment choice and prevent attributing every performance difference solely to model architecture.")
    m.paragraph("The calibration follow-up addresses the mismatch between score ranking and binary decisions by selecting thresholds on validation labels. It does not guarantee a false-positive budget, and its validation optimum may transfer poorly when anomaly prevalence or score distributions change. Machine-3-1 remains on the original rule because labelled calibration is unsupported there. The follow-up was motivated by previously observed test performance; although test labels do not enter the calibration function, the revised findings should be treated as exploratory until confirmed on a new evaluation period.")
    m.paragraph("The Google sample narrows the study to a failure-proxy task and lacks sufficient within-task history for the temporal model. Its positive rate is not representative of the original production trace by default. The SMD adaptation uses only three machines, changes the original benchmark split and relies on supplied labels. Selected fitting caps, a small candidate grid and three seeds limit the range of conclusions. The experiment does not evaluate streaming drift, incident response, attack novelty, explanation quality, model serving or cost in a production cloud.")
    m.paragraph("Comparison with prior literature is therefore qualitative. The results can illustrate the trade-off between supervised discrimination, normal-pattern detection and reconstruction cost, but numerical claims of improvement over published Google or SMD studies would require matching their labels, splits, preprocessing, thresholding and evaluation definitions. This study makes no such superiority claim.")
    m.heading("4.12 Chapter Summary", 2)
    m.paragraph("The implemented framework produced repeatable experiment artifacts, measured detection and resource metrics, and a usable local interface. Dataset auditing required a documented SMD split adaptation and the exclusion of LSTM from the sparse Google sample. The recorded evidence supports conditional conclusions about the selected workloads and provides the basis for Chapter Five.")

    m.heading("CHAPTER FIVE", page=True).alignment = WD_ALIGN_PARAGRAPH.CENTER
    m.heading("SUMMARY CONCLUSION AND RECOMMENDATION").alignment = WD_ALIGN_PARAGRAPH.CENTER
    m.heading("5.1 Summary of Study and Findings", 2)
    m.paragraph("The study investigated the relative effectiveness and computational requirements of Isolation Forest, One-Class SVM, Random Forest and an LSTM Autoencoder for offline anomaly detection in cloud-related telemetry. A modular Python implementation provided data acquisition, training-only preprocessing, controlled model fitting, common metric computation and a local reporting frontend. Three seeds were used to expose variation among repeated fits.")
    m.paragraph("The primary four-model comparison used three SMD machines under an explicitly adapted chronological protocol that provided labelled training examples for Random Forest. A secondary experiment used the requested Google Cluster 2019 sample. Auditing that sample identified extensive repeated telemetry and task histories of at most four observations, so only the three tabular algorithms could be meaningfully evaluated there. Its failed flag represented a failure proxy rather than a comprehensive cloud anomaly label.")
    m.paragraph(f"The highest SMD macro-average F1 was {best_smd.f1:.4f}, achieved by {best_smd.model_name}. The highest Google F1 was {best_google.f1:.4f}, achieved by {best_google.model_name}. The findings also showed that results differed across SMD machines and that metric rankings must be considered alongside threshold behaviour and computational measurements. {fastest_smd.model_name} recorded the lowest mean SMD fitting time under the specified measurement boundary.")
    m.paragraph("A controlled follow-up retained the fitted models and revised decision thresholds using labelled validation data. The baseline was preserved and reported alongside the revision. This isolated the effect of calibration from learning and showed why a high score-ranking metric can coexist with weak F1 at a particular operating point. The revised evaluation reused a reviewed test period and therefore remains exploratory.")
    m.paragraph("The objectives were addressed by implementing the framework and algorithms, evaluating the stated detection and resource metrics, and identifying selection criteria grounded in the observed conditions. The four-model objective was fulfilled on SMD; the Google experiment remained a three-model supplementary comparison. The final source code, launchers, saved results and dataset documentation provide an auditable record of the implementation.")
    m.heading("5.2 Conclusion", 2)
    m.paragraph(f"Within this exploratory adapted SMD evaluation, {best_smd.model_name} ranked highest by mean F1 under the revised decision rule. This conclusion is limited to the selected machines, supervision, features, sample budget and CPU environment. It does not establish universal superiority or justify deployment. A new evaluation period is required to confirm that the apparent calibration gains generalise.")
    m.paragraph("The study concludes that dataset suitability and evaluation design are central to meaningful anomaly-detector comparison. Temporal models require genuine histories, supervised methods require representative anomaly labels, and validation choices influence the operational false-alarm trade-off. Resource measurements add necessary context, but offline batch results alone do not establish real-time monitoring capability. Preserving these qualifications makes the implementation a defensible basis for further investigation.")
    m.heading("5.3 Recommendation for Future Research", 2)
    m.paragraph("Future research should first broaden the temporal evidence. A verified contiguous extraction from the original Google trace should link usage windows to event outcomes with explicit prediction horizons and exclude post-outcome information. Such an extraction would permit a meaningful LSTM comparison and a clearer distinction between anomaly detection and failure prediction. Extending the SMD evaluation to all machines and adding independent infrastructure benchmarks would improve the evidence for generalisation.")
    m.paragraph("Further work should compare the present supervised adaptation with a separate normal-only benchmark protocol. Where anomaly labels are unavailable for fitting, Random Forest should either be omitted or trained under a clearly different, independently evaluated weak-supervision design. Sensitivity analyses should vary temporal window length, training budget, anomaly prevalence and threshold calibration while preserving a final untouched evaluation period.")
    m.paragraph("A subsequent deployment study should measure concept drift, event-level detection delay, false alerts per unit time and end-to-end serving latency. It should assess retraining frequency, model loading and preprocessing cost, memory under sustained operation and the effect of hardware acceleration. Repeated temporal holdouts, uncertainty estimates that account for serial dependence and inspection of failure cases would support stronger conclusions than seed repetition alone.")

    m.heading("APPENDIX A", page=True).alignment = WD_ALIGN_PARAGRAPH.CENTER
    m.heading("CODEBASE").alignment = WD_ALIGN_PARAGRAPH.CENTER
    m.paragraph("The complete runnable codebase is supplied in the project directory. This appendix maps every main component and reproduces the detector and evaluation modules. The acquisition, orchestration and frontend files remain available as source alongside these listings; the printed appendix does not replace the runnable files. Line wrapping below is for presentation only.")
    m.table("Table A.1 Codebase organisation", ["Path", "Responsibility"], [
        ["src/cloud_anomaly/data.py", "Downloads, provenance, schema checks, splitting, imputation and feature preparation"],
        ["src/cloud_anomaly/models.py", "Classical detectors and LSTM Autoencoder"],
        ["src/cloud_anomaly/worker.py", "Candidate fitting, threshold calibration, selection, prediction and persistence"],
        ["src/cloud_anomaly/evaluation.py", "Detection metrics, RSS sampling and inference measurements"],
        ["src/cloud_anomaly/cli.py", "Sequential isolated execution and resume checks"],
        ["src/cloud_anomaly/reporting.py", "Aggregated tables and figures"],
        ["app.py", "Local Streamlit results interface"],
        ["configs/research.json and quick.json", "Recorded research and verification settings"],
        ["requirements.txt and requirements-lock.txt", "Direct and exact installed dependency versions"],
        ["run_project.bat", "Environment setup, data acquisition and full execution"],
        ["run_frontend.bat", "Frontend launcher without retraining"],
        ["scripts/recalibrate_results.py", "Validation-only threshold revision with frozen models and preserved baseline"],
        [".gitignore", "Python, credentials, datasets, artifacts and Word exclusions"],
    ], [2.5, 3.268])
    m.heading("A.1 Execution Instructions", 2)
    m.paragraph("Install 64-bit Python 3.12 with the Python launcher, then run run_project.bat from the project directory. The optional command run_project.bat quick uses the smaller verification configuration. After completion, run run_frontend.bat and open http://localhost:8501. The launchers quote paths and return nonzero on failure. Internet access is required for uncached dependencies and datasets.")
    m.paragraph("Each run directory preserves its configuration, raw-file manifest, dependency snapshot, dataset preparation metadata, fitted models, per-record predictions and figures. A run can be resumed only when its source and configuration hashes match. The .gitignore deliberately excludes all .docx files, including this research document; retain or share the document separately from Git.")
    m.heading("A.2 Detector Implementations", 2)
    m.paragraph("Subsequent implementation maintenance strengthens input validation without changing the reported experimental results. Google records must contain telemetry keys and finite, ordered usage times, and the failed flag must agree with event=FAIL. SMD labels are validated before integer conversion. Calibration and metric calculation require non-empty, aligned one-dimensional labels and finite scores and thresholds. Single-class validation retains the baseline threshold, and an evaluation period with no positives reports F1 as not applicable without terminating the worker. These safeguards do not constitute a new training experiment or evidence of improved detection accuracy. Historical source hashes remain attached to the original runs; the listings below reflect the maintained implementation.")
    m.code("src/cloud_anomaly/models.py")
    m.heading("A.3 Metric and Resource Measurement Implementations", 2)
    m.code("src/cloud_anomaly/evaluation.py")
    m.heading("A.4 Experiment Worker", 2)
    m.code("src/cloud_anomaly/worker.py")
    m.heading("A.5 Threshold Calibration", 2)
    m.paragraph("The original worker listing above produces the preserved percentile-threshold baseline. The full follow-up implementation is supplied in scripts/recalibrate_results.py. It first saves validation-selected thresholds for every model, then applies those frozen decisions to unchanged test scores. The normal-only fallback, tie handling, score persistence and baseline link are recorded in calibration.json and run.json. The full project launcher executes both stages.")

    m.heading("APPENDIX B", page=True).alignment = WD_ALIGN_PARAGRAPH.CENTER
    m.heading("SAMPLE SCREENSHOTS").alignment = WD_ALIGN_PARAGRAPH.CENTER
    m.paragraph("These screenshots were captured from the running local Streamlit application using saved research outputs. They are interface evidence, not design mock-ups. The overview distinguishes the two label semantics, and the model comparison displays detection and resource measurements. The interface supports CSV exports for further inspection.")
    for name, caption in [
        ("01_overview.png", "Figure B.1 Research overview with completed evaluations and separate dataset comparisons"),
        ("02_model_comparison.png", "Figure B.2 Dataset-specific detection and computational results"),
        ("03_prediction_explorer.png", "Figure B.3 Prediction explorer showing scores and misclassified observations"),
        ("04_dataset_details.png", "Figure B.4 Dataset partition counts and interpretation notes"),
    ]:
        path = ROOT / "artifacts/screenshots" / name
        if not path.exists():
            raise FileNotFoundError(f"Actual screenshot required: {path}")
        m.figure(path, caption)

    m.heading("APPENDIX C", page=True).alignment = WD_ALIGN_PARAGRAPH.CENTER
    m.heading("DATASET DETAILS").alignment = WD_ALIGN_PARAGRAPH.CENTER
    m.heading("C.1 Provenance and Licence", 2)
    m.paragraph("Google sample: Derrick Mwiti, Google 2019 Cluster sample, Kaggle version 1. Source: https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample. The Kaggle listing identifies CC BY-NC-SA 4.0. The upstream Google trace documentation identifies CC BY 4.0. Both provenance layers are recorded; the derivative sample should not be assumed to have identical licensing or preparation guarantees to the upstream trace.")
    m.paragraph(f"SMD source: NetManAIOps OmniAnomaly repository, revision {manifest['smd_revision']}. Source: https://github.com/NetManAIOps/OmniAnomaly. The repository includes an MIT licence. No separate dataset-specific licence was identified in the downloaded source documentation. The experiment downloads train, test and test_label files for each selected machine, but fits only the declared re-partitioned labelled sequence.")
    m.heading("C.2 Feature Dictionary", 2)
    m.table("Table C.1 Google feature definitions", ["Feature", "Meaning and unit"], [
        ["average_cpu and maximum_cpu", "Mean and maximum CPU usage in the usage window; normalised trace CPU units"],
        ["average_memory and maximum_memory", "Mean and maximum memory usage; normalised trace memory units"],
        ["sample_cpu", "CPU usage from the supplied random sample; normalised units"],
        ["requested_cpu and requested_memory", "Requested CPU and memory parsed from resource_request; normalised units"],
        ["assigned_memory", "Assigned memory limit in normalised trace units"],
        ["page_cache_memory", "Memory used for page cache in normalised trace units"],
        ["cycles_per_instruction", "Supplied cycles-per-instruction ratio"],
        ["memory_accesses_per_instruction", "Supplied memory-accesses-per-instruction ratio"],
    ], [2.3, 3.468])
    m.paragraph("SMD metric_01 through metric_38 preserve the 38 source columns without assigning invented semantic names. Their rolling_mean features are trailing five-observation means and their delta features are first differences. Absolute timestamps, physical channel units and sampling interval are not encoded in these downloaded files. Observation indices therefore identify ordering rather than elapsed wall-clock time.")
    m.heading("C.3 Labels and Exclusions", 2)
    m.paragraph("SMD label 1 denotes a published point anomaly and label 0 denotes a normal-labelled point. Google label 1 is the supplied failed flag; its exact agreement with event=FAIL was checked. Google outcome fields, identifiers and timestamps are excluded from the 11-feature predictor set. The source sample does not supply verified disk or network throughput channels. Synthetic anomalies were not substituted for genuine labels.")
    m.paragraph(f"Google chronological boundaries in source microseconds are {g['split_time_boundaries_microseconds'][0]} and {g['split_time_boundaries_microseconds'][1]}. Rows with a timestamp equal to a boundary enter the later partition. Entire tasks appearing on both sides of a boundary are purged. The final sample contains {sum(x['rows'] for x in g['partitions'].values()):,} records after deduplication and purging.")
    m.table("Table C.2 SMD chronological boundaries in zero-based observation positions", ["Machine", "Training", "Validation", "Final test"], [
        [info["display_name"].replace("SMD machine-", ""), f"[0, {info['split_bounds_exclusive'][1]})", f"[{info['split_bounds_exclusive'][1]}, {info['split_bounds_exclusive'][2]})", f"[{info['split_bounds_exclusive'][2]}, {info['split_bounds_exclusive'][3]})"]
        for key, info in datasets.items() if key.startswith("smd_")
    ], [.9, 1.6, 1.6, 1.668])
    m.paragraph("Intervals are left-inclusive and right-exclusive. Within each SMD partition, the first eligible endpoint is its 30th observation. Overlapping windows never cross a partition or machine boundary. Sampled training endpoint indices are retained for every fitted model and seed.")
    m.heading("C.4 Missing Values and Preparation Records", 2)
    missing = []
    for key, info in datasets.items():
        for part, columns in info["raw_missing_values"].items():
            for column, count in columns.items():
                if count:
                    missing.append(["Google" if key == "google_cluster" else key.removeprefix("smd_"), part, column, f"{count:,}"])
    m.table("Table C.3 Missing values in retained raw predictors before imputation", ["Dataset", "Partition", "Feature", "Missing"], missing or [["All", "All", "None", "0"]], [.65, 1.05, 3.0, 1.068])
    m.paragraph("The table counts missing values after Google deduplication and task purging, before the training-fitted median is applied. Complete preparation details are stored in datasets/<dataset>/dataset.json, and the fitted transformation is stored in preprocessor.joblib. No transformation is fitted to the final evaluation set.")
    m.heading("C.5 File Integrity", 2)
    m.paragraph("The following SHA-256 values identify the raw inputs used for this run. The complete machine-readable manifest also records source URLs and byte sizes. An exact match verifies file identity, not correctness of the original labels.")
    for file in manifest["files"]:
        m.paragraph(file["path"] + f" ({file['bytes']:,} bytes)")
        p = m.doc.add_paragraph(file["sha256"], "Research Code")
        m.markdown.append(file["sha256"] + "\n")
    m.heading("References for the Implementation")
    references = [
        "Google. (n.d.). ClusterData 2019 traces and trace format v3. https://github.com/google/cluster-data/blob/master/ClusterData2019.md",
        "Google. Cluster trace format definitions. https://github.com/google/cluster-data/blob/master/clusterdata_trace_format_v3.proto",
        "Mwiti, D. (2022). Google 2019 Cluster sample, version 1 [Data set]. Kaggle. https://www.kaggle.com/datasets/derrickmwiti/google-2019-cluster-sample",
        "NetManAIOps. OmniAnomaly and Server Machine Dataset [Repository and data]. https://github.com/NetManAIOps/OmniAnomaly",
        "Scikit-learn developers. Common pitfalls and recommended practices. https://scikit-learn.org/stable/common_pitfalls.html",
        "Scikit-learn developers. Tuning the decision threshold for class prediction. https://scikit-learn.org/stable/modules/classification_threshold.html",
        "Scikit-learn developers. OneClassSVM API reference. https://scikit-learn.org/stable/modules/generated/sklearn.svm.OneClassSVM.html",
        "TensorFlow developers. Enable operation determinism. https://www.tensorflow.org/api_docs/python/tf/config/experimental/enable_op_determinism",
        "psutil developers. Process memory information. https://psutil.readthedocs.io/",
    ]
    for reference in references:
        m.paragraph(reference)
    m.doc.core_properties.title = "Comparative Analysis of Anomaly Detection for Cloud Infrastructure"
    m.doc.core_properties.subject = "Implementation and results with research appendices"
    m.doc.core_properties.author = ""
    output = args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    m.doc.save(output)
    (ROOT / "docs/research_manuscript.md").write_text("\n".join(m.markdown), encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
