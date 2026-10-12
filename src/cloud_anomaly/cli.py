"""Reproducible command-line runner; no unit-test suite is included."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys

import psutil

from .data import acquire, prepare_google, prepare_smd, sha256, write_json


def run(root: Path, config_path: Path, resume: Path | None):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if abs(sum(config["split_fractions"]) - 1) > 1e-8 or any(x <= 0 for x in config["split_fractions"]):
        raise ValueError("Split fractions must be positive and sum to one")
    if config["sequence_length"] < 2 or config["max_fit_samples"] < 32:
        raise ValueError("Invalid sequence length or sample budget")
    if config.get("threshold_policy", "normal_percentile") not in {"normal_percentile", "validation_f1"}:
        raise ValueError("Unknown threshold policy")
    fingerprint = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    source_digest = hashlib.sha256()
    for file in sorted((root / "src/cloud_anomaly").glob("*.py")):
        source_digest.update(file.name.encode())
        source_digest.update(file.read_bytes())
    source_hash = source_digest.hexdigest()
    if resume:
        run_dir = resume.resolve()
        saved = json.loads((run_dir / "run.json").read_text(encoding="utf-8"))
        if saved["config_sha256"] != fingerprint or saved["source_sha256"] != source_hash:
            raise ValueError("Resume requires the exact original configuration and source code")
    else:
        run_dir = root / "artifacts/runs" / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + config["name"])
        run_dir.mkdir(parents=True, exist_ok=False)
        saved = {"run_id": run_dir.name, "started_utc": datetime.now(timezone.utc).isoformat(),
            "experiment_revision": config.get("experiment_revision", False),
            "comparison_run": config.get("comparison_run"),
            "config_sha256": fingerprint, "source_sha256": source_hash, "status": "running",
            "hardware": {"platform": platform.platform(), "processor": platform.processor(),
                "logical_cpus": psutil.cpu_count(), "physical_cpus": psutil.cpu_count(logical=False),
                "ram_gib": psutil.virtual_memory().total / 1024**3, "python": sys.version},
            "versions": {name: importlib.metadata.version(name) for name in
                ["numpy", "pandas", "scikit-learn", "tensorflow", "keras", "psutil", "streamlit"]},
            "limitations": ["Google LSTM is not applicable because task sequences are too short.",
                "SMD split is adapted for supervised training, not the published benchmark split.",
                "Seed variability does not quantify uncertainty across unseen environments.",
                "Memory is sampled process RSS, not exact per-model allocation; latency is warm batch amortised time."]}
        write_json(run_dir / "run.json", saved)
        write_json(run_dir / "config.json", config)
        (run_dir / "environment.txt").write_text(subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True), encoding="utf-8")
    print(f"RUN DIRECTORY: {run_dir}", flush=True)
    try:
        manifest = acquire(root, config)
        if resume and (run_dir / "data_manifest.json").exists():
            prior = json.loads((run_dir / "data_manifest.json").read_text(encoding="utf-8"))
            if prior != manifest:
                raise ValueError("Dataset files changed since the original run")
        write_json(run_dir / "data_manifest.json", manifest)
        datasets = ["google_cluster", *[f"smd_{m}" for m in config["smd_machines"]]]
        for dataset in datasets:
            directory = run_dir / "datasets" / dataset
            if not (directory / "dataset.json").exists():
                print(f"Preparing {dataset}", flush=True)
                if dataset == "google_cluster":
                    prepare_google(root, directory, config)
                else:
                    prepare_smd(root, directory, dataset.removeprefix("smd_"), config)
        env = os.environ.copy()
        env.update({"PYTHONHASHSEED": "42", "TF_CPP_MIN_LOG_LEVEL": "2", "CUDA_VISIBLE_DEVICES": "-1",
                    "TF_ENABLE_ONEDNN_OPTS": "0", "OMP_NUM_THREADS": str(config["threads"]),
                    "OPENBLAS_NUM_THREADS": str(config["threads"]), "MKL_NUM_THREADS": str(config["threads"])})
        for dataset in datasets:
            for kind in config["models"]:
                if dataset == "google_cluster" and kind == "lstm_autoencoder":
                    continue
                for seed in config["seeds"]:
                    output = run_dir / "models" / dataset / kind / str(seed)
                    if (output / "result.json").exists():
                        continue
                    output.mkdir(parents=True, exist_ok=True)
                    command = [sys.executable, "-u", "-m", "cloud_anomaly.worker", "--dataset", str(run_dir / "datasets" / dataset),
                        "--output", str(output), "--config", str(run_dir / "config.json"), "--model", kind, "--seed", str(seed)]
                    with (output / "execution.log").open("w", encoding="utf-8") as log:
                        process = subprocess.Popen(command, cwd=root, env=env, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, encoding="utf-8", errors="replace")
                        for line in process.stdout:
                            print(line, end="", flush=True)
                            log.write(line)
                            log.flush()
                        if process.wait():
                            raise RuntimeError(f"Model process failed; see {output / 'execution.log'}")
        from .reporting import generate
        results = generate(run_dir)
        saved.update({"status": "complete", "completed_utc": datetime.now(timezone.utc).isoformat(), "completed_fits": len(results)})
        write_json(run_dir / "run.json", saved)
        write_json(root / "artifacts/latest.json", {"run_dir": str(run_dir.relative_to(root))})
        print(f"Completed {len(results)} model/seed evaluations. Dashboard: run_frontend.bat", flush=True)
    except Exception as exc:
        saved.update({"status": "failed", "error": str(exc)})
        write_json(run_dir / "run.json", saved)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    execute = commands.add_parser("run")
    execute.add_argument("--config", type=Path, default=Path("configs/research.json"))
    execute.add_argument("--resume", type=Path)
    report = commands.add_parser("report")
    report.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[2]
    if args.command == "run":
        run(root, args.config.resolve(), args.resume)
    else:
        from .reporting import generate
        generate(args.run_dir.resolve())


if __name__ == "__main__":
    main()

