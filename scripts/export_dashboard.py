"""Export only the saved files read by the dashboard, without fitted models or raw data."""
from pathlib import Path
import json
import shutil

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "artifacts/runs"
destination = ROOT / "dashboard_data/runs"
patterns = ["run.json", "config.json", "data_manifest.json", "*comparison.csv",
            "datasets/*/dataset.json", "models/*/*/*/result.json",
            "models/*/*/*/predictions.csv", "figures/*.png"]

for run in sorted(source.iterdir()):
    manifest = run / "run.json"
    if not manifest.exists() or json.loads(manifest.read_text(encoding="utf-8")).get("status") != "complete":
        continue
    # Keep the revised research run and its comparison baseline for public viewing.
    if run.name not in {"20260928T092901Z_calibrated", "20260930T155937Z_research_improved"}:
        continue
    for pattern in patterns:
        for item in run.glob(pattern):
            target = destination / run.name / item.relative_to(run)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target)
    print(f"Exported {run.name}")
