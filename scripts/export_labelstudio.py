from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autolabel.config_loader import load_config
from autolabel.exporters.labelstudio import export_metadata_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Export AutoLabelSample metadata to Label Studio import JSON.")
    parser.add_argument("--config", default="configs/autolabel.yaml")
    parser.add_argument("--metadata-dir", default=None)
    parser.add_argument("--output", default=None)
    parser.add_argument("--generated-quality-gate", action="store_true")
    parser.add_argument("--no-generated-quality-gate", action="store_true")
    parser.add_argument("--rejected-report", default=None)
    parser.add_argument("--update-samples", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config.get("paths", {})
    export_cfg = config.get("export", {}) if isinstance(config.get("export"), dict) else {}
    quality_gate_cfg = (
        export_cfg.get("generated_quality_gate", {})
        if isinstance(export_cfg.get("generated_quality_gate"), dict)
        else {}
    )
    generated_quality_gate = bool(quality_gate_cfg.get("enabled", False))
    if args.generated_quality_gate:
        generated_quality_gate = True
    if args.no_generated_quality_gate:
        generated_quality_gate = False
    metadata_dir = args.metadata_dir or paths.get("metadata_dir", "data/processed/metadata")
    output = args.output or paths.get("labelstudio_export", "data/exports/labelstudio/import.json")
    rejected_report = args.rejected_report or quality_gate_cfg.get("rejected_report")
    tasks = export_metadata_dir(
        metadata_dir,
        output,
        update_samples=args.update_samples,
        generated_quality_gate=generated_quality_gate,
        rejected_report_path=rejected_report,
    )
    print(f"Wrote {len(tasks)} Label Studio tasks to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
