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
    parser.add_argument("--update-samples", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    paths = config.get("paths", {})
    metadata_dir = args.metadata_dir or paths.get("metadata_dir", "data/processed/metadata")
    output = args.output or paths.get("labelstudio_export", "data/exports/labelstudio/import.json")
    tasks = export_metadata_dir(metadata_dir, output, update_samples=args.update_samples)
    print(f"Wrote {len(tasks)} Label Studio tasks to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
