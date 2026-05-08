from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autolabel.adapters.detector_service import load_detector_config
from autolabel.model_config import build_detector_runtime_config
from autolabel.pipeline import load_pipeline_config, run_direct_pipeline


def main() -> int:
    parser = argparse.ArgumentParser(description="Run direct detection/segmentation plus crop classification branch.")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--pipeline-config", default="configs/autolabel.yaml")
    parser.add_argument("--detector-config", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--no-classify", action="store_true")
    args = parser.parse_args()

    pipeline_config = load_pipeline_config(args.pipeline_config)
    paths = pipeline_config.get("paths", {})
    detector_config = (
        load_detector_config(args.detector_config)
        if args.detector_config
        else build_detector_runtime_config(pipeline_config)
    )
    manifest = args.manifest or paths.get("image_sequence_manifest") or "data/staging/image_sequence/manifest.csv"
    output_root = args.output_root or str(Path(paths.get("metadata_dir", "data/processed/metadata")).parent)
    written = run_direct_pipeline(
        manifest_csv=manifest,
        pipeline_config=pipeline_config,
        detector_config=detector_config,
        output_root=output_root,
        classify=not args.no_classify,
    )
    print(f"Wrote {len(written)} AutoLabelSample metadata files.")
    for path in written:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
