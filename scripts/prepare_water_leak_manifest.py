from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autolabel.modules.generation.manifest_builder import write_water_leak_generation_manifest


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a water_leak generation manifest for batch I2I runs.")
    parser.add_argument("--input", default="data/staging/image_sequence/manifest.csv")
    parser.add_argument("--output", default="data/staging/image_sequence/water_leak_generation_1000.csv")
    parser.add_argument("--count", type=int, default=1000)
    parser.add_argument("--anomaly-type", default="water_leak")
    parser.add_argument("--object-type", default="leakage_area")
    parser.add_argument("--sample-prefix", default="water_leak")
    parser.add_argument("--allow-repeat", action="store_true")
    args = parser.parse_args()

    output = write_water_leak_generation_manifest(
        args.input,
        args.output,
        count=args.count,
        anomaly_type=args.anomaly_type,
        object_type=args.object_type,
        sample_prefix=args.sample_prefix,
        allow_repeat=args.allow_repeat,
    )
    print(f"Wrote {args.count} {args.anomaly_type} generation rows to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
