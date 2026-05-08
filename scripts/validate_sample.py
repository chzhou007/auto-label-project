from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autolabel.utils import read_json
from autolabel.validators import ValidationError, validate_sample_contract


def validate_path(path: Path) -> tuple[bool, str]:
    try:
        validate_sample_contract(read_json(path))
        return True, "ok"
    except (ValidationError, ValueError, TypeError, KeyError) as exc:
        return False, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate AutoLabelSample metadata JSON files.")
    parser.add_argument("--sample", default=None)
    parser.add_argument("--metadata-dir", default=None)
    args = parser.parse_args()

    paths: list[Path] = []
    if args.sample:
        paths.append(Path(args.sample))
    if args.metadata_dir:
        paths.extend(sorted(Path(args.metadata_dir).glob("*.json")))
    if not paths:
        raise SystemExit("Provide --sample or --metadata-dir")

    failed = 0
    for path in paths:
        ok, message = validate_path(path)
        print(f"{path}: {message}")
        if not ok:
            failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
