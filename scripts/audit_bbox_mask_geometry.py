from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
from pathlib import Path
import sys
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from autolabel.qc_agent import resolve_local_uri  # noqa: E402


def box_values(box: Any) -> tuple[int, int, int, int] | None:
    if isinstance(box, dict):
        values = [box.get(key) for key in ("x1", "y1", "x2", "y2")]
    elif isinstance(box, (list, tuple)) and len(box) == 4:
        values = list(box)
    else:
        return None
    try:
        x1, y1, x2, y2 = (int(round(float(value))) for value in values)
    except (TypeError, ValueError):
        return None
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def resolve_mask_path(
    metadata_path: Path,
    mask_uri: Any,
    asset_base_dirs: list[Path] | None = None,
) -> Path | None:
    if not mask_uri:
        return None
    resolved, exists, _ = resolve_local_uri(
        str(mask_uri),
        metadata_path,
        asset_base_dirs,
    )
    if exists and resolved is not None:
        return resolved.resolve()

    uri_path = Path(str(mask_uri))
    candidates = [
        uri_path if uri_path.is_absolute() else Path.cwd() / uri_path,
        metadata_path.parent.parent / "masks" / uri_path.name,
        metadata_path.parent / uri_path.name,
    ]
    return next(
        (candidate.resolve() for candidate in candidates if candidate.is_file()), None
    )


def bbox_iou(
    first: tuple[int, int, int, int], second: tuple[int, int, int, int]
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1)
    )
    first_area = (ax2 - ax1) * (ay2 - ay1)
    second_area = (bx2 - bx1) * (by2 - by1)
    union = first_area + second_area - intersection
    return intersection / union if union else 0.0


def inspect_object(
    metadata_path: Path,
    sample: dict[str, Any],
    obj: dict[str, Any],
    asset_base_dirs: list[Path] | None = None,
) -> dict[str, Any]:
    detail = (
        obj.get("geometry_detail")
        if isinstance(obj.get("geometry_detail"), dict)
        else {}
    )
    mask_uri = detail.get("mask_uri")
    mask_path = resolve_mask_path(metadata_path, mask_uri, asset_base_dirs)
    bbox = box_values(obj.get("box"))
    record: dict[str, Any] = {
        "sample_id": sample.get("sample_id") or metadata_path.stem,
        "object_id": obj.get("object_id") or "object",
        "metadata_path": str(metadata_path),
        "mask_uri": mask_uri,
        "mask_path": str(mask_path) if mask_path else "",
        "mask_exists": mask_path is not None,
        "object_bbox": json.dumps(bbox),
        "final_bbox_source": (detail.get("generation_params") or {}).get(
            "final_bbox_source"
        ),
        "issue_flags": [],
    }
    if bbox is None:
        record["issue_flags"].append("invalid_object_bbox")
    if mask_path is None:
        record["issue_flags"].append("mask_missing")
        return record

    with Image.open(mask_path) as source:
        mask = np.asarray(source.convert("L")) > 0
    record["mask_width"] = int(mask.shape[1])
    record["mask_height"] = int(mask.shape[0])
    expected_width = sample.get("image_asset", {}).get("width")
    expected_height = sample.get("image_asset", {}).get("height")
    record["mask_size_matches_image"] = (
        expected_width is None
        or expected_height is None
        or (int(expected_width), int(expected_height)) == (mask.shape[1], mask.shape[0])
    )
    if not record["mask_size_matches_image"]:
        record["issue_flags"].append("mask_size_mismatch")

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        record["mask_nonzero_area"] = 0
        record["issue_flags"].append("mask_empty")
        return record
    mask_bbox = (int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1))
    record["mask_bbox"] = json.dumps(mask_bbox)
    record["mask_nonzero_area"] = int(mask.sum())
    if bbox is None:
        return record

    x1, y1, x2, y2 = bbox
    clipped = mask[
        max(0, y1) : min(mask.shape[0], y2), max(0, x1) : min(mask.shape[1], x2)
    ]
    inside_area = int(clipped.sum())
    bbox_area = (x2 - x1) * (y2 - y1)
    record["mask_inside_bbox_ratio"] = inside_area / record["mask_nonzero_area"]
    record["mask_fill_ratio"] = inside_area / bbox_area if bbox_area else 0.0
    record["mask_bbox_iou"] = bbox_iou(bbox, mask_bbox)
    record["mask_bbox_exact_match"] = bbox == mask_bbox
    if record["mask_inside_bbox_ratio"] < 0.999:
        record["issue_flags"].append("mask_pixels_outside_object_bbox")
    record["issue_flags"] = ";".join(record["issue_flags"])
    return record


def load_rows(
    metadata_dirs: list[Path],
    asset_base_dirs: list[Path] | None = None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for metadata_dir in metadata_dirs:
        for metadata_path in sorted(metadata_dir.glob("*.json")):
            sample = json.loads(metadata_path.read_text(encoding="utf-8"))
            for obj in sample.get("objects") or []:
                if isinstance(obj, dict):
                    rows.append(
                        inspect_object(
                            metadata_path,
                            sample,
                            obj,
                            asset_base_dirs,
                        )
                    )
    return rows


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    fields = sorted({field for row in rows for field in row})
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(rows: list[dict[str, Any]], path: Path) -> None:
    found = [row for row in rows if row.get("mask_exists")]
    nonempty = [row for row in found if (row.get("mask_nonzero_area") or 0) > 0]
    exact = [row for row in nonempty if row.get("mask_bbox_exact_match")]
    size_match = [row for row in found if row.get("mask_size_matches_image")]
    issue_counts = Counter(
        flag
        for row in rows
        for flag in str(row.get("issue_flags") or "").split(";")
        if flag
    )
    fill_ratios = [
        float(row["mask_fill_ratio"])
        for row in nonempty
        if row.get("mask_fill_ratio") is not None
    ]
    exact_rate = len(exact) / len(nonempty) if nonempty else 0.0
    circular = exact_rate >= 0.95
    lines = [
        "# BBox-mask geometry audit",
        "",
        "| Metric | Count | Rate |",
        "|---|---:|---:|",
        f"| Objects | {len(rows)} | 100.0% |",
        (
            f"| Mask found | {len(found)} | {100 * len(found) / len(rows):.1f}% |"
            if rows
            else "| Mask found | 0 | n/a |"
        ),
        (
            f"| Non-empty mask | {len(nonempty)} | {100 * len(nonempty) / len(rows):.1f}% |"
            if rows
            else "| Non-empty mask | 0 | n/a |"
        ),
        (
            f"| Mask size matches image | {len(size_match)} | {100 * len(size_match) / len(rows):.1f}% |"
            if rows
            else "| Mask size matches image | 0 | n/a |"
        ),
        f"| Object bbox exactly equals mask nonzero bbox | {len(exact)} | {100 * exact_rate:.1f}% |",
        "",
        f"- mask_fill_ratio p10/p50/p90: `{np.percentile(fill_ratios, [10, 50, 90]).round(4).tolist() if fill_ratios else []}`",
        f"- issue_counts: `{json.dumps(issue_counts, sort_keys=True)}`",
        f"- mask_bbox_circularity_detected: `{str(circular).lower()}`",
        "",
        "## Interpretation",
        "",
    ]
    if circular:
        lines.extend(
            [
                "- The object bbox is effectively derived from the mask nonzero extent.",
                "- Mask-to-bbox IoU or containment is therefore a pipeline consistency check, not independent bbox quality evidence.",
                "- The mask may highlight generated change pixels for human review, but it must not auto-approve semantic correctness or visible-water containment.",
            ]
        )
    else:
        lines.append(
            "- Mask geometry is not uniformly identical to object bbox geometry; inspect the CSV issue flags by source pipeline."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit whether bbox and mask geometry provide independent evidence."
    )
    parser.add_argument("--metadata-dir", action="append", required=True)
    parser.add_argument("--asset-base-dir", action="append", default=[])
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    rows = load_rows(
        [Path(value) for value in args.metadata_dir],
        [Path(value) for value in args.asset_base_dir],
    )
    if not rows:
        raise SystemExit("no metadata objects found")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(rows, output_dir / "bbox_mask_geometry.csv")
    write_summary(rows, output_dir / "bbox_mask_geometry.md")
    print(output_dir / "bbox_mask_geometry.md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
