from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Any

from .constants import IMAGE_SUFFIXES
from .utils import get_image_size, now_iso_shanghai, read_csv, read_json, write_csv, write_json


ASSET_QC_FIELDS = [
    "asset_kind",
    "status",
    "issue_codes",
    "issue_messages",
    "asset_uri",
    "resolved_asset_uri",
    "file_name",
    "file_size_bytes",
    "width",
    "height",
    "aspect_ratio",
    "sample_id",
    "image_id",
    "source_type",
    "source_image_uri",
    "object_type",
    "object_index",
]


DEFAULT_ASSET_QC_RULES = {
    "min_image_width": 1,
    "min_image_height": 1,
    "min_crop_width": 4,
    "min_crop_height": 4,
    "max_crop_aspect_ratio": 20.0,
    "warn_unmatched_crop_source": True,
}


def _issue(code: str, message: str, severity: str = "error") -> dict[str, str]:
    return {"code": code, "message": message, "severity": severity}


def _status(issues: list[dict[str, str]]) -> str:
    if any(issue.get("severity") == "error" for issue in issues):
        return "failed"
    if issues:
        return "needs_human_review"
    return "passed"


def _is_image(path: Path) -> bool:
    return path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES


def scan_image_paths(paths: list[str | Path]) -> list[Path]:
    found: list[Path] = []
    for raw_path in paths:
        path = Path(raw_path)
        if path.is_file() and _is_image(path):
            found.append(path)
        elif path.is_dir():
            found.extend(sorted(child for child in path.rglob("*") if _is_image(child)))
    return sorted(dict.fromkeys(found))


def _sample_id_from_name(path: Path, pattern: str) -> str | None:
    match = re.match(pattern, path.name, flags=re.I)
    return match.group(1) if match else None


def _add_sample_id(target: set[str], path: Path, pattern: str) -> None:
    sample_id = _sample_id_from_name(path, pattern)
    if sample_id:
        target.add(sample_id)


def review_generated_inventory(
    *,
    image_files: list[Path],
    crop_files: list[Path],
    metadata_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Check that VLM/generated anomaly output directories form complete sample sets."""

    inventory: dict[str, set[str]] = {
        "generated_images": set(),
        "metadata": set(),
        "crops": set(),
        "masks": set(),
        "grid_previews": set(),
    }

    for path in image_files:
        parent = path.parent.name.lower()
        if parent == "generated_images":
            _add_sample_id(inventory["generated_images"], path, r"^(sample_\d+)\.[^.]+$")
        elif parent == "masks":
            _add_sample_id(inventory["masks"], path, r"^(sample_\d+)_obj_.+")
        elif parent == "grid_previews":
            _add_sample_id(inventory["grid_previews"], path, r"^(sample_\d+)_grid\.[^.]+$")

    for path in crop_files:
        if path.parent.name.lower() == "crops":
            _add_sample_id(inventory["crops"], path, r"^(sample_\d+)_obj_.+")

    if metadata_dir:
        for path in sorted(Path(metadata_dir).glob("*.json")):
            _add_sample_id(inventory["metadata"], path, r"^(sample_\d+)\.json$")

    generated = inventory["generated_images"]
    checks: list[dict[str, Any]] = []
    if generated:
        expected = {
            "metadata": inventory["metadata"],
            "crops": inventory["crops"],
            "masks": inventory["masks"],
            "grid_previews": inventory["grid_previews"],
        }
        for kind, observed in expected.items():
            missing = sorted(generated - observed)
            if missing:
                checks.append(
                    {
                        "code": f"generated_missing_{kind}",
                        "severity": "error",
                        "message": f"generated samples are missing {kind}",
                        "sample_ids": missing,
                    }
                )

        orphan_metadata = sorted(inventory["metadata"] - generated)
        if orphan_metadata:
            checks.append(
                {
                    "code": "metadata_without_generated_image",
                    "severity": "warning",
                    "message": "metadata samples do not have matching generated image",
                    "sample_ids": orphan_metadata,
                }
            )

    return {
        "counts": {kind: len(sample_ids) for kind, sample_ids in sorted(inventory.items())},
        "checks": checks,
    }


def load_manifest_index(manifest_csv: str | Path | None) -> dict[str, dict[str, Any]]:
    if not manifest_csv:
        return {}
    rows = read_csv(manifest_csv)
    index: dict[str, dict[str, Any]] = {}
    for row in rows:
        sample_id = row.get("sample_id")
        image_id = row.get("image_id")
        image_uri = row.get("image_uri")
        for key in (
            sample_id,
            image_id,
            Path(image_uri).stem if image_uri else None,
            Path(image_uri).name if image_uri else None,
        ):
            if key:
                index[str(key).lower()] = row
    return index


def load_metadata_index(metadata_dir: str | Path | None) -> dict[str, dict[str, Any]]:
    if not metadata_dir:
        return {}
    index: dict[str, dict[str, Any]] = {}
    for metadata_path in sorted(Path(metadata_dir).glob("*.json")):
        try:
            sample = read_json(metadata_path)
        except Exception:
            continue
        image_asset = sample.get("image_asset") if isinstance(sample.get("image_asset"), dict) else {}
        base_context = {
            "sample_id": sample.get("sample_id"),
            "image_id": image_asset.get("image_id"),
            "image_uri": image_asset.get("image_uri"),
            "source_type": image_asset.get("source_type"),
            "metadata_uri": str(metadata_path),
        }
        _index_uri(index, image_asset.get("image_uri"), base_context)
        for obj in sample.get("objects", []) if isinstance(sample.get("objects"), list) else []:
            if not isinstance(obj, dict):
                continue
            object_context = {
                **base_context,
                "object_id": obj.get("object_id"),
                "object_type": obj.get("object_type"),
            }
            crop = obj.get("crop") if isinstance(obj.get("crop"), dict) else {}
            detail = obj.get("geometry_detail") if isinstance(obj.get("geometry_detail"), dict) else {}
            _index_uri(index, crop.get("crop_uri"), object_context)
            _index_uri(index, detail.get("mask_uri"), object_context)
    return index


def _index_uri(index: dict[str, dict[str, Any]], uri: Any, context: dict[str, Any]) -> None:
    if not uri:
        return
    path = Path(str(uri))
    for key in (str(uri), path.name, path.stem):
        if key:
            index[str(key).lower()] = context


def infer_crop_source(path: Path) -> dict[str, Any]:
    stem = path.stem
    if not stem.lower().startswith("sample_"):
        return {}
    raw = stem[len("sample_") :]
    patterns = [
        (r"(?P<source>.+?)_bbox_person_(?P<object_index>\d+)$", "person"),
        (r"(?P<source>.+?)_(?P<object_type>person|bbox)_(?P<object_index>\d+)$", None),
        (r"(?P<source>.+?)_(?P<object_type>item|unique_id|auto|obj)_(?P<object_index>\d+)$", None),
        (r"(?P<source>.+?)_(?P<object_index>\d+)$", "unknown"),
    ]
    match = None
    object_type = None
    for pattern, fixed_object_type in patterns:
        match = re.match(pattern, raw, flags=re.I)
        if match:
            object_type = fixed_object_type or match.groupdict().get("object_type", "unknown")
            break
    if not match or object_type is None:
        return {}
    source_stem = match.group("source")
    return {
        "sample_id": f"sample_{source_stem}",
        "image_id": source_stem,
        "source_stem": source_stem,
        "source_image_name": f"{source_stem}.jpg",
        "object_type": object_type.lower(),
        "object_index": match.group("object_index"),
    }


def _manifest_row_for_asset(
    path: Path,
    manifest_index: dict[str, dict[str, Any]],
    crop_info: dict[str, Any] | None = None,
) -> dict[str, Any] | None:
    keys = [
        path.stem,
        path.name,
    ]
    if crop_info:
        keys.extend(
            [
                crop_info.get("sample_id"),
                crop_info.get("image_id"),
                crop_info.get("source_stem"),
                crop_info.get("source_image_name"),
            ]
        )
    for key in keys:
        if key and str(key).lower() in manifest_index:
            return manifest_index[str(key).lower()]
    return None


def review_asset(
    path: Path,
    *,
    asset_kind: str,
    manifest_index: dict[str, dict[str, Any]] | None = None,
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    rules = {**DEFAULT_ASSET_QC_RULES, **(rules or {})}
    manifest_index = manifest_index or {}
    crop_info = infer_crop_source(path) if asset_kind == "crop" else {}
    row = _manifest_row_for_asset(path, manifest_index, crop_info)
    issues: list[dict[str, str]] = []
    width: int | None = None
    height: int | None = None

    try:
        width, height = get_image_size(path)
    except Exception as exc:
        issues.append(_issue("image_unreadable", str(exc)))

    if width is not None and height is not None:
        if asset_kind == "crop":
            min_width = int(rules.get("min_crop_width", 4) or 0)
            min_height = int(rules.get("min_crop_height", 4) or 0)
            max_aspect = float(rules.get("max_crop_aspect_ratio", 20.0) or 0)
            if width < min_width or height < min_height:
                issues.append(_issue("tiny_crop", f"crop is too small: {width}x{height}"))
            if max_aspect and max(width / height, height / width) > max_aspect:
                issues.append(_issue("extreme_crop_aspect_ratio", f"crop aspect ratio is too extreme: {width}x{height}"))
        else:
            min_width = int(rules.get("min_image_width", 1) or 0)
            min_height = int(rules.get("min_image_height", 1) or 0)
            if width < min_width or height < min_height:
                issues.append(_issue("tiny_image", f"image is too small: {width}x{height}"))

    if asset_kind == "crop" and not row and bool(rules.get("warn_unmatched_crop_source", True)):
        issues.append(
            _issue(
                "crop_source_unmatched",
                "crop filename could not be matched back to manifest source frame",
                severity="warning",
            )
        )

    return {
        "asset_kind": asset_kind,
        "status": _status(issues),
        "issues": issues,
        "asset_uri": str(path),
        "resolved_asset_uri": str(path.resolve()),
        "file_name": path.name,
        "file_size_bytes": path.stat().st_size if path.exists() else None,
        "width": width,
        "height": height,
        "aspect_ratio": round(width / height, 6) if width and height else None,
        "sample_id": (row or {}).get("sample_id") or crop_info.get("sample_id"),
        "image_id": (row or {}).get("image_id") or crop_info.get("image_id"),
        "source_type": (row or {}).get("source_type"),
        "source_image_uri": (row or {}).get("image_uri"),
        "object_type": crop_info.get("object_type") or (row or {}).get("object_type"),
        "object_index": crop_info.get("object_index"),
    }


def summarize_assets(records: list[dict[str, Any]]) -> dict[str, Any]:
    status_counts = Counter(record.get("status") for record in records)
    kind_counts = Counter(record.get("asset_kind") for record in records)
    issue_counts = Counter(
        issue.get("code")
        for record in records
        for issue in record.get("issues", [])
    )
    return {
        "total_assets": len(records),
        "asset_kind_counts": dict(sorted(kind_counts.items())),
        "passed_assets": status_counts.get("passed", 0),
        "failed_assets": status_counts.get("failed", 0),
        "needs_human_review_assets": status_counts.get("needs_human_review", 0),
        "issue_counts": dict(sorted(issue_counts.items())),
    }


def _csv_row(record: dict[str, Any]) -> dict[str, Any]:
    issues = record.get("issues", [])
    return {
        **{field: record.get(field, "") for field in ASSET_QC_FIELDS},
        "issue_codes": ";".join(str(issue.get("code")) for issue in issues),
        "issue_messages": " | ".join(str(issue.get("message")) for issue in issues),
    }


def run_asset_qc(
    *,
    image_paths: list[str | Path] | None = None,
    crop_paths: list[str | Path] | None = None,
    manifest_csv: str | Path | None = None,
    metadata_dir: str | Path | None = None,
    output_dir: str | Path = "data/qc",
    rules: dict[str, Any] | None = None,
) -> dict[str, Any]:
    manifest_index = load_manifest_index(manifest_csv)
    metadata_index = load_metadata_index(metadata_dir)
    manifest_index.update(metadata_index)
    records = []
    image_files = scan_image_paths(image_paths or [])
    crop_files = scan_image_paths(crop_paths or [])
    for path in image_files:
        records.append(review_asset(path, asset_kind="image", manifest_index=manifest_index, rules=rules))
    for path in crop_files:
        records.append(review_asset(path, asset_kind="crop", manifest_index=manifest_index, rules=rules))
    inventory = review_generated_inventory(
        image_files=image_files,
        crop_files=crop_files,
        metadata_dir=metadata_dir,
    )
    inventory_issue_counts = Counter(check.get("code") for check in inventory.get("checks", []))

    run_id = f"asset_qc_{now_iso_shanghai().replace(':', '').replace('+', '_').replace('-', '')}"
    summary = summarize_assets(records)
    summary["inventory_issue_counts"] = dict(sorted(inventory_issue_counts.items()))
    summary["failed_inventory_checks"] = sum(
        1 for check in inventory.get("checks", []) if check.get("severity") == "error"
    )
    report = {
        "asset_qc_run_id": run_id,
        "created_time": now_iso_shanghai(),
        "image_paths": [str(path) for path in image_paths or []],
        "crop_paths": [str(path) for path in crop_paths or []],
        "manifest_csv": str(manifest_csv) if manifest_csv else None,
        "metadata_dir": str(metadata_dir) if metadata_dir else None,
        "summary": summary,
        "inventory": inventory,
        "assets": records,
    }
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    report_path = target_dir / f"{run_id}_report.json"
    csv_path = target_dir / f"{run_id}_assets.csv"
    write_json(report_path, report)
    write_csv(csv_path, [_csv_row(record) for record in records], ASSET_QC_FIELDS)
    report["report_path"] = str(report_path)
    report["asset_csv_path"] = str(csv_path)
    write_json(report_path, report)
    return report
